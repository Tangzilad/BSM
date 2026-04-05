"""
curves.py – FX forward-point curves, implied-yield curves, and Nelson-Siegel parametric curves.

All curves are immutable: shock / bump methods return new instances.

Terminology
-----------
- Forward points (fwd_pts): market-observable price of an FX swap in paise
  (1 paise = 0.01 INR).  fwd_pts > 0 means INR trades at a premium (depreciation).
- Implied yield: the annualised INR interest rate implied by CIP:
      r_INR ≈ r_USD + ln(F / S) / T   (continuous)
  where F = S + fwd_pts / 100.
- Tenors for FX STIR are short-dated: O/N … 12M.

Curve storage
-------------
* `ImpliedYieldCurve`  – tenor nodes + continuously-compounded zero rates,
  log-linear discount-factor interpolation.
* `FXForwardCurve`     – spot + tenor nodes + forward points (paise),
  linear interpolation on fwd_pts; derives implied yields via a USD curve.
* `NelsonSiegelCurve`  – parametric (β0, β1, β2, τ) for scenario generation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Standard FX STIR tenors (year fractions, ACT/365)
# ---------------------------------------------------------------------------
TENOR_LABELS: List[str] = [
    "O/N", "T/N", "S/N", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "12M",
]
TENOR_DAYS: Dict[str, int] = {
    "O/N": 1, "T/N": 2, "S/N": 3, "1W": 7, "2W": 14,
    "1M": 30, "2M": 61, "3M": 91, "6M": 182, "9M": 274, "12M": 365,
}
TENOR_YEARFRACS: Dict[str, float] = {k: v / 365.0 for k, v in TENOR_DAYS.items()}


def label_to_yearfrac(label: str) -> float:
    """Convert a tenor label like '3M' to a year fraction."""
    return TENOR_YEARFRACS[label]


def yearfrac_to_label(yf: float, tol: float = 1e-4) -> str:
    """Best-effort reverse lookup; returns '?.?Y' if no match."""
    for lbl, val in TENOR_YEARFRACS.items():
        if abs(val - yf) < tol:
            return lbl
    return f"{yf:.3f}Y"


# ===================================================================
# ImpliedYieldCurve  (primary analytical view)
# ===================================================================
@dataclass(frozen=True)
class ImpliedYieldCurve:
    """Continuously-compounded zero-rate curve with log-linear DF interpolation.

    Parameters
    ----------
    tenors : sequence of year-fractions  (e.g. [0.0027, 0.25, 0.5, 1.0])
    rates  : sequence of cc zero rates   (e.g. [0.065, 0.068, 0.070, 0.072])
    label  : human-readable name         (e.g. 'INR implied', 'TRY implied')
    """
    tenors: Tuple[float, ...]
    rates: Tuple[float, ...]
    label: str = ""

    def __post_init__(self):
        if len(self.tenors) != len(self.rates):
            raise ValueError("tenors and rates must have the same length")
        if len(self.tenors) < 2:
            raise ValueError("need at least 2 tenor nodes")
        # ensure sorted
        order = np.argsort(self.tenors)
        object.__setattr__(self, "tenors", tuple(np.array(self.tenors)[order]))
        object.__setattr__(self, "rates", tuple(np.array(self.rates)[order]))

    # -- core analytics ------------------------------------------------

    def zero_rate(self, t: float) -> float:
        """Interpolated continuously-compounded zero rate at tenor *t*."""
        if t <= 0:
            return self.rates[0]
        # log-linear on DF  ⟹  linear on  r·t
        ts = np.array(self.tenors)
        rt = np.array(self.rates) * ts          # r*t at each node
        rt_interp = float(np.interp(t, ts, rt))  # linear interp of r*t
        return rt_interp / t

    def discount_factor(self, t: float) -> float:
        """DF(t) = exp(-r(t)·t)."""
        if t <= 0:
            return 1.0
        return math.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simply-compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (df1 / df2 - 1.0) / (t2 - t1)

    def instantaneous_forward(self, t: float, dt: float = 1e-5) -> float:
        """Approximate instantaneous forward rate at t."""
        return self.forward_rate(max(t, dt / 2), t + dt)

    # -- shock methods (return new curve) ------------------------------

    def parallel_shift(self, bp: float) -> ImpliedYieldCurve:
        """Shift all rates by *bp* basis points."""
        shift = bp / 10_000
        return ImpliedYieldCurve(
            self.tenors,
            tuple(r + shift for r in self.rates),
            label=self.label,
        )

    def bump(self, idx: int, bp: float) -> ImpliedYieldCurve:
        """Bump a single tenor node by *bp* basis points."""
        shift = bp / 10_000
        new_rates = list(self.rates)
        new_rates[idx] += shift
        return ImpliedYieldCurve(self.tenors, tuple(new_rates), label=self.label)

    def twist(self, pivot: float, bp: float) -> ImpliedYieldCurve:
        """Linear twist: –bp at short end, +bp at long end, zero at *pivot*.

        The shock at tenor t is  bp * (t – pivot) / max_dist  (in bp).
        """
        ts = np.array(self.tenors)
        dists = ts - pivot
        max_dist = max(abs(dists.min()), abs(dists.max()), 1e-8)
        shocks = bp * dists / max_dist / 10_000
        return ImpliedYieldCurve(
            self.tenors,
            tuple(np.array(self.rates) + shocks),
            label=self.label,
        )

    def curvature_shock(self, hump_tenor: float, bp: float) -> ImpliedYieldCurve:
        """Butterfly / curvature shock centred on *hump_tenor*.

        Shape: Gaussian-like bump peaking at hump_tenor.
        """
        ts = np.array(self.tenors)
        sigma = (ts[-1] - ts[0]) / 4.0 or 0.25
        weights = np.exp(-0.5 * ((ts - hump_tenor) / sigma) ** 2)
        shocks = bp * weights / 10_000
        return ImpliedYieldCurve(
            self.tenors,
            tuple(np.array(self.rates) + shocks),
            label=self.label,
        )

    def roll_down(self, holding_period: float) -> ImpliedYieldCurve:
        """Return curve as seen after *holding_period* years have elapsed.

        Tenors shift left; rates re-read from the original curve at the
        new (shorter) tenors.  Tenors that would go ≤ 0 are dropped.
        """
        new_tenors = []
        new_rates = []
        for t in self.tenors:
            t_new = t - holding_period
            if t_new > 1e-6:
                new_tenors.append(t_new)
                new_rates.append(self.zero_rate(t_new))
        if len(new_tenors) < 2:
            # fallback: keep at least 2 nodes
            new_tenors = [1e-4, max(self.tenors) - holding_period]
            new_rates = [self.zero_rate(1e-4), self.zero_rate(new_tenors[-1])]
        return ImpliedYieldCurve(
            tuple(new_tenors), tuple(new_rates), label=self.label
        )

    # -- helpers -------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(self.tenors, self.rates))

    def tenor_labels(self) -> List[str]:
        return [yearfrac_to_label(t) for t in self.tenors]


# ===================================================================
# FXForwardCurve  (market-observable view)
# ===================================================================
@dataclass(frozen=True)
class FXForwardCurve:
    """USD/INR forward-point curve.

    Parameters
    ----------
    spot       : USD/INR spot rate (e.g. 83.50)
    tenors     : year-fractions
    fwd_points : forward points in **paise** (1 paise = 0.01 INR)
    usd_curve  : ImpliedYieldCurve for USD rates (needed for implied-yield derivation)
    basis_wedge: additional basis in paise applied uniformly (onshore/offshore dislocation)
    label      : e.g. 'Onshore' or 'Offshore NDF'
    """
    spot: float
    tenors: Tuple[float, ...]
    fwd_points: Tuple[float, ...]           # paise
    usd_curve: ImpliedYieldCurve
    basis_wedge: float = 0.0                # paise
    label: str = ""

    def __post_init__(self):
        if len(self.tenors) != len(self.fwd_points):
            raise ValueError("tenors and fwd_points must have same length")
        order = np.argsort(self.tenors)
        object.__setattr__(self, "tenors", tuple(np.array(self.tenors)[order]))
        object.__setattr__(self, "fwd_points", tuple(np.array(self.fwd_points)[order]))

    # -- core ----------------------------------------------------------

    def forward_points_at(self, t: float) -> float:
        """Interpolated forward points (paise) at tenor *t*, incl. basis."""
        base = float(np.interp(t, self.tenors, self.fwd_points))
        return base + self.basis_wedge

    def forward(self, t: float) -> float:
        """Outright forward rate at tenor *t*."""
        return self.spot + self.forward_points_at(t) / 100.0

    def implied_yield(self, t: float) -> float:
        """CIP-implied INR cc yield at tenor *t*.

        r_INR = r_USD + ln(F/S) / t
        """
        if t <= 0:
            return self.usd_curve.zero_rate(0)
        F = self.forward(t)
        r_usd = self.usd_curve.zero_rate(t)
        return r_usd + math.log(F / self.spot) / t

    def to_implied_yield_curve(self) -> ImpliedYieldCurve:
        """Derive an ImpliedYieldCurve from this forward-point curve."""
        rates = [self.implied_yield(t) for t in self.tenors]
        return ImpliedYieldCurve(self.tenors, tuple(rates), label=self.label)

    # -- shock methods -------------------------------------------------

    def with_basis(self, wedge_paise: float) -> FXForwardCurve:
        """Return new curve with a different basis wedge."""
        return FXForwardCurve(
            self.spot, self.tenors, self.fwd_points,
            self.usd_curve, basis_wedge=wedge_paise, label=self.label,
        )

    def shift_spot(self, pct: float) -> FXForwardCurve:
        """Shock spot by *pct* percent (e.g. –5 means INR depreciates 5%)."""
        new_spot = self.spot * (1.0 + pct / 100.0)
        return FXForwardCurve(
            new_spot, self.tenors, self.fwd_points,
            self.usd_curve, self.basis_wedge, self.label,
        )

    def shift_fwd_points(self, paise: float) -> FXForwardCurve:
        """Parallel shift all forward points by *paise*."""
        new_pts = tuple(p + paise for p in self.fwd_points)
        return FXForwardCurve(
            self.spot, self.tenors, new_pts,
            self.usd_curve, self.basis_wedge, self.label,
        )

    def bump_fwd_point(self, idx: int, paise: float) -> FXForwardCurve:
        """Bump a single tenor's forward point by *paise*."""
        new_pts = list(self.fwd_points)
        new_pts[idx] += paise
        return FXForwardCurve(
            self.spot, self.tenors, tuple(new_pts),
            self.usd_curve, self.basis_wedge, self.label,
        )

    def shift_implied_yield(self, bp: float) -> FXForwardCurve:
        """Bump the implied-yield curve by *bp* and recompute fwd points.

        Useful for computing yield-space DV01.
        """
        iy_curve = self.to_implied_yield_curve().parallel_shift(bp)
        new_pts = []
        for t in self.tenors:
            r_inr = iy_curve.zero_rate(t)
            r_usd = self.usd_curve.zero_rate(t)
            F = self.spot * math.exp((r_inr - r_usd) * t)
            new_pts.append((F - self.spot) * 100.0)  # paise
        return FXForwardCurve(
            self.spot, self.tenors, tuple(new_pts),
            self.usd_curve, self.basis_wedge, self.label,
        )

    def bump_implied_yield(self, idx: int, bp: float) -> FXForwardCurve:
        """Bump implied yield at a single tenor node by *bp* and recompute fwd pts."""
        iy_curve = self.to_implied_yield_curve().bump(idx, bp)
        new_pts = []
        for i, t in enumerate(self.tenors):
            r_inr = iy_curve.zero_rate(t)
            r_usd = self.usd_curve.zero_rate(t)
            F = self.spot * math.exp((r_inr - r_usd) * t)
            new_pts.append((F - self.spot) * 100.0)
        return FXForwardCurve(
            self.spot, self.tenors, tuple(new_pts),
            self.usd_curve, self.basis_wedge, self.label,
        )

    def twist_implied_yield(self, pivot: float, bp: float) -> FXForwardCurve:
        """Twist implied yields and recompute fwd pts."""
        iy_curve = self.to_implied_yield_curve().twist(pivot, bp)
        new_pts = []
        for t in self.tenors:
            r_inr = iy_curve.zero_rate(t)
            r_usd = self.usd_curve.zero_rate(t)
            F = self.spot * math.exp((r_inr - r_usd) * t)
            new_pts.append((F - self.spot) * 100.0)
        return FXForwardCurve(
            self.spot, self.tenors, tuple(new_pts),
            self.usd_curve, self.basis_wedge, self.label,
        )

    def curvature_shock_implied_yield(self, hump: float, bp: float) -> FXForwardCurve:
        """Curvature shock on implied yields, recompute fwd pts."""
        iy_curve = self.to_implied_yield_curve().curvature_shock(hump, bp)
        new_pts = []
        for t in self.tenors:
            r_inr = iy_curve.zero_rate(t)
            r_usd = self.usd_curve.zero_rate(t)
            F = self.spot * math.exp((r_inr - r_usd) * t)
            new_pts.append((F - self.spot) * 100.0)
        return FXForwardCurve(
            self.spot, self.tenors, tuple(new_pts),
            self.usd_curve, self.basis_wedge, self.label,
        )

    # -- helpers -------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {yearfrac_to_label(t): p for t, p in zip(self.tenors, self.fwd_points)}


# ===================================================================
# NelsonSiegelCurve  (parametric, for scenario generation)
# ===================================================================
@dataclass(frozen=True)
class NelsonSiegelCurve:
    """Nelson-Siegel parametric yield curve.

    y(t) = β0 + β1·(1 − e^{−t/τ})/(t/τ)
              + β2·((1 − e^{−t/τ})/(t/τ) − e^{−t/τ})

    β0 = level, β1 = slope (negative → upward-sloping), β2 = curvature, τ = decay.
    """
    beta0: float
    beta1: float
    beta2: float
    tau: float = 0.5

    def zero_rate(self, t: float) -> float:
        if t <= 1e-8:
            return self.beta0 + self.beta1
        x = t / self.tau
        ex = math.exp(-x)
        factor1 = (1.0 - ex) / x
        factor2 = factor1 - ex
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2

    def discount_factor(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return math.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (df1 / df2 - 1.0) / (t2 - t1)

    def to_implied_yield_curve(self, tenors: Sequence[float]) -> ImpliedYieldCurve:
        """Sample the parametric curve at given tenors."""
        rates = tuple(self.zero_rate(t) for t in tenors)
        return ImpliedYieldCurve(tuple(tenors), rates, label="Nelson-Siegel")


# ===================================================================
# Helpers: build curves from forward points + USD curve
# ===================================================================

def fwd_points_from_implied_yield(
    spot: float, tenors: Sequence[float],
    inr_rates: Sequence[float], usd_rates: Sequence[float],
) -> List[float]:
    """Compute forward points (paise) from implied INR yields and USD yields."""
    pts = []
    for t, r_inr, r_usd in zip(tenors, inr_rates, usd_rates):
        F = spot * math.exp((r_inr - r_usd) * t)
        pts.append((F - spot) * 100.0)
    return pts


def build_fx_curve_from_csv(
    spot: float,
    tenors: Sequence[float],
    fwd_points_paise: Sequence[float],
    usd_curve: ImpliedYieldCurve,
    basis_wedge: float = 0.0,
    label: str = "",
) -> FXForwardCurve:
    """Convenience constructor from raw CSV-style data."""
    return FXForwardCurve(
        spot=spot,
        tenors=tuple(tenors),
        fwd_points=tuple(fwd_points_paise),
        usd_curve=usd_curve,
        basis_wedge=basis_wedge,
        label=label,
    )
