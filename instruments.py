"""
instruments.py – FX swap, NDF, steepener/flattener, and butterfly instruments
for USD/INR FX STIR trading.

All instruments are priced in USD terms.  Forward points are in paise.

Conventions
-----------
* **Buy-sell (receive implied yield):** buy USD spot, sell USD forward.
  You earn the INR implied yield (INR rate > USD rate → positive carry).
  Analogous to a "receiver" in rates.  Profits when implied yields fall /
  forward points narrow.

* **Sell-buy (pay implied yield):** sell USD spot, buy USD forward.
  You pay the INR implied yield.  Analogous to a "payer".  Profits when
  implied yields rise / forward points widen.

* **Steepener:** receive short-tenor implied yield + pay long-tenor implied
  yield.  Profits when the implied-yield curve steepens.

* **Flattener:** pay short-tenor implied yield + receive long-tenor implied
  yield.  Profits when the curve flattens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from curves import FXForwardCurve, ImpliedYieldCurve

# ===================================================================
# FX Swap
# ===================================================================

@dataclass
class FXSwap:
    """A single-tenor FX swap on USD/INR.

    Parameters
    ----------
    tenor        : year-fraction (e.g. 0.25 for 3M)
    trade_fwd_pts: forward points at which the trade was done (paise)
    notional_usd : USD notional (positive)
    direction    : 'buy_sell' (receive implied yield) or 'sell_buy' (pay)
    label        : human-readable tag
    """
    tenor: float
    trade_fwd_pts: float          # paise, at trade inception
    notional_usd: float
    direction: str                # 'buy_sell' or 'sell_buy'
    label: str = ""

    def __post_init__(self):
        if self.direction not in ("buy_sell", "sell_buy"):
            raise ValueError("direction must be 'buy_sell' or 'sell_buy'")

    @property
    def sign(self) -> float:
        """Direction multiplier: +1 for buy-sell, -1 for sell-buy."""
        return 1.0 if self.direction == "buy_sell" else -1.0

    # -- pricing -------------------------------------------------------

    def pv(self, curve: FXForwardCurve) -> float:
        """Mark-to-market PV in USD.

        Buy-sell entered at trade_fwd_pts, market now at curve fwd_pts:
          PV = sign * notional * (trade_pts - market_pts) / 100 / spot * DF_usd(T)

        (trade_pts - market_pts) because buy-sell profits when fwd pts FALL.
        """
        market_pts = curve.forward_points_at(self.tenor)
        pts_diff = (self.trade_fwd_pts - market_pts) / 100.0  # INR per USD
        df_usd = curve.usd_curve.discount_factor(self.tenor)
        return self.sign * self.notional_usd * pts_diff * df_usd / curve.spot

    def pv_components(self, curve: FXForwardCurve) -> Dict[str, float]:
        """Decompose PV into spot and forward-point components."""
        # Full PV
        full_pv = self.pv(curve)
        # Spot-only component: bump spot by 1 paise, measure change
        curve_spot_up = curve.shift_spot(0.01 / curve.spot * 100)
        pv_spot_up = self.pv(curve_spot_up)
        spot_delta_per_paise = pv_spot_up - full_pv
        # Forward-point component is the residual
        return {
            "total_pv": full_pv,
            "spot_delta_per_paise": spot_delta_per_paise,
            "fwd_pts_component": full_pv,  # at inception spot legs cancel
        }

    # -- carry & roll-down ---------------------------------------------

    def carry_and_rolldown(
        self, curve: FXForwardCurve, holding_period: float,
    ) -> Dict[str, float]:
        """Decompose expected P&L into carry and roll-down (in USD).

        Carry
        -----
        Net interest earned over the holding period, assuming rates unchanged.
        For a buy-sell (receive INR yield, pay USD yield):
          carry ≈ sign * notional * (r_INR - r_USD) * dt / spot_conversion

        Roll-down
        ---------
        MTM change from the swap aging on an unchanged curve.
        After dt, the swap that was tenor T is now tenor (T - dt).
        Market fwd_pts at (T - dt) differ from those at T on a non-flat curve.
          rolldown = PV(T-dt on same curve, same trade_pts) - PV(T on same curve)
        On an upward-sloping fwd-pts curve, a buy-sell (receiver) benefits
        because shorter tenors have fewer fwd_pts → the position gains.
        """
        dt = holding_period
        T = self.tenor

        # -- carry --
        iy = curve.to_implied_yield_curve()
        r_inr = iy.zero_rate(T)
        r_usd = curve.usd_curve.zero_rate(T)
        # Interest differential carry (in INR per USD, annualised → over dt)
        carry_inr = (r_inr - r_usd) * dt
        carry_usd = self.sign * self.notional_usd * carry_inr / curve.spot

        # -- roll-down --
        pv_now = self.pv(curve)
        remaining = T - dt
        if remaining < 1e-6:
            # swap matures within holding period → roll-down = convergence to 0
            rolldown_usd = -pv_now
        else:
            # reprice with shorter tenor but same curve and same trade_fwd_pts
            aged_swap = FXSwap(
                remaining, self.trade_fwd_pts, self.notional_usd,
                self.direction, self.label,
            )
            pv_aged = aged_swap.pv(curve)
            rolldown_usd = pv_aged - pv_now

        return {
            "carry_usd": carry_usd,
            "rolldown_usd": rolldown_usd,
            "total_usd": carry_usd + rolldown_usd,
            "carry_paise_per_usd": carry_inr * 100,  # paise
            "rolldown_paise_per_usd": rolldown_usd * curve.spot / self.notional_usd * 100 if self.notional_usd else 0,
        }


# ===================================================================
# NDF (Non-Deliverable Forward)
# ===================================================================

@dataclass
class NDF:
    """USD/INR Non-Deliverable Forward.

    Cash-settled: at fixing, settlement = Notional * (fixing_rate - trade_fwd) / fixing_rate
    (in USD, for USD notional).

    Before fixing, PV uses market forward instead of fixing rate.

    Parameters
    ----------
    tenor          : year-fraction to settlement
    trade_forward  : outright forward rate at trade inception (INR per USD)
    notional       : notional amount
    notional_ccy   : 'USD' or 'INR'
    direction      : 'buy_usd' (long USD/short INR) or 'sell_usd' (short USD/long INR)
    basis_wedge    : extra paise added to NDF forward vs onshore (offshore dislocation)
    label          : human-readable tag
    """
    tenor: float
    trade_forward: float
    notional: float
    notional_ccy: str = "USD"
    direction: str = "buy_usd"
    basis_wedge: float = 0.0
    label: str = ""

    def __post_init__(self):
        if self.notional_ccy not in ("USD", "INR"):
            raise ValueError("notional_ccy must be 'USD' or 'INR'")
        if self.direction not in ("buy_usd", "sell_usd"):
            raise ValueError("direction must be 'buy_usd' or 'sell_usd'")

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "buy_usd" else -1.0

    def pv(self, curve: FXForwardCurve, usd_discount_curve: Optional[ImpliedYieldCurve] = None) -> float:
        """PV in USD.

        Uses the NDF forward curve (which may include its own basis).
        An additional basis_wedge is applied on top.

        For USD notional, buy_usd:
          PV = notional * (market_fwd - trade_fwd) / market_fwd * DF_usd(T)

        For INR notional, buy_usd:
          PV = notional_INR / trade_fwd - notional_INR / market_fwd, discounted
        """
        market_fwd = curve.forward(self.tenor) + self.basis_wedge / 100.0
        usd_crv = usd_discount_curve or curve.usd_curve
        df_usd = usd_crv.discount_factor(self.tenor)

        if self.notional_ccy == "USD":
            settlement_usd = self.notional * (market_fwd - self.trade_forward) / market_fwd
        else:
            # INR notional
            settlement_usd = self.notional * (1.0 / self.trade_forward - 1.0 / market_fwd)

        return self.sign * settlement_usd * df_usd

    def fx_spot_sensitivity(self, curve: FXForwardCurve, bump_pct: float = 0.01) -> float:
        """Delta to a spot move (per 1% spot change), in USD."""
        pv_base = self.pv(curve)
        curve_bumped = curve.shift_spot(bump_pct)
        pv_bumped = self.pv(curve_bumped)
        return (pv_bumped - pv_base) / bump_pct

    def forward_point_sensitivity(self, curve: FXForwardCurve, bump_paise: float = 1.0) -> float:
        """Delta to a 1-paise parallel shift in forward points, in USD."""
        pv_base = self.pv(curve)
        curve_bumped = curve.shift_fwd_points(bump_paise)
        pv_bumped = self.pv(curve_bumped)
        return pv_bumped - pv_base

    def carry_and_rolldown(
        self, curve: FXForwardCurve, holding_period: float,
    ) -> Dict[str, float]:
        """Carry + roll-down decomposition for NDF."""
        dt = holding_period
        T = self.tenor

        # Carry: interest differential over holding period
        iy = curve.to_implied_yield_curve()
        r_inr = iy.zero_rate(T)
        r_usd = curve.usd_curve.zero_rate(T)
        notional_usd = self.notional if self.notional_ccy == "USD" else self.notional / curve.spot
        carry_usd = self.sign * notional_usd * (r_inr - r_usd) * dt

        # Roll-down
        pv_now = self.pv(curve)
        remaining = T - dt
        if remaining < 1e-6:
            rolldown_usd = -pv_now
        else:
            aged = NDF(
                remaining, self.trade_forward, self.notional,
                self.notional_ccy, self.direction, self.basis_wedge, self.label,
            )
            rolldown_usd = aged.pv(curve) - pv_now

        return {
            "carry_usd": carry_usd,
            "rolldown_usd": rolldown_usd,
            "total_usd": carry_usd + rolldown_usd,
        }


# ===================================================================
# Steepener / Flattener  (2-leg, DV01-neutral)
# ===================================================================

@dataclass
class Steepener:
    """DV01-neutral 2-leg curve trade using FX swaps.

    Steepener = receive short implied yield + pay long implied yield.
    Flattener = pay short implied yield + receive long implied yield.

    Notionals are sized so that the DV01 of each leg offsets, leaving
    the portfolio approximately flat to parallel moves.

    Parameters
    ----------
    short_tenor   : year-fraction for short leg
    long_tenor    : year-fraction for long leg
    notional_usd  : base notional (applied to short leg; long leg sized by DV01 ratio)
    direction     : 'steepener' or 'flattener'
    trade_short_pts, trade_long_pts : forward points at inception (paise)
    label         : human-readable tag
    """
    short_tenor: float
    long_tenor: float
    notional_usd: float
    direction: str
    trade_short_pts: float
    trade_long_pts: float
    label: str = ""

    def __post_init__(self):
        if self.direction not in ("steepener", "flattener"):
            raise ValueError("direction must be 'steepener' or 'flattener'")

    def build_legs(self, curve: FXForwardCurve) -> Tuple[FXSwap, FXSwap]:
        """Construct the two FX swap legs, DV01-weighted.

        Returns (short_leg, long_leg).
        """
        # Determine directions
        if self.direction == "steepener":
            short_dir, long_dir = "buy_sell", "sell_buy"
        else:
            short_dir, long_dir = "sell_buy", "buy_sell"

        # Build unit-notional swaps for DV01 ratio
        from risk import compute_dv01_fxswap
        unit_short = FXSwap(self.short_tenor, self.trade_short_pts, 1_000_000, short_dir)
        unit_long = FXSwap(self.long_tenor, self.trade_long_pts, 1_000_000, long_dir)
        dv01_short = abs(compute_dv01_fxswap(unit_short, curve))
        dv01_long = abs(compute_dv01_fxswap(unit_long, curve))

        # Size long leg so DV01s offset
        if dv01_long > 1e-12:
            ratio = dv01_short / dv01_long
        else:
            ratio = 1.0
        long_notional = self.notional_usd * ratio

        short_leg = FXSwap(
            self.short_tenor, self.trade_short_pts, self.notional_usd,
            short_dir, label=f"{self.label} short",
        )
        long_leg = FXSwap(
            self.long_tenor, self.trade_long_pts, long_notional,
            long_dir, label=f"{self.label} long",
        )
        return short_leg, long_leg

    def pv(self, curve: FXForwardCurve) -> float:
        short_leg, long_leg = self.build_legs(curve)
        return short_leg.pv(curve) + long_leg.pv(curve)

    def carry_and_rolldown(
        self, curve: FXForwardCurve, holding_period: float,
    ) -> Dict[str, float]:
        short_leg, long_leg = self.build_legs(curve)
        cr_short = short_leg.carry_and_rolldown(curve, holding_period)
        cr_long = long_leg.carry_and_rolldown(curve, holding_period)
        return {
            "carry_usd": cr_short["carry_usd"] + cr_long["carry_usd"],
            "rolldown_usd": cr_short["rolldown_usd"] + cr_long["rolldown_usd"],
            "total_usd": cr_short["total_usd"] + cr_long["total_usd"],
            "short_leg": cr_short,
            "long_leg": cr_long,
        }


# ===================================================================
# Butterfly  (3-leg, DV01-neutral)
# ===================================================================

@dataclass
class Butterfly:
    """DV01-neutral 3-leg butterfly on the implied-yield curve.

    Standard: sell wings (pay short + pay long implied yield),
    buy belly (receive belly implied yield).
    Profits when curvature increases (belly richens).

    Parameters
    ----------
    short_tenor, belly_tenor, long_tenor : year-fractions
    notional_usd : base notional (belly leg)
    trade_short_pts, trade_belly_pts, trade_long_pts : fwd pts at inception
    label : tag
    """
    short_tenor: float
    belly_tenor: float
    long_tenor: float
    notional_usd: float
    trade_short_pts: float
    trade_belly_pts: float
    trade_long_pts: float
    label: str = ""

    def build_legs(self, curve: FXForwardCurve) -> Tuple[FXSwap, FXSwap, FXSwap]:
        """Build 3 legs: pay short, receive belly, pay long.

        Wing notionals sized so combined wing DV01 = belly DV01.
        Split equally between short and long wings.
        """
        from risk import compute_dv01_fxswap

        belly_swap = FXSwap(self.belly_tenor, self.trade_belly_pts,
                            self.notional_usd, "buy_sell", f"{self.label} belly")
        dv01_belly = abs(compute_dv01_fxswap(belly_swap, curve))

        unit_short = FXSwap(self.short_tenor, self.trade_short_pts, 1_000_000, "sell_buy")
        unit_long = FXSwap(self.long_tenor, self.trade_long_pts, 1_000_000, "sell_buy")
        dv01_unit_short = abs(compute_dv01_fxswap(unit_short, curve))
        dv01_unit_long = abs(compute_dv01_fxswap(unit_long, curve))

        # Target: wing_notional_short * dv01_unit_short + wing_notional_long * dv01_unit_long = dv01_belly
        # Split equally in DV01 terms
        half_dv01 = dv01_belly / 2.0
        not_short = (half_dv01 / dv01_unit_short * 1_000_000) if dv01_unit_short > 1e-12 else self.notional_usd
        not_long = (half_dv01 / dv01_unit_long * 1_000_000) if dv01_unit_long > 1e-12 else self.notional_usd

        short_swap = FXSwap(self.short_tenor, self.trade_short_pts, not_short,
                            "sell_buy", f"{self.label} short wing")
        long_swap = FXSwap(self.long_tenor, self.trade_long_pts, not_long,
                           "sell_buy", f"{self.label} long wing")
        return short_swap, belly_swap, long_swap

    def pv(self, curve: FXForwardCurve) -> float:
        legs = self.build_legs(curve)
        return sum(leg.pv(curve) for leg in legs)

    def carry_and_rolldown(
        self, curve: FXForwardCurve, holding_period: float,
    ) -> Dict[str, float]:
        legs = self.build_legs(curve)
        crs = [leg.carry_and_rolldown(curve, holding_period) for leg in legs]
        return {
            "carry_usd": sum(c["carry_usd"] for c in crs),
            "rolldown_usd": sum(c["rolldown_usd"] for c in crs),
            "total_usd": sum(c["total_usd"] for c in crs),
            "legs": crs,
        }
