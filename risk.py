"""
risk.py – Risk analytics for the FX STIR trading sandbox.

All DV01s are computed via bump-and-reprice (finite difference).
DV01 convention: +1 bp shift in implied INR yield → ΔPV.

Buckets: near (≤ 2M), belly (3M–6M), far (9M–12M) – configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from curves import FXForwardCurve, ImpliedYieldCurve, yearfrac_to_label
from instruments import FXSwap, NDF, Steepener, Butterfly

# Type alias for any supported trade
Trade = Union[FXSwap, NDF, Steepener, Butterfly]

# Default bucket boundaries (year-fractions)
DEFAULT_BUCKETS = {"near": 2 / 12, "belly": 0.5, "far": 1.0}
# near: ≤ 2M,  belly: 2M < t ≤ 6M,  far: > 6M

# ===================================================================
# Single-trade helpers
# ===================================================================

def _trade_pv(trade: Trade, curve: FXForwardCurve) -> float:
    """Unified PV dispatcher."""
    if isinstance(trade, (FXSwap, Steepener, Butterfly)):
        return trade.pv(curve)
    elif isinstance(trade, NDF):
        return trade.pv(curve)
    raise TypeError(f"Unknown trade type: {type(trade)}")


def compute_dv01_fxswap(trade: FXSwap, curve: FXForwardCurve, bp: float = 1.0) -> float:
    """DV01 of an FX swap: PV change for +bp shift in implied yield."""
    pv_base = trade.pv(curve)
    curve_bumped = curve.shift_implied_yield(bp)
    pv_bumped = trade.pv(curve_bumped)
    return pv_bumped - pv_base


def compute_dv01(trade: Trade, curve: FXForwardCurve, bp: float = 1.0) -> float:
    """DV01 for any trade type (parallel +bp in implied yield)."""
    pv_base = _trade_pv(trade, curve)
    curve_bumped = curve.shift_implied_yield(bp)
    pv_bumped = _trade_pv(trade, curve_bumped)
    return pv_bumped - pv_base


def compute_key_rate_dv01(
    trade: Trade, curve: FXForwardCurve, bp: float = 1.0,
) -> Dict[float, float]:
    """Key-rate DV01: bump each tenor node individually by +bp.

    Returns {tenor_yearfrac: dv01_usd}.
    """
    pv_base = _trade_pv(trade, curve)
    kr = {}
    for i, t in enumerate(curve.tenors):
        curve_bumped = curve.bump_implied_yield(i, bp)
        pv_bumped = _trade_pv(trade, curve_bumped)
        kr[t] = pv_bumped - pv_base
    return kr


def compute_bucketed_dv01(
    kr_dv01: Dict[float, float],
    boundaries: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Aggregate key-rate DV01 into named buckets.

    Default buckets: near ≤ 2M, belly 2M–6M, far > 6M.
    """
    if boundaries is None:
        boundaries = DEFAULT_BUCKETS
    near_b = boundaries.get("near", 2 / 12)
    belly_b = boundaries.get("belly", 0.5)

    buckets = {"near": 0.0, "belly": 0.0, "far": 0.0}
    for t, dv01 in kr_dv01.items():
        if t <= near_b:
            buckets["near"] += dv01
        elif t <= belly_b:
            buckets["belly"] += dv01
        else:
            buckets["far"] += dv01
    return buckets


def compute_fwd_pts_dv01(trade: Trade, curve: FXForwardCurve, paise: float = 1.0) -> float:
    """DV01 in forward-points space: PV change for +1 paise parallel shift."""
    pv_base = _trade_pv(trade, curve)
    curve_bumped = curve.shift_fwd_points(paise)
    pv_bumped = _trade_pv(trade, curve_bumped)
    return pv_bumped - pv_base


# ===================================================================
# Portfolio-level risk
# ===================================================================

@dataclass
class PortfolioRisk:
    """Container for portfolio-level risk metrics."""
    pv: float
    dv01_yield: float          # parallel implied-yield DV01 (USD)
    dv01_fwd_pts: float        # parallel fwd-pts DV01 (USD)
    key_rate_dv01: Dict[float, float]
    bucketed_dv01: Dict[str, float]
    trade_pvs: Dict[str, float]
    trade_dv01s: Dict[str, float]
    trade_kr_dv01s: Dict[str, Dict[float, float]]


def portfolio_risk(
    trades: List[Tuple[str, Trade]],  # [(label, trade), ...]
    curve: FXForwardCurve,
    bucket_boundaries: Optional[Dict[str, float]] = None,
) -> PortfolioRisk:
    """Compute full risk suite for a portfolio of trades."""
    total_pv = 0.0
    total_dv01_yield = 0.0
    total_dv01_fwd = 0.0
    total_kr: Dict[float, float] = {}
    trade_pvs = {}
    trade_dv01s = {}
    trade_kr_dv01s = {}

    for label, trade in trades:
        pv = _trade_pv(trade, curve)
        dv01 = compute_dv01(trade, curve)
        dv01_fwd = compute_fwd_pts_dv01(trade, curve)
        kr = compute_key_rate_dv01(trade, curve)

        total_pv += pv
        total_dv01_yield += dv01
        total_dv01_fwd += dv01_fwd
        for t, v in kr.items():
            total_kr[t] = total_kr.get(t, 0.0) + v

        trade_pvs[label] = pv
        trade_dv01s[label] = dv01
        trade_kr_dv01s[label] = kr

    bucketed = compute_bucketed_dv01(total_kr, bucket_boundaries)

    return PortfolioRisk(
        pv=total_pv,
        dv01_yield=total_dv01_yield,
        dv01_fwd_pts=total_dv01_fwd,
        key_rate_dv01=total_kr,
        bucketed_dv01=bucketed,
        trade_pvs=trade_pvs,
        trade_dv01s=trade_dv01s,
        trade_kr_dv01s=trade_kr_dv01s,
    )


# ===================================================================
# Hedge ratio
# ===================================================================

def hedge_ratio(
    swap_a: FXSwap, swap_b: FXSwap, curve: FXForwardCurve,
) -> float:
    """DV01-neutral hedge ratio: how many units of swap_b to hedge swap_a.

    ratio = -DV01(A) / DV01(B)
    """
    dv01_a = compute_dv01_fxswap(swap_a, curve)
    dv01_b = compute_dv01_fxswap(swap_b, curve)
    if abs(dv01_b) < 1e-15:
        return float("inf")
    return -dv01_a / dv01_b


# ===================================================================
# Scenario P&L
# ===================================================================

def scenario_pnl(
    trades: List[Tuple[str, Trade]],
    base_curve: FXForwardCurve,
    shocked_curve: FXForwardCurve,
) -> Dict[str, float]:
    """P&L per trade and total from a scenario shock.

    P&L = PV(shocked) – PV(base).
    """
    pnls: Dict[str, float] = {}
    total = 0.0
    for label, trade in trades:
        pv_base = _trade_pv(trade, base_curve)
        pv_shocked = _trade_pv(trade, shocked_curve)
        pnl = pv_shocked - pv_base
        pnls[label] = pnl
        total += pnl
    pnls["TOTAL"] = total
    return pnls


# ===================================================================
# Carry / Roll-down aggregation
# ===================================================================

def portfolio_carry_rolldown(
    trades: List[Tuple[str, Trade]],
    curve: FXForwardCurve,
    holding_period: float,
) -> Dict[str, Dict[str, float]]:
    """Carry + roll-down for each trade and portfolio total."""
    results: Dict[str, Dict[str, float]] = {}
    total_carry = 0.0
    total_rolldown = 0.0

    for label, trade in trades:
        cr = trade.carry_and_rolldown(curve, holding_period)
        results[label] = cr
        total_carry += cr["carry_usd"]
        total_rolldown += cr["rolldown_usd"]

    results["TOTAL"] = {
        "carry_usd": total_carry,
        "rolldown_usd": total_rolldown,
        "total_usd": total_carry + total_rolldown,
    }
    return results


# ===================================================================
# Roll-down profile (for the learning module)
# ===================================================================

def rolldown_profile(
    curve: FXForwardCurve,
    holding_period: float,
    notional_usd: float = 1_000_000,
) -> pd.DataFrame:
    """Compute roll-down for a receiver (buy-sell) at each tenor on the curve.

    Returns a DataFrame with columns: tenor, tenor_label, rolldown_usd,
    rolldown_paise, implied_yield_bp_change.
    """
    rows = []
    for i, t in enumerate(curve.tenors):
        if t <= holding_period + 1e-6:
            continue  # skip tenors that mature within holding period
        swap = FXSwap(t, curve.forward_points_at(t), notional_usd, "buy_sell")
        cr = swap.carry_and_rolldown(curve, holding_period)

        # implied yield change from rolling
        iy_curve = curve.to_implied_yield_curve()
        r_now = iy_curve.zero_rate(t)
        r_aged = iy_curve.zero_rate(t - holding_period)
        iy_change_bp = (r_aged - r_now) * 10_000

        rows.append({
            "tenor": t,
            "tenor_label": yearfrac_to_label(t),
            "rolldown_usd": cr["rolldown_usd"],
            "carry_usd": cr["carry_usd"],
            "total_usd": cr["total_usd"],
            "rolldown_paise": cr.get("rolldown_paise_per_usd", 0),
            "implied_yield_change_bp": iy_change_bp,
        })
    return pd.DataFrame(rows)
