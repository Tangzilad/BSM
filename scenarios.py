"""
scenarios.py – Scenario engine and pre-built EM macro scenario library.

A Scenario encodes a combination of shocks (parallel, twist, curvature,
FX spot, NDF basis, liquidity) plus a narrative explaining the macro driver.

Applying a scenario to a base FXForwardCurve produces a shocked curve;
the risk module then computes trade-level and portfolio P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from curves import FXForwardCurve


@dataclass
class Scenario:
    """A named scenario with shocks and a macro narrative.

    Parameters in `params` dict
    ---------------------------
    parallel_shift_bp   : parallel shift to implied yields (bp)
    twist_bp            : twist shock (bp); positive = steepening
    twist_pivot         : pivot tenor for twist (year-frac, default 0.25)
    curvature_bp        : curvature / butterfly shock (bp)
    curvature_hump      : tenor of hump peak (year-frac, default 0.25)
    fx_spot_shock_pct   : spot shock in % (negative = INR depreciation)
    basis_shock_paise   : NDF basis widening in paise
    liquidity_shock_bp  : bid-offer widening in bp (informational)
    """
    name: str
    narrative: str
    tags: List[str] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)

    # -- apply ---------------------------------------------------------

    def apply(self, base_curve: FXForwardCurve) -> FXForwardCurve:
        """Apply all shocks and return a new FXForwardCurve."""
        curve = base_curve

        # 1. FX spot shock
        if self.params.get("fx_spot_shock_pct", 0):
            curve = curve.shift_spot(self.params["fx_spot_shock_pct"])

        # 2. Parallel shift in implied yield
        if self.params.get("parallel_shift_bp", 0):
            curve = curve.shift_implied_yield(self.params["parallel_shift_bp"])

        # 3. Twist
        if self.params.get("twist_bp", 0):
            pivot = self.params.get("twist_pivot", 0.25)
            curve = curve.twist_implied_yield(pivot, self.params["twist_bp"])

        # 4. Curvature
        if self.params.get("curvature_bp", 0):
            hump = self.params.get("curvature_hump", 0.25)
            curve = curve.curvature_shock_implied_yield(hump, self.params["curvature_bp"])

        # 5. Basis wedge
        if self.params.get("basis_shock_paise", 0):
            new_wedge = curve.basis_wedge + self.params["basis_shock_paise"]
            curve = curve.with_basis(new_wedge)

        return curve

    # -- auto-generated explanation ------------------------------------

    def explain(self) -> str:
        """Generate a human-readable explanation of the scenario shocks."""
        lines = [f"**{self.name}**", "", self.narrative, "", "Shocks applied:"]

        p = self.params
        if p.get("fx_spot_shock_pct"):
            direction = "depreciation" if p["fx_spot_shock_pct"] > 0 else "appreciation"
            lines.append(f"  • FX spot: {p['fx_spot_shock_pct']:+.1f}% ({direction})")

        if p.get("parallel_shift_bp"):
            direction = "higher" if p["parallel_shift_bp"] > 0 else "lower"
            lines.append(f"  • Implied yields: {p['parallel_shift_bp']:+.0f} bp parallel ({direction})")

        if p.get("twist_bp"):
            direction = "steepening" if p["twist_bp"] > 0 else "flattening"
            pivot = p.get("twist_pivot", 0.25)
            lines.append(f"  • Twist: {p['twist_bp']:+.0f} bp around {pivot:.2f}Y ({direction})")

        if p.get("curvature_bp"):
            hump = p.get("curvature_hump", 0.25)
            lines.append(f"  • Curvature: {p['curvature_bp']:+.0f} bp hump at {hump:.2f}Y")

        if p.get("basis_shock_paise"):
            lines.append(f"  • NDF basis: {p['basis_shock_paise']:+.0f} paise wider")

        if p.get("liquidity_shock_bp"):
            lines.append(f"  • Liquidity / bid-offer: {p['liquidity_shock_bp']:+.0f} bp wider")

        # P&L intuition
        lines.append("")
        lines.append("Expected P&L direction:")
        if p.get("parallel_shift_bp", 0) > 0:
            lines.append("  → Receivers (buy-sell) lose; payers (sell-buy) gain")
        elif p.get("parallel_shift_bp", 0) < 0:
            lines.append("  → Receivers (buy-sell) gain; payers (sell-buy) lose")
        if p.get("twist_bp", 0) > 0:
            lines.append("  → Steepeners gain; flatteners lose")
        elif p.get("twist_bp", 0) < 0:
            lines.append("  → Flatteners gain; steepeners lose")
        if p.get("basis_shock_paise", 0) > 0:
            lines.append("  → Offshore NDF positions see wider onshore/offshore spread")

        return "\n".join(lines)

    def driver_checklist(self) -> List[Dict[str, Any]]:
        """Return a list of macro drivers with relevance flags."""
        drivers = [
            {"driver": "Risk-off / global sentiment", "relevant": "risk-off" in self.tags},
            {"driver": "Oil prices / CAD impact", "relevant": "oil" in self.tags},
            {"driver": "Portfolio flows (FII/FDI)", "relevant": "flows" in self.tags},
            {"driver": "RBI intervention / regulation", "relevant": "rbi" in self.tags or "regulation" in self.tags},
            {"driver": "Monetary policy surprise", "relevant": "policy" in self.tags},
            {"driver": "Global dollar strength (DXY)", "relevant": "dxy" in self.tags or "risk-off" in self.tags},
            {"driver": "Onshore/offshore dislocation", "relevant": "basis" in self.tags},
            {"driver": "Liquidity / market functioning", "relevant": "liquidity" in self.tags},
        ]
        return drivers


# ===================================================================
# Pre-built scenario library
# ===================================================================

SCENARIO_LIBRARY: List[Scenario] = [
    Scenario(
        name="Taper Tantrum Risk-Off",
        narrative=(
            "Global risk-off triggered by unexpected Fed tightening signal. "
            "EM currencies sell off sharply; offshore NDF market leads the move "
            "with wider basis.  Implied yields spike as hedging demand surges. "
            "Onshore/offshore divergence widens as RBI intervenes selectively."
        ),
        tags=["risk-off", "flows", "basis", "liquidity", "dxy"],
        params={
            "parallel_shift_bp": 50,
            "twist_bp": 30,
            "twist_pivot": 0.25,
            "fx_spot_shock_pct": 5.0,    # INR depreciates 5%
            "basis_shock_paise": 50,
            "liquidity_shock_bp": 5,
        },
    ),
    Scenario(
        name="COVID Stress",
        narrative=(
            "Pandemic-style shock: extreme volatility, flight to USD, EM FX "
            "collapses.  NDF market leads onshore with aggressive INR selling. "
            "Hedging costs spike as counterparty risk increases.  Forward points "
            "blow out across the curve."
        ),
        tags=["risk-off", "flows", "basis", "liquidity", "dxy"],
        params={
            "parallel_shift_bp": 75,
            "twist_bp": 15,
            "twist_pivot": 0.25,
            "curvature_bp": 20,
            "curvature_hump": 0.25,
            "fx_spot_shock_pct": 8.0,
            "basis_shock_paise": 80,
            "liquidity_shock_bp": 10,
        },
    ),
    Scenario(
        name="Regulatory Liquidity Shock",
        narrative=(
            "RBI introduces new regulations restricting offshore NDF "
            "warehousing or imposing position limits.  Onshore/offshore "
            "spread widens sharply as market-making capacity shrinks.  "
            "Underlying rates are mostly stable but basis explodes."
        ),
        tags=["regulation", "rbi", "basis", "liquidity"],
        params={
            "parallel_shift_bp": 10,
            "basis_shock_paise": 100,
            "liquidity_shock_bp": 15,
        },
    ),
    Scenario(
        name="Policy Surprise – Hawkish",
        narrative=(
            "RBI delivers a surprise rate hike or hawkish forward guidance. "
            "Front-end implied yields jump; curve bear-flattens as short rates "
            "reprice aggressively.  INR appreciates modestly on carry appeal."
        ),
        tags=["policy", "rbi"],
        params={
            "parallel_shift_bp": 30,
            "twist_bp": -25,          # flattening
            "twist_pivot": 0.08,      # pivot near 1M
            "fx_spot_shock_pct": -1.0, # INR appreciates
        },
    ),
    Scenario(
        name="Policy Surprise – Dovish",
        narrative=(
            "RBI cuts rates or signals extended accommodation.  Front-end "
            "implied yields drop; curve bull-steepens.  INR weakens modestly "
            "as carry attractiveness diminishes."
        ),
        tags=["policy", "rbi"],
        params={
            "parallel_shift_bp": -25,
            "twist_bp": 20,           # steepening
            "twist_pivot": 0.08,
            "fx_spot_shock_pct": 1.0,  # INR depreciates
        },
    ),
]


def get_scenario_library() -> List[Scenario]:
    """Return the pre-built scenario library."""
    return list(SCENARIO_LIBRARY)


def get_scenario_by_name(name: str) -> Optional[Scenario]:
    """Look up a scenario by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for s in SCENARIO_LIBRARY:
        if name_lower in s.name.lower():
            return s
    return None
