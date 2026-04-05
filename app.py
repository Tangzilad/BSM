"""
app.py – Streamlit UI for the EM STIR Trading Sandbox.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from curves import (
    TENOR_DAYS, TENOR_LABELS, TENOR_YEARFRACS,
    FXForwardCurve, ImpliedYieldCurve, NelsonSiegelCurve,
    label_to_yearfrac, yearfrac_to_label,
)
from instruments import FXSwap, NDF, Steepener, Butterfly
from risk import (
    Trade, compute_dv01, compute_key_rate_dv01, compute_bucketed_dv01,
    compute_fwd_pts_dv01, portfolio_risk, scenario_pnl,
    portfolio_carry_rolldown, rolldown_profile, hedge_ratio,
)
from scenarios import Scenario, get_scenario_library, get_scenario_by_name
from viz import (
    plot_yield_curves, plot_dv01_bars, plot_pnl_heatmap,
    plot_waterfall, plot_rolldown_profile, plot_rolldown_comparison,
    plot_carry_rolldown_decomposition, format_macro_notebook,
)

# ===================================================================
# Page config
# ===================================================================
st.set_page_config(
    page_title="EM STIR Trading Sandbox",
    page_icon="📈",
    layout="wide",
)

# ===================================================================
# Session state init
# ===================================================================
if "trades" not in st.session_state:
    st.session_state.trades: List[Tuple[str, Trade]] = []
if "learning_mode" not in st.session_state:
    st.session_state.learning_mode = False


# ===================================================================
# Helpers: build default curves
# ===================================================================

@st.cache_data
def load_default_usd_curve() -> ImpliedYieldCurve:
    df = pd.read_csv("data/sample_usd_curve.csv")
    tenors = tuple(d / 365.0 for d in df["tenor_days"])
    rates = tuple(df["rate"])
    return ImpliedYieldCurve(tenors, rates, label="USD OIS")


@st.cache_data
def load_default_inr_fwd_points() -> pd.DataFrame:
    return pd.read_csv("data/sample_inr_fwd_points.csv")


@st.cache_data
def load_default_try_fwd_points() -> pd.DataFrame:
    return pd.read_csv("data/sample_try_fwd_points.csv")


def build_fx_curve(
    spot: float, fwd_pts_df: pd.DataFrame, usd_curve: ImpliedYieldCurve,
    pts_col: str = "onshore_fwd_pts_paise", basis: float = 0.0, label: str = "",
) -> FXForwardCurve:
    tenors = tuple(d / 365.0 for d in fwd_pts_df["tenor_days"])
    pts = tuple(fwd_pts_df[pts_col])
    return FXForwardCurve(spot, tenors, pts, usd_curve, basis_wedge=basis, label=label)


def build_try_curve(
    spot_try: float, df: pd.DataFrame, usd_curve: ImpliedYieldCurve,
) -> FXForwardCurve:
    tenors = tuple(d / 365.0 for d in df["tenor_days"])
    pts = tuple(df["fwd_pts_kurus"])  # kurus are like paise for TRY
    return FXForwardCurve(spot_try, tenors, pts, usd_curve, label="USD/TRY")


# ===================================================================
# Learning mode content
# ===================================================================

LEARNING = {
    "buy_sell": """
**Buy-Sell (Receive Implied Yield)**
You buy USD spot and sell USD forward. You earn the INR implied yield
(since INR rates > USD rates, the forward points are positive — INR
depreciates forward). This is like being a "receiver" in rates: you
benefit when implied yields **fall** (forward points narrow).
""",
    "sell_buy": """
**Sell-Buy (Pay Implied Yield)**
You sell USD spot and buy USD forward. You pay the INR implied yield.
Analogous to a "payer". You benefit when implied yields **rise**
(forward points widen).
""",
    "steepener": """
**Steepener**
Receive short-tenor implied yield + pay long-tenor implied yield.
Profits when the implied-yield curve **steepens** (short rates fall
relative to long rates). DV01-neutral to parallel moves.
""",
    "flattener": """
**Flattener**
Pay short-tenor + receive long-tenor. Profits when the curve **flattens**.
""",
    "dv01": """
**DV01 (Dollar Value of 1 Basis Point)**
The P&L change for a 1 bp parallel shift in implied yields.
Computed by bumping all yield nodes +1 bp and repricing.
- Buy-sell (receiver): **negative** DV01 (rates up → you lose)
- Sell-buy (payer): **positive** DV01 (rates up → you gain)
""",
    "key_rate_dv01": """
**Key-Rate DV01**
DV01 when bumping only a single tenor node. Shows where on the curve
your risk is concentrated. Sum of all key-rate DV01s ≈ parallel DV01.
""",
    "rolldown": """
**Carry & Roll-Down**
- **Carry**: net interest earned over a holding period (INR yield − USD yield)
  for a buy-sell position.
- **Roll-down**: MTM gain from the swap ageing on an unchanged curve.
  A 3M swap becomes a 2M swap after 1M. On an upward-sloping implied-yield
  curve, the 2M rate is lower → the position gains (for a receiver).
- Steeper curves (e.g. TRY >> INR) produce more carry + roll-down.
""",
    "ndf": """
**NDF (Non-Deliverable Forward)**
Cash-settled forward: at fixing, settlement =
  Notional × (Fixing − Trade_Fwd) / Fixing  (in USD, for USD notional).
Before fixing, PV uses the market forward. The **basis wedge** captures
onshore/offshore dislocations (NDF forward ≠ onshore forward).
""",
    "basis": """
**Onshore/Offshore Basis**
The difference between onshore deliverable forward and offshore NDF forward
(in paise). Widens during stress (EM risk-off, regulation changes).
A positive basis wedge means the NDF implies more INR depreciation than onshore.
""",
}


def show_learning(key: str):
    if st.session_state.learning_mode and key in LEARNING:
        st.info(LEARNING[key], icon="📖")


# ===================================================================
# Sidebar
# ===================================================================

with st.sidebar:
    st.title("EM STIR Sandbox")
    st.session_state.learning_mode = st.toggle("Learning Mode", value=st.session_state.learning_mode)

    st.header("Market Data")
    spot = st.number_input("USD/INR Spot", value=83.50, step=0.10, format="%.2f")
    basis_wedge = st.number_input("NDF Basis Wedge (paise)", value=0.0, step=1.0, format="%.1f")

    # Build curves
    usd_curve = load_default_usd_curve()
    inr_df = load_default_inr_fwd_points()
    onshore_curve = build_fx_curve(spot, inr_df, usd_curve, "onshore_fwd_pts_paise", 0.0, "Onshore")
    offshore_curve = build_fx_curve(spot, inr_df, usd_curve, "offshore_fwd_pts_paise", basis_wedge, "Offshore NDF")

    st.markdown("---")

    # ---------------------------------------------------------------
    # Trade Builder
    # ---------------------------------------------------------------
    st.header("Trade Builder")
    trade_type = st.selectbox("Instrument", ["FX Swap", "NDF", "Steepener", "Butterfly"])

    if trade_type == "FX Swap":
        show_learning("buy_sell")
        show_learning("sell_buy")
        col1, col2 = st.columns(2)
        with col1:
            direction = st.selectbox("Direction", ["buy_sell", "sell_buy"],
                                     format_func=lambda x: "Buy-Sell (Receive)" if x == "buy_sell" else "Sell-Buy (Pay)")
        with col2:
            tenor_lbl = st.selectbox("Tenor", TENOR_LABELS[3:], index=5)  # default 3M
        tenor_yf = label_to_yearfrac(tenor_lbl)
        market_pts = onshore_curve.forward_points_at(tenor_yf)
        trade_pts = st.number_input("Trade Fwd Pts (paise)", value=round(market_pts, 1), step=0.5, format="%.1f")
        notional = st.number_input("Notional USD", value=10_000_000, step=1_000_000, format="%d")
        label = st.text_input("Label", value=f"{'Rcv' if direction == 'buy_sell' else 'Pay'} {tenor_lbl}")

        if st.button("Add FX Swap"):
            trade = FXSwap(tenor_yf, trade_pts, notional, direction, label)
            st.session_state.trades.append((label, trade))
            st.success(f"Added: {label}")

    elif trade_type == "NDF":
        show_learning("ndf")
        show_learning("basis")
        col1, col2 = st.columns(2)
        with col1:
            ndf_dir = st.selectbox("Direction", ["buy_usd", "sell_usd"],
                                   format_func=lambda x: "Buy USD" if x == "buy_usd" else "Sell USD")
        with col2:
            tenor_lbl = st.selectbox("Tenor", TENOR_LABELS[5:], index=2)
        tenor_yf = label_to_yearfrac(tenor_lbl)
        market_fwd = offshore_curve.forward(tenor_yf)
        trade_fwd = st.number_input("Trade Forward", value=round(market_fwd, 4), step=0.01, format="%.4f")
        notional_ccy = st.selectbox("Notional Currency", ["USD", "INR"])
        notional = st.number_input("Notional", value=10_000_000, step=1_000_000, format="%d")
        ndf_basis = st.number_input("Extra Basis (paise)", value=0.0, step=1.0, format="%.1f")
        label = st.text_input("Label", value=f"NDF {ndf_dir.replace('_', ' ').title()} {tenor_lbl}")

        if st.button("Add NDF"):
            trade = NDF(tenor_yf, trade_fwd, notional, notional_ccy, ndf_dir, ndf_basis, label)
            st.session_state.trades.append((label, trade))
            st.success(f"Added: {label}")

    elif trade_type == "Steepener":
        show_learning("steepener")
        show_learning("flattener")
        col1, col2 = st.columns(2)
        with col1:
            steep_dir = st.selectbox("Direction", ["steepener", "flattener"])
        with col2:
            short_lbl = st.selectbox("Short Tenor", TENOR_LABELS[5:8], index=0)
            long_lbl = st.selectbox("Long Tenor", TENOR_LABELS[8:], index=2)
        short_yf = label_to_yearfrac(short_lbl)
        long_yf = label_to_yearfrac(long_lbl)
        notional = st.number_input("Notional USD (short leg)", value=10_000_000, step=1_000_000, format="%d")
        short_pts = onshore_curve.forward_points_at(short_yf)
        long_pts = onshore_curve.forward_points_at(long_yf)
        label = st.text_input("Label", value=f"{short_lbl}/{long_lbl} {steep_dir.title()}")

        if st.button("Add Steepener/Flattener"):
            trade = Steepener(short_yf, long_yf, notional, steep_dir, short_pts, long_pts, label)
            st.session_state.trades.append((label, trade))
            st.success(f"Added: {label}")

    elif trade_type == "Butterfly":
        col1, col2, col3 = st.columns(3)
        with col1:
            short_lbl = st.selectbox("Short Wing", TENOR_LABELS[5:7], index=0)
        with col2:
            belly_lbl = st.selectbox("Belly", TENOR_LABELS[7:9], index=0)
        with col3:
            long_lbl = st.selectbox("Long Wing", TENOR_LABELS[9:], index=1)
        notional = st.number_input("Notional USD (belly)", value=10_000_000, step=1_000_000, format="%d")
        s_yf = label_to_yearfrac(short_lbl)
        b_yf = label_to_yearfrac(belly_lbl)
        l_yf = label_to_yearfrac(long_lbl)
        label = st.text_input("Label", value=f"{short_lbl}/{belly_lbl}/{long_lbl} Fly")

        if st.button("Add Butterfly"):
            trade = Butterfly(
                s_yf, b_yf, l_yf, notional,
                onshore_curve.forward_points_at(s_yf),
                onshore_curve.forward_points_at(b_yf),
                onshore_curve.forward_points_at(l_yf),
                label,
            )
            st.session_state.trades.append((label, trade))
            st.success(f"Added: {label}")

    # -- Portfolio table -----------------------------------------------
    st.markdown("---")
    st.header("Portfolio")
    if st.session_state.trades:
        for i, (lbl, t) in enumerate(st.session_state.trades):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{i+1}. {lbl} ({type(t).__name__})")
            with col2:
                if st.button("✕", key=f"rm_{i}"):
                    st.session_state.trades.pop(i)
                    st.rerun()
        if st.button("Clear All"):
            st.session_state.trades = []
            st.rerun()
    else:
        st.caption("No trades yet. Use the Trade Builder above.")


# ===================================================================
# Main area: tabs
# ===================================================================

# Determine which curve to use for pricing
# FX swaps use onshore; NDFs use offshore
def get_pricing_curve(trade: Trade) -> FXForwardCurve:
    if isinstance(trade, NDF):
        return offshore_curve
    return onshore_curve

tab_risk, tab_scenario, tab_heatmap, tab_curves, tab_rolldown = st.tabs([
    "Risk Dashboard", "Scenario Runner", "P&L Heatmap", "Curves", "Roll-Down Lab",
])

# ===================================================================
# Tab 1: Risk Dashboard
# ===================================================================
with tab_risk:
    st.header("Risk Dashboard")
    show_learning("dv01")
    show_learning("key_rate_dv01")

    if not st.session_state.trades:
        st.info("Add trades in the sidebar to see risk metrics.")
    else:
        # Compute portfolio risk using onshore curve (primary)
        pr = portfolio_risk(st.session_state.trades, onshore_curve)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio PV", f"${pr.pv:,.0f}")
        col2.metric("Yield DV01 (1bp)", f"${pr.dv01_yield:,.0f}")
        col3.metric("Fwd Pts DV01 (1ps)", f"${pr.dv01_fwd_pts:,.0f}")
        col4.metric("# Trades", len(st.session_state.trades))

        # Bucketed DV01
        st.subheader("Bucketed DV01")
        buck_cols = st.columns(3)
        for i, (bucket, val) in enumerate(pr.bucketed_dv01.items()):
            buck_cols[i].metric(f"{bucket.title()}", f"${val:,.0f}")

        # Trade-level table
        st.subheader("Trade-Level Risk")
        rows = []
        for lbl in pr.trade_pvs:
            rows.append({
                "Trade": lbl,
                "PV (USD)": f"{pr.trade_pvs[lbl]:,.0f}",
                "DV01 (USD)": f"{pr.trade_dv01s[lbl]:,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Key-rate DV01 chart
        st.subheader("Key-Rate DV01")
        fig_dv01 = plot_dv01_bars(pr.trade_kr_dv01s)
        st.plotly_chart(fig_dv01, use_container_width=True)

        # CSV export
        risk_df = pd.DataFrame(rows)
        csv_buf = io.StringIO()
        risk_df.to_csv(csv_buf, index=False)
        st.download_button("Download Risk CSV", csv_buf.getvalue(), "risk_report.csv", "text/csv")


# ===================================================================
# Tab 2: Scenario Runner
# ===================================================================
with tab_scenario:
    st.header("Scenario Runner")

    scenario_mode = st.radio("Mode", ["Library", "Custom"], horizontal=True)

    if scenario_mode == "Library":
        library = get_scenario_library()
        scenario_name = st.selectbox("Select Scenario", [s.name for s in library])
        scenario = get_scenario_by_name(scenario_name)
    else:
        st.subheader("Custom Scenario")
        sc_name = st.text_input("Scenario Name", "Custom Shock")
        sc_narrative = st.text_area("Narrative", "User-defined scenario.")
        col1, col2, col3 = st.columns(3)
        with col1:
            par_bp = st.slider("Parallel Shift (bp)", -100, 100, 0, 5)
            twist_bp = st.slider("Twist (bp)", -50, 50, 0, 5)
        with col2:
            curv_bp = st.slider("Curvature (bp)", -30, 30, 0, 5)
            fx_pct = st.slider("FX Spot Shock (%)", -10.0, 10.0, 0.0, 0.5)
        with col3:
            basis_ps = st.slider("Basis Shock (paise)", -100, 100, 0, 5)
            liq_bp = st.slider("Liquidity Shock (bp)", 0, 20, 0, 1)

        scenario = Scenario(
            name=sc_name, narrative=sc_narrative,
            tags=["custom"],
            params={
                "parallel_shift_bp": par_bp,
                "twist_bp": twist_bp,
                "twist_pivot": 0.25,
                "curvature_bp": curv_bp,
                "curvature_hump": 0.25,
                "fx_spot_shock_pct": fx_pct,
                "basis_shock_paise": basis_ps,
                "liquidity_shock_bp": liq_bp,
            },
        )

    if scenario:
        # Apply scenario
        shocked_onshore = scenario.apply(onshore_curve)
        shocked_offshore = scenario.apply(offshore_curve)

        # Macro notebook
        st.markdown(format_macro_notebook(scenario))

        # Curve comparison
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_yield_curves(onshore_curve, shocked_onshore, mode="implied_yield",
                                  title="Implied Yield: Base vs Shocked"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                plot_yield_curves(onshore_curve, shocked_onshore, mode="fwd_points",
                                  title="Forward Points: Base vs Shocked"),
                use_container_width=True,
            )

        # P&L
        if st.session_state.trades:
            st.subheader("Scenario P&L")
            pnls = scenario_pnl(st.session_state.trades, onshore_curve, shocked_onshore)
            pnl_rows = [{"Trade": k, "P&L (USD)": f"{v:+,.0f}"} for k, v in pnls.items()]
            st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True, hide_index=True)

            # Waterfall
            st.plotly_chart(plot_waterfall(pnls), use_container_width=True)

            # Export
            pnl_df = pd.DataFrame([{"Trade": k, "PnL_USD": v} for k, v in pnls.items()])
            csv_buf = io.StringIO()
            pnl_df.to_csv(csv_buf, index=False)
            st.download_button("Download P&L CSV", csv_buf.getvalue(), "scenario_pnl.csv", "text/csv")
        else:
            st.info("Add trades to see scenario P&L.")


# ===================================================================
# Tab 3: P&L Heatmap
# ===================================================================
with tab_heatmap:
    st.header("P&L Heatmap (Parallel × Twist)")

    if not st.session_state.trades:
        st.info("Add trades to generate the heatmap.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            par_min = st.number_input("Parallel min (bp)", value=-50, step=5)
            par_max = st.number_input("Parallel max (bp)", value=50, step=5)
        with col2:
            tw_min = st.number_input("Twist min (bp)", value=-30, step=5)
            tw_max = st.number_input("Twist max (bp)", value=30, step=5)
        with col3:
            step = st.number_input("Step (bp)", value=5, min_value=1, max_value=25)

        if st.button("Generate Heatmap"):
            with st.spinner("Computing P&L grid..."):
                fig = plot_pnl_heatmap(
                    st.session_state.trades, onshore_curve,
                    parallel_range=range(int(par_min), int(par_max) + 1, int(step)),
                    twist_range=range(int(tw_min), int(tw_max) + 1, int(step)),
                )
                st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 4: Curves
# ===================================================================
with tab_curves:
    st.header("Curve Viewer")

    curve_view = st.radio("View", ["Implied Yield (bp)", "Forward Points (paise)"], horizontal=True)
    mode = "implied_yield" if "Implied" in curve_view else "fwd_points"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Onshore")
        st.plotly_chart(
            plot_yield_curves(onshore_curve, mode=mode, title="Onshore USD/INR"),
            use_container_width=True,
        )
    with col2:
        st.subheader("Offshore NDF")
        st.plotly_chart(
            plot_yield_curves(offshore_curve, mode=mode, title="Offshore NDF"),
            use_container_width=True,
        )

    # Show curve data table
    st.subheader("Curve Data")
    iy_onshore = onshore_curve.to_implied_yield_curve()
    iy_offshore = offshore_curve.to_implied_yield_curve()
    curve_rows = []
    for i, t in enumerate(onshore_curve.tenors):
        lbl = yearfrac_to_label(t)
        curve_rows.append({
            "Tenor": lbl,
            "Onshore Fwd Pts (ps)": f"{onshore_curve.forward_points_at(t):.1f}",
            "Offshore Fwd Pts (ps)": f"{offshore_curve.forward_points_at(t):.1f}",
            "Basis (ps)": f"{offshore_curve.forward_points_at(t) - onshore_curve.forward_points_at(t):.1f}",
            "Onshore Impl Yield (bp)": f"{iy_onshore.zero_rate(t)*10000:.1f}",
            "Offshore Impl Yield (bp)": f"{iy_offshore.zero_rate(t)*10000:.1f}",
            "USD Rate (bp)": f"{usd_curve.zero_rate(t)*10000:.1f}",
        })
    st.dataframe(pd.DataFrame(curve_rows), use_container_width=True, hide_index=True)


# ===================================================================
# Tab 5: Roll-Down Lab
# ===================================================================
with tab_rolldown:
    st.header("Roll-Down Lab")
    show_learning("rolldown")

    col1, col2 = st.columns(2)
    with col1:
        holding_label = st.select_slider(
            "Holding Period",
            options=["1W", "2W", "1M", "2M", "3M", "6M"],
            value="1M",
        )
        holding_yf = label_to_yearfrac(holding_label)

    with col2:
        notional_rd = st.number_input("Notional (USD)", value=1_000_000, step=100_000,
                                      format="%d", key="rd_notional")

    # INR roll-down profile
    st.subheader("INR Implied Yield Curve – Roll-Down Profile")
    df_inr = rolldown_profile(onshore_curve, holding_yf, notional_rd)
    if not df_inr.empty:
        st.plotly_chart(
            plot_rolldown_profile(df_inr, f"INR Roll-Down ({holding_label} hold, ${notional_rd/1e6:.1f}M)"),
            use_container_width=True,
        )

    # TRY comparison
    st.subheader("INR vs TRY Roll-Down Comparison")
    if st.session_state.learning_mode:
        st.info(
            "**Why compare INR and TRY?**\n\n"
            "TRY has a much steeper implied-yield curve (rates ~25-35%) vs INR (~6-7%). "
            "Steeper curves produce more roll-down P&L: a receiver (buy-sell) earns more "
            "from the position ageing to a lower-rate point on the curve.\n\n"
            "This is why high-yielding EM FX swaps are popular carry trades — the roll-down "
            "adds significantly to the interest differential carry.",
            icon="📖",
        )

    try:
        try_df = load_default_try_fwd_points()
        try_curve = build_try_curve(32.50, try_df, usd_curve)
        df_try = rolldown_profile(try_curve, holding_yf, notional_rd)

        if not df_inr.empty and not df_try.empty:
            st.plotly_chart(
                plot_rolldown_comparison({"INR": df_inr, "TRY": df_try},
                                         f"Roll-Down: INR vs TRY ({holding_label} hold)"),
                use_container_width=True,
            )
    except Exception as e:
        st.warning(f"Could not load TRY data: {e}")

    # Portfolio carry/roll-down decomposition
    if st.session_state.trades:
        st.subheader("Portfolio Carry & Roll-Down Decomposition")
        cr = portfolio_carry_rolldown(st.session_state.trades, onshore_curve, holding_yf)
        st.plotly_chart(
            plot_carry_rolldown_decomposition(cr, f"Carry vs Roll-Down ({holding_label} holding period)"),
            use_container_width=True,
        )

        # Table
        cr_rows = []
        for lbl, vals in cr.items():
            cr_rows.append({
                "Trade": lbl,
                "Carry (USD)": f"{vals['carry_usd']:+,.0f}",
                "Roll-Down (USD)": f"{vals['rolldown_usd']:+,.0f}",
                "Total (USD)": f"{vals['total_usd']:+,.0f}",
            })
        st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)
