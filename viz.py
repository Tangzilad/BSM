"""
viz.py – Plotly visualisations for the FX STIR trading sandbox.

All functions return plotly.graph_objects.Figure instances for embedding
in Streamlit via st.plotly_chart().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from curves import FXForwardCurve, ImpliedYieldCurve, yearfrac_to_label


# ===================================================================
# Colour palette
# ===================================================================
COLORS = {
    "base": "#1f77b4",
    "shocked": "#d62728",
    "near": "#2ca02c",
    "belly": "#ff7f0e",
    "far": "#9467bd",
    "carry": "#17becf",
    "rolldown": "#bcbd22",
    "total": "#e377c2",
    "positive": "#2ca02c",
    "negative": "#d62728",
}


# ===================================================================
# Yield curve chart
# ===================================================================

def plot_yield_curves(
    base_curve: FXForwardCurve,
    shocked_curve: Optional[FXForwardCurve] = None,
    mode: str = "implied_yield",
    title: str = "Implied Yield Curve",
) -> go.Figure:
    """Plot base (and optionally shocked) curves.

    mode: 'implied_yield' (bp) or 'fwd_points' (paise).
    """
    fig = go.Figure()
    tenors = list(base_curve.tenors)
    labels = [yearfrac_to_label(t) for t in tenors]

    if mode == "implied_yield":
        iy_base = base_curve.to_implied_yield_curve()
        y_base = [iy_base.zero_rate(t) * 10_000 for t in tenors]  # bp
        y_label = "Implied Yield (bp)"
        if shocked_curve:
            iy_shock = shocked_curve.to_implied_yield_curve()
            y_shock = [iy_shock.zero_rate(t) * 10_000 for t in tenors]
    else:
        y_base = [base_curve.forward_points_at(t) for t in tenors]
        y_label = "Forward Points (paise)"
        if shocked_curve:
            y_shock = [shocked_curve.forward_points_at(t) for t in tenors]

    fig.add_trace(go.Scatter(
        x=labels, y=y_base, mode="lines+markers", name="Base",
        line=dict(color=COLORS["base"], width=2),
        marker=dict(size=8),
    ))

    if shocked_curve:
        fig.add_trace(go.Scatter(
            x=labels, y=y_shock, mode="lines+markers", name="Shocked",
            line=dict(color=COLORS["shocked"], width=2, dash="dash"),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title=title, xaxis_title="Tenor", yaxis_title=y_label,
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ===================================================================
# DV01 bar chart (key-rate, stacked by trade + bucketed)
# ===================================================================

def plot_dv01_bars(
    trade_kr_dv01s: Dict[str, Dict[float, float]],
    bucket_boundaries: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Stacked bar chart of key-rate DV01 by tenor, one colour per trade.

    Also shows bucketed DV01 as a secondary grouped bar.
    """
    from risk import compute_bucketed_dv01

    # Collect all tenors
    all_tenors = sorted(set(
        t for kr in trade_kr_dv01s.values() for t in kr
    ))
    labels = [yearfrac_to_label(t) for t in all_tenors]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.65, 0.35],
        subplot_titles=("Key-Rate DV01 by Tenor", "Bucketed DV01"),
    )

    # Key-rate bars (stacked)
    for trade_label, kr in trade_kr_dv01s.items():
        y_vals = [kr.get(t, 0) for t in all_tenors]
        fig.add_trace(go.Bar(
            x=labels, y=y_vals, name=trade_label,
        ), row=1, col=1)

    # Bucketed bars
    total_kr = {}
    for kr in trade_kr_dv01s.values():
        for t, v in kr.items():
            total_kr[t] = total_kr.get(t, 0) + v
    bucketed = compute_bucketed_dv01(total_kr, bucket_boundaries)

    bucket_colors = [COLORS["near"], COLORS["belly"], COLORS["far"]]
    fig.add_trace(go.Bar(
        x=list(bucketed.keys()),
        y=list(bucketed.values()),
        marker_color=bucket_colors,
        name="Bucketed",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        barmode="stack", template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    fig.update_yaxes(title_text="DV01 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="DV01 (USD)", row=1, col=2)
    return fig


# ===================================================================
# P&L heatmap (parallel × twist)
# ===================================================================

def plot_pnl_heatmap(
    trades: list,
    base_curve: FXForwardCurve,
    parallel_range: Sequence[float] = range(-50, 55, 5),
    twist_range: Sequence[float] = range(-30, 35, 5),
    pivot: float = 0.25,
) -> go.Figure:
    """2-D heatmap: x = parallel shift (bp), y = twist (bp), cell = portfolio P&L."""
    from risk import _trade_pv

    par_vals = list(parallel_range)
    tw_vals = list(twist_range)
    z = np.zeros((len(tw_vals), len(par_vals)))

    for i, tw in enumerate(tw_vals):
        for j, par in enumerate(par_vals):
            shocked = base_curve.shift_implied_yield(par)
            if tw != 0:
                shocked = shocked.twist_implied_yield(pivot, tw)
            total_pnl = 0.0
            for _label, trade in trades:
                pv_base = _trade_pv(trade, base_curve)
                pv_shocked = _trade_pv(trade, shocked)
                total_pnl += pv_shocked - pv_base
            z[i, j] = total_pnl

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{p:+d}" for p in par_vals],
        y=[f"{t:+d}" for t in tw_vals],
        colorscale="RdYlGn",
        colorbar=dict(title="P&L (USD)"),
        hovertemplate="Parallel: %{x} bp<br>Twist: %{y} bp<br>P&L: %{z:,.0f} USD<extra></extra>",
    ))
    fig.update_layout(
        title="Portfolio P&L Heatmap (Parallel × Twist)",
        xaxis_title="Parallel Shift (bp)",
        yaxis_title="Twist (bp)",
        template="plotly_white",
        height=500,
    )
    return fig


# ===================================================================
# Waterfall chart (trade-level P&L)
# ===================================================================

def plot_waterfall(trade_pnls: Dict[str, float]) -> go.Figure:
    """Waterfall chart of trade-level P&L contributions."""
    labels = []
    values = []
    measures = []

    for label, pnl in trade_pnls.items():
        if label == "TOTAL":
            continue
        labels.append(label)
        values.append(pnl)
        measures.append("relative")

    # Total bar
    labels.append("TOTAL")
    values.append(trade_pnls.get("TOTAL", sum(v for v in trade_pnls.values() if v != trade_pnls.get("TOTAL"))))
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        increasing=dict(marker_color=COLORS["positive"]),
        decreasing=dict(marker_color=COLORS["negative"]),
        totals=dict(marker_color=COLORS["base"]),
        textposition="outside",
        text=[f"{v:+,.0f}" for v in values],
    ))
    fig.update_layout(
        title="P&L Waterfall by Trade",
        yaxis_title="P&L (USD)",
        template="plotly_white",
        height=420,
    )
    return fig


# ===================================================================
# Carry & Roll-down charts
# ===================================================================

def plot_rolldown_profile(
    df: pd.DataFrame,
    title: str = "Roll-Down Profile (Buy-Sell / Receiver)",
) -> go.Figure:
    """Bar chart of roll-down + carry by tenor from a rolldown_profile DataFrame."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["tenor_label"], y=df["carry_usd"], name="Carry",
        marker_color=COLORS["carry"],
    ))
    fig.add_trace(go.Bar(
        x=df["tenor_label"], y=df["rolldown_usd"], name="Roll-Down",
        marker_color=COLORS["rolldown"],
    ))
    fig.update_layout(
        barmode="stack", title=title,
        xaxis_title="Tenor", yaxis_title="P&L (USD per $1M notional)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_rolldown_comparison(
    profiles: Dict[str, pd.DataFrame],
    title: str = "Roll-Down Comparison Across Curves",
) -> go.Figure:
    """Side-by-side roll-down profiles for multiple curves (e.g. INR vs TRY)."""
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (curve_name, df) in enumerate(profiles.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Bar(
            x=df["tenor_label"],
            y=df["total_usd"],
            name=f"{curve_name} (carry + roll-down)",
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group", title=title,
        xaxis_title="Tenor", yaxis_title="Total P&L (USD per $1M notional)",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_carry_rolldown_decomposition(
    cr_results: Dict[str, Dict[str, float]],
    title: str = "Carry vs Roll-Down by Trade",
) -> go.Figure:
    """Stacked bar: carry vs roll-down for each trade in portfolio."""
    labels = [k for k in cr_results if k != "TOTAL"]
    carry_vals = [cr_results[k]["carry_usd"] for k in labels]
    rolldown_vals = [cr_results[k]["rolldown_usd"] for k in labels]

    # Add total
    labels.append("TOTAL")
    carry_vals.append(cr_results["TOTAL"]["carry_usd"])
    rolldown_vals.append(cr_results["TOTAL"]["rolldown_usd"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=carry_vals, name="Carry", marker_color=COLORS["carry"]))
    fig.add_trace(go.Bar(x=labels, y=rolldown_vals, name="Roll-Down", marker_color=COLORS["rolldown"]))
    fig.update_layout(
        barmode="stack", title=title,
        xaxis_title="Trade", yaxis_title="P&L (USD)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ===================================================================
# Macro notebook panel
# ===================================================================

def format_macro_notebook(scenario) -> str:
    """Return a Markdown string for the macro notebook panel."""
    lines = [scenario.explain(), "", "---", "", "**Macro Drivers:**"]
    for d in scenario.driver_checklist():
        check = "x" if d["relevant"] else " "
        lines.append(f"- [{check}] {d['driver']}")
    return "\n".join(lines)
