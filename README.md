# EM STIR Trading Sandbox

Interactive trading sandbox for FX STIR (Short-Term Interest Rate) trading, focused on **USD/INR FX swaps** and **NDFs**. Build positions, compute risk, run macro scenarios, and understand carry + roll-down — all offline with no external APIs.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What It Does

### Instruments
- **FX Swaps** (buy-sell / sell-buy): priced in forward points (paise), with implied INR yield derivation
- **NDFs**: cash-settled, with basis wedge for onshore/offshore dislocation
- **Steepeners / Flatteners**: DV01-neutral 2-leg curve trades
- **Butterflies**: 3-leg curvature trades

### Risk Metrics
- **PV** via forward-points MTM
- **DV01**: parallel +1bp in implied yield → ΔPV
- **Key-rate DV01**: per-tenor bucketed risk (near ≤2M / belly 3M–6M / far 9M–12M)
- **Carry & Roll-Down**: decomposed P&L from interest differential and curve ageing

### Scenarios
Five pre-built EM macro scenarios with narratives:
1. Taper Tantrum Risk-Off
2. COVID Stress
3. Regulatory Liquidity Shock
4. Policy Surprise – Hawkish
5. Policy Surprise – Dovish

Plus custom scenario builder with parallel/twist/curvature/FX/basis shocks.

### Visualisations (Plotly)
- Implied yield and forward-point curves (base vs shocked)
- Key-rate DV01 stacked bars + bucketed DV01
- P&L heatmap (parallel × twist grid)
- Waterfall chart of trade-level P&L
- Roll-down profiles and INR vs TRY comparison

## Tenors

FX STIR tenors up to 1 year: O/N, T/N, S/N, 1W, 2W, 1M, 2M, 3M, 6M, 9M, 12M.

## Data Format

CSV templates in `data/`:

| File | Columns |
|------|---------|
| `sample_usd_curve.csv` | tenor_label, tenor_days, rate |
| `sample_inr_fwd_points.csv` | tenor_label, tenor_days, onshore_fwd_pts_paise, offshore_fwd_pts_paise |
| `sample_try_fwd_points.csv` | tenor_label, tenor_days, fwd_pts_kurus |
| `sample_trades.csv` | type, direction, tenor_label, notional_usd, trade_fwd_pts_paise, ... |

Replace these CSVs with your own data to use real market levels.

## Conventions

| Term | Meaning |
|------|---------|
| **Buy-sell** | Buy USD spot, sell USD forward → receive INR implied yield (like a "receiver") |
| **Sell-buy** | Sell USD spot, buy USD forward → pay INR implied yield (like a "payer") |
| **Steepener** | Receive short + pay long → profits when curve steepens |
| **Flattener** | Pay short + receive long → profits when curve flattens |
| **Forward points** | F − S in paise (1 paise = 0.01 INR) |
| **Implied yield** | CIP-derived: r_INR = r_USD + ln(F/S) / T |
| **Basis wedge** | NDF forward − onshore forward (in paise); widens in stress |

## Running Tests

```bash
pytest tests/test_pricing.py -v
```

## Project Structure

```
curves.py        – Curve construction & interpolation
instruments.py   – FX swap, NDF, steepener, butterfly
risk.py          – DV01, key-rate DV01, carry/roll-down
scenarios.py     – Scenario engine + 5 pre-built scenarios
viz.py           – Plotly visualisations
app.py           – Streamlit UI
data/            – CSV templates
tests/           – pytest suite
```

## How to Extend

### Adding a new instrument
1. Create a class in `instruments.py` with `pv(curve)` and `carry_and_rolldown(curve, dt)` methods
2. Update `risk._trade_pv()` to dispatch to it
3. Add a section in `app.py` Trade Builder

### Adding a new scenario
```python
from scenarios import Scenario, SCENARIO_LIBRARY

SCENARIO_LIBRARY.append(Scenario(
    name="Your Scenario",
    narrative="What happens and why.",
    tags=["risk-off", "basis"],
    params={
        "parallel_shift_bp": 25,
        "fx_spot_shock_pct": 3.0,
        "basis_shock_paise": 40,
    },
))
```

### Adding a new curve type
Subclass or add to `curves.py`. Key methods: `forward_points_at(t)`, `forward(t)`, `implied_yield(t)`, and shock methods that return new instances.

## Learning Mode

Toggle "Learning Mode" in the sidebar for inline explanations of:
- Buy-sell vs sell-buy conventions
- DV01 and key-rate DV01
- Carry vs roll-down mechanics
- NDF cash settlement and basis
- Why steeper curves produce more roll-down (INR vs TRY comparison)
