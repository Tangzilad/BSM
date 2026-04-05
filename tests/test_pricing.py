"""
tests/test_pricing.py – pytest suite for the EM STIR trading sandbox.

Covers: curve construction, FX swap pricing, NDF pricing, DV01 sanity,
key-rate DV01 additivity, steepener/butterfly neutrality, and roll-down.
"""

import math
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from curves import (
    ImpliedYieldCurve, FXForwardCurve, NelsonSiegelCurve,
    fwd_points_from_implied_yield,
)
from instruments import FXSwap, NDF, Steepener, Butterfly
from risk import (
    compute_dv01, compute_dv01_fxswap, compute_key_rate_dv01,
    compute_bucketed_dv01, compute_fwd_pts_dv01, portfolio_risk,
    scenario_pnl, rolldown_profile, hedge_ratio,
)
from scenarios import get_scenario_library, Scenario


# ===================================================================
# Fixtures: realistic USD/INR curves
# ===================================================================

@pytest.fixture
def usd_curve():
    tenors = (1/365, 7/365, 30/365, 91/365, 182/365, 274/365, 365/365)
    rates = (0.053, 0.0532, 0.0535, 0.054, 0.0545, 0.0548, 0.055)
    return ImpliedYieldCurve(tenors, rates, label="USD")


@pytest.fixture
def onshore_curve(usd_curve):
    spot = 83.50
    tenors = (1/365, 7/365, 30/365, 91/365, 182/365, 274/365, 365/365)
    fwd_pts = (0.5, 3.5, 15.0, 47.0, 97.0, 150.0, 205.0)
    return FXForwardCurve(spot, tenors, fwd_pts, usd_curve, label="Onshore")


@pytest.fixture
def offshore_curve(usd_curve):
    spot = 83.50
    tenors = (1/365, 7/365, 30/365, 91/365, 182/365, 274/365, 365/365)
    fwd_pts = (0.6, 4.0, 17.0, 53.0, 110.0, 170.0, 230.0)
    return FXForwardCurve(spot, tenors, fwd_pts, usd_curve, label="Offshore NDF")


@pytest.fixture
def try_curve(usd_curve):
    """Steep TRY curve for roll-down comparison."""
    spot = 32.50
    tenors = (7/365, 30/365, 91/365, 182/365, 274/365, 365/365)
    fwd_pts = (110.0, 480.0, 1500.0, 3100.0, 4800.0, 6600.0)
    return FXForwardCurve(spot, tenors, fwd_pts, usd_curve, label="USD/TRY")


# ===================================================================
# Curve tests
# ===================================================================

class TestCurves:

    def test_discount_factor_at_zero(self, usd_curve):
        assert usd_curve.discount_factor(0) == 1.0

    def test_discount_factor_decreasing(self, usd_curve):
        df_3m = usd_curve.discount_factor(0.25)
        df_1y = usd_curve.discount_factor(1.0)
        assert 0 < df_1y < df_3m < 1.0

    def test_zero_rate_positive(self, usd_curve):
        for t in [0.01, 0.25, 0.5, 1.0]:
            assert usd_curve.zero_rate(t) > 0

    def test_forward_rate_positive(self, usd_curve):
        fwd = usd_curve.forward_rate(0.25, 0.5)
        assert fwd > 0

    def test_parallel_shift(self, usd_curve):
        shifted = usd_curve.parallel_shift(10)
        for t in usd_curve.tenors:
            assert abs(shifted.zero_rate(t) - usd_curve.zero_rate(t) - 0.001) < 1e-10

    def test_fx_forward_curve_forward_points(self, onshore_curve):
        # At tenor nodes, should recover the input fwd points (before basis)
        for t, pts in zip(onshore_curve.tenors, onshore_curve.fwd_points):
            assert abs(onshore_curve.forward_points_at(t) - pts) < 1e-8

    def test_fx_forward_positive(self, onshore_curve):
        for t in onshore_curve.tenors:
            assert onshore_curve.forward(t) > onshore_curve.spot

    def test_implied_yield_gt_usd(self, onshore_curve):
        """INR implied yield should be > USD yield (positive carry)."""
        for t in onshore_curve.tenors:
            if t > 0.01:
                iy = onshore_curve.implied_yield(t)
                r_usd = onshore_curve.usd_curve.zero_rate(t)
                assert iy > r_usd, f"INR implied yield {iy} <= USD {r_usd} at {t}"

    def test_basis_wedge(self, onshore_curve):
        """Basis wedge shifts forward points uniformly."""
        with_basis = onshore_curve.with_basis(10.0)
        for t in onshore_curve.tenors:
            diff = with_basis.forward_points_at(t) - onshore_curve.forward_points_at(t)
            assert abs(diff - 10.0) < 1e-8


class TestNelsonSiegel:

    def test_flat_curve(self):
        """beta1=beta2=0 → flat curve at beta0."""
        ns = NelsonSiegelCurve(beta0=0.065, beta1=0.0, beta2=0.0, tau=0.5)
        for t in [0.01, 0.25, 0.5, 1.0]:
            assert abs(ns.zero_rate(t) - 0.065) < 1e-8

    def test_upward_sloping(self):
        """Negative beta1 → upward-sloping curve."""
        ns = NelsonSiegelCurve(beta0=0.07, beta1=-0.02, beta2=0.0, tau=0.5)
        r_short = ns.zero_rate(0.01)
        r_long = ns.zero_rate(1.0)
        assert r_long > r_short

    def test_to_implied_yield_curve(self):
        ns = NelsonSiegelCurve(beta0=0.065, beta1=-0.01, beta2=0.005, tau=0.5)
        tenors = [0.25, 0.5, 1.0]
        iy = ns.to_implied_yield_curve(tenors)
        assert len(iy.tenors) == 3
        for t in tenors:
            assert abs(iy.zero_rate(t) - ns.zero_rate(t)) < 1e-8


# ===================================================================
# FX Swap pricing tests
# ===================================================================

class TestFXSwap:

    def test_pv_at_market_is_zero(self, onshore_curve):
        """FX swap done at current market fwd pts → PV ≈ 0."""
        t = 91 / 365
        market_pts = onshore_curve.forward_points_at(t)
        swap = FXSwap(t, market_pts, 10_000_000, "buy_sell", "test")
        assert abs(swap.pv(onshore_curve)) < 1e-6

    def test_buy_sell_profits_when_pts_fall(self, onshore_curve):
        """Buy-sell (receiver) profits when fwd pts fall."""
        t = 91 / 365
        # Entered at 50 paise, market now at 47 → profit
        swap = FXSwap(t, 50.0, 10_000_000, "buy_sell")
        pv = swap.pv(onshore_curve)
        assert pv > 0, f"Expected positive PV, got {pv}"

    def test_sell_buy_profits_when_pts_rise(self, onshore_curve):
        """Sell-buy (payer) profits when fwd pts rise."""
        t = 91 / 365
        # Entered at 44 paise, market now at 47 → profit
        swap = FXSwap(t, 44.0, 10_000_000, "sell_buy")
        pv = swap.pv(onshore_curve)
        assert pv > 0, f"Expected positive PV, got {pv}"

    def test_buy_sell_dv01_negative(self, onshore_curve):
        """Buy-sell (receiver): implied yields up → loss → DV01 < 0."""
        t = 91 / 365
        swap = FXSwap(t, 47.0, 10_000_000, "buy_sell")
        dv01 = compute_dv01_fxswap(swap, onshore_curve)
        assert dv01 < 0, f"Expected negative DV01, got {dv01}"

    def test_sell_buy_dv01_positive(self, onshore_curve):
        """Sell-buy (payer): implied yields up → gain → DV01 > 0."""
        t = 91 / 365
        swap = FXSwap(t, 47.0, 10_000_000, "sell_buy")
        dv01 = compute_dv01_fxswap(swap, onshore_curve)
        assert dv01 > 0, f"Expected positive DV01, got {dv01}"

    def test_direction_symmetry(self, onshore_curve):
        """Buy-sell and sell-buy at same terms have opposite PV."""
        t = 91 / 365
        bs = FXSwap(t, 50.0, 10_000_000, "buy_sell")
        sb = FXSwap(t, 50.0, 10_000_000, "sell_buy")
        assert abs(bs.pv(onshore_curve) + sb.pv(onshore_curve)) < 1e-8


# ===================================================================
# NDF tests
# ===================================================================

class TestNDF:

    def test_pv_zero_at_market(self, offshore_curve):
        """NDF PV = 0 when trade forward = market forward."""
        t = 91 / 365
        market_fwd = offshore_curve.forward(t)
        ndf = NDF(t, market_fwd, 10_000_000, "USD", "buy_usd", label="test")
        assert abs(ndf.pv(offshore_curve)) < 1e-6

    def test_buy_usd_profits_when_inr_depreciates(self, offshore_curve):
        """Buy USD NDF profits when INR depreciates (market fwd > trade fwd)."""
        t = 91 / 365
        trade_fwd = 83.50  # below current market forward
        ndf = NDF(t, trade_fwd, 10_000_000, "USD", "buy_usd")
        pv = ndf.pv(offshore_curve)
        assert pv > 0, f"Expected positive PV, got {pv}"

    def test_basis_wedge_shifts_pv(self, offshore_curve):
        """Nonzero basis wedge changes NDF PV."""
        t = 91 / 365
        market_fwd = offshore_curve.forward(t)
        ndf_no_basis = NDF(t, market_fwd, 10_000_000, "USD", "buy_usd", basis_wedge=0)
        ndf_with_basis = NDF(t, market_fwd, 10_000_000, "USD", "buy_usd", basis_wedge=50)
        pv_no = ndf_no_basis.pv(offshore_curve)
        pv_with = ndf_with_basis.pv(offshore_curve)
        assert abs(pv_no) < 1e-6
        assert abs(pv_with) > 0.01  # material change

    def test_inr_notional(self, offshore_curve):
        """INR notional NDF should also price correctly."""
        t = 91 / 365
        market_fwd = offshore_curve.forward(t)
        ndf = NDF(t, market_fwd, 835_000_000, "INR", "buy_usd")
        assert abs(ndf.pv(offshore_curve)) < 1e-4

    def test_fx_spot_sensitivity(self, offshore_curve):
        """Spot sensitivity should be nonzero for NDF."""
        t = 91 / 365
        ndf = NDF(t, 84.00, 10_000_000, "USD", "buy_usd")
        delta = ndf.fx_spot_sensitivity(offshore_curve)
        assert abs(delta) > 0

    def test_fwd_point_sensitivity(self, offshore_curve):
        """Forward point sensitivity should be nonzero."""
        t = 91 / 365
        ndf = NDF(t, 84.00, 10_000_000, "USD", "buy_usd")
        sens = ndf.forward_point_sensitivity(offshore_curve)
        assert abs(sens) > 0


# ===================================================================
# Key-rate DV01 tests
# ===================================================================

class TestKeyRateDV01:

    def test_kr_dv01_sum_approx_parallel(self, onshore_curve):
        """Sum of key-rate DV01s ≈ parallel DV01."""
        swap = FXSwap(91/365, 47.0, 10_000_000, "buy_sell")
        parallel = compute_dv01(swap, onshore_curve)
        kr = compute_key_rate_dv01(swap, onshore_curve)
        kr_sum = sum(kr.values())
        # Allow 10% tolerance due to interpolation effects
        assert abs(kr_sum - parallel) < abs(parallel) * 0.10 + 1e-4, \
            f"KR sum {kr_sum} vs parallel {parallel}"

    def test_bucketed_dv01_sums_to_total(self, onshore_curve):
        """Bucketed DV01 should sum to total key-rate DV01."""
        swap = FXSwap(182/365, 97.0, 10_000_000, "sell_buy")
        kr = compute_key_rate_dv01(swap, onshore_curve)
        bucketed = compute_bucketed_dv01(kr)
        assert abs(sum(bucketed.values()) - sum(kr.values())) < 1e-8

    def test_longer_tenor_larger_dv01(self, onshore_curve):
        """Longer tenor FX swap should have larger absolute DV01."""
        short = FXSwap(30/365, 15.0, 10_000_000, "buy_sell")
        long = FXSwap(365/365, 205.0, 10_000_000, "buy_sell")
        dv01_short = abs(compute_dv01(short, onshore_curve))
        dv01_long = abs(compute_dv01(long, onshore_curve))
        assert dv01_long > dv01_short


# ===================================================================
# Steepener / Butterfly tests
# ===================================================================

class TestCurveTrades:

    def test_steepener_residual_dv01_small(self, onshore_curve):
        """DV01-neutral steepener should have near-zero parallel DV01."""
        steep = Steepener(
            30/365, 365/365, 10_000_000, "steepener",
            15.0, 205.0, "1M/12M steep",
        )
        dv01 = compute_dv01(steep, onshore_curve)
        # Should be small relative to single-leg DV01
        single = FXSwap(30/365, 15.0, 10_000_000, "buy_sell")
        single_dv01 = abs(compute_dv01(single, onshore_curve))
        assert abs(dv01) < single_dv01 * 0.15, \
            f"Steepener DV01 {dv01} too large vs single-leg {single_dv01}"

    def test_steepener_profits_from_steepening(self, onshore_curve):
        """Steepener should profit when curve steepens."""
        steep = Steepener(
            30/365, 365/365, 10_000_000, "steepener",
            15.0, 205.0, "1M/12M steep",
        )
        pv_base = steep.pv(onshore_curve)
        # Steepen: short end down, long end up
        shocked = onshore_curve.twist_implied_yield(0.25, 20)
        pv_shocked = steep.pv(shocked)
        pnl = pv_shocked - pv_base
        assert pnl > 0, f"Steepener should profit from steepening, got P&L={pnl}"

    def test_butterfly_residual_dv01_small(self, onshore_curve):
        """Butterfly should have near-zero parallel DV01."""
        fly = Butterfly(
            30/365, 91/365, 365/365, 10_000_000,
            15.0, 47.0, 205.0, "1M/3M/12M fly",
        )
        dv01 = compute_dv01(fly, onshore_curve)
        single = FXSwap(91/365, 47.0, 10_000_000, "buy_sell")
        single_dv01 = abs(compute_dv01(single, onshore_curve))
        assert abs(dv01) < single_dv01 * 0.20, \
            f"Butterfly DV01 {dv01} too large vs belly {single_dv01}"


# ===================================================================
# Roll-down tests
# ===================================================================

class TestRollDown:

    def test_receiver_positive_rolldown_on_steep_curve(self, onshore_curve):
        """Buy-sell (receiver) earns positive roll-down on upward-sloping curve."""
        swap = FXSwap(182/365, 97.0, 1_000_000, "buy_sell")
        cr = swap.carry_and_rolldown(onshore_curve, 30/365)
        assert cr["rolldown_usd"] > 0, \
            f"Expected positive roll-down, got {cr['rolldown_usd']}"

    def test_carry_positive_for_receiver(self, onshore_curve):
        """Buy-sell earns positive carry (INR yield > USD yield)."""
        swap = FXSwap(91/365, 47.0, 1_000_000, "buy_sell")
        cr = swap.carry_and_rolldown(onshore_curve, 30/365)
        assert cr["carry_usd"] > 0, \
            f"Expected positive carry, got {cr['carry_usd']}"

    def test_steeper_curve_more_rolldown(self, onshore_curve, try_curve):
        """TRY (steeper) should produce more roll-down than INR."""
        holding = 30 / 365

        df_inr = rolldown_profile(onshore_curve, holding, 1_000_000)
        df_try = rolldown_profile(try_curve, holding, 1_000_000)

        # Compare total (carry + rolldown) at comparable tenors
        # Both should have 6M and 12M rows; TRY totals should be larger
        inr_total = df_inr["total_usd"].sum()
        try_total = df_try["total_usd"].sum()
        assert try_total > inr_total, \
            f"TRY total {try_total} should exceed INR {inr_total}"

    def test_rolldown_profile_non_empty(self, onshore_curve):
        df = rolldown_profile(onshore_curve, 30/365, 1_000_000)
        assert len(df) > 0
        assert "tenor_label" in df.columns
        assert "rolldown_usd" in df.columns


# ===================================================================
# Scenario tests
# ===================================================================

class TestScenarios:

    def test_library_has_five_scenarios(self):
        lib = get_scenario_library()
        assert len(lib) == 5

    def test_scenario_apply_returns_curve(self, onshore_curve):
        lib = get_scenario_library()
        for s in lib:
            shocked = s.apply(onshore_curve)
            assert isinstance(shocked, FXForwardCurve)

    def test_hawkish_hurts_receiver(self, onshore_curve):
        """Hawkish surprise (rates up) should hurt a receiver."""
        swap = FXSwap(91/365, 47.0, 10_000_000, "buy_sell", "Rcv 3M")
        from scenarios import get_scenario_by_name
        hawkish = get_scenario_by_name("Hawkish")
        shocked = hawkish.apply(onshore_curve)
        pnl = swap.pv(shocked) - swap.pv(onshore_curve)
        assert pnl < 0, f"Receiver should lose in hawkish, got P&L={pnl}"

    def test_dovish_helps_receiver(self, onshore_curve):
        """Dovish surprise (rates down) should help a receiver."""
        swap = FXSwap(91/365, 47.0, 10_000_000, "buy_sell", "Rcv 3M")
        from scenarios import get_scenario_by_name
        dovish = get_scenario_by_name("Dovish")
        shocked = dovish.apply(onshore_curve)
        pnl = swap.pv(shocked) - swap.pv(onshore_curve)
        assert pnl > 0, f"Receiver should gain in dovish, got P&L={pnl}"

    def test_scenario_explain_not_empty(self):
        lib = get_scenario_library()
        for s in lib:
            text = s.explain()
            assert len(text) > 50


# ===================================================================
# Hedge ratio test
# ===================================================================

class TestHedgeRatio:

    def test_hedge_ratio_positive(self, onshore_curve):
        a = FXSwap(91/365, 47.0, 10_000_000, "buy_sell")
        b = FXSwap(91/365, 47.0, 10_000_000, "sell_buy")
        ratio = hedge_ratio(a, b, onshore_curve)
        assert ratio > 0  # opposite directions → positive ratio
