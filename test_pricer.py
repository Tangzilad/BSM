"""Tests for the Black-Scholes FX pricer and spot generator."""

import numpy as np

from black_scholes import BlackScholesFX
from spot_generator import SpotGenerator


def test_put_call_parity():
    """Garman-Kohlhagen put-call parity: C - P = S*exp(-r_f*T) - K*exp(-r_d*T)."""
    S, K, T, vol = 150.0, 151.0, 0.25, 0.10
    r_d, r_f = 0.001, 0.05

    p = BlackScholesFX(spot=S, strike=K, T=T, vol=vol, r_d=r_d, r_f=r_f)
    lhs = p.call_price() - p.put_price()
    rhs = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
    assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"


def test_call_put_delta_relation():
    """call_delta - put_delta = exp(-r_f*T) (from put-call parity)."""
    S, K, T, vol = 150.0, 150.0, 0.5, 0.10
    r_d, r_f = 0.001, 0.05

    p = BlackScholesFX(spot=S, strike=K, T=T, vol=vol, r_d=r_d, r_f=r_f)
    assert abs(p.delta("call") - p.delta("put") - np.exp(-r_f * T)) < 1e-10


def test_positive_gamma_vega():
    p = BlackScholesFX(spot=150.0, strike=151.0, T=0.25, vol=0.10, r_d=0.001, r_f=0.05)
    assert p.gamma() > 0
    assert p.vega() > 0


def test_call_price_bounds():
    """Call price bounded: 0 < C < S."""
    p = BlackScholesFX(spot=150.0, strike=151.0, T=0.25, vol=0.10, r_d=0.001, r_f=0.05)
    c = p.call_price()
    assert 0 < c < 150.0


def test_deep_itm_call():
    """Deep ITM call should be close to intrinsic value."""
    p = BlackScholesFX(spot=200.0, strike=100.0, T=0.01, vol=0.10, r_d=0.001, r_f=0.05)
    c = p.call_price()
    assert abs(c - 100.0) < 1.0


def test_spot_generator_mean():
    """Terminal spot mean should converge to S * exp(drift * T)."""
    S, drift, vol, T = 150.0, 0.02, 0.10, 1.0
    gen = SpotGenerator(spot=S, drift=drift, volatility=vol, seed=123)
    S_T = gen.generate_terminal_spots(n_samples=200_000, T=T)
    expected_mean = S * np.exp(drift * T)
    assert abs(np.mean(S_T) - expected_mean) / expected_mean < 0.01


def test_spot_generator_path_shape():
    gen = SpotGenerator(spot=150.0, seed=42)
    t, path = gen.generate_path(T=1.0, steps=252)
    assert len(t) == 253
    assert len(path) == 253
    assert path[0] == 150.0


def test_monte_carlo_convergence():
    """MC price should converge to BS analytical price."""
    S, K, T, vol = 150.0, 151.0, 0.25, 0.10
    r_d, r_f = 0.001, 0.05

    pricer = BlackScholesFX(spot=S, strike=K, T=T, vol=vol, r_d=r_d, r_f=r_f)
    gen = SpotGenerator(spot=S, drift=r_d - r_f, volatility=vol, seed=99)

    S_T = gen.generate_terminal_spots(n_samples=500_000, T=T)
    mc_call = np.exp(-r_d * T) * np.mean(np.maximum(S_T - K, 0))
    bs_call = pricer.call_price()
    assert abs(mc_call - bs_call) < 0.1, f"MC={mc_call:.4f} vs BS={bs_call:.4f}"


if __name__ == "__main__":
    tests = [
        test_put_call_parity,
        test_call_put_delta_relation,
        test_positive_gamma_vega,
        test_call_price_bounds,
        test_deep_itm_call,
        test_spot_generator_mean,
        test_spot_generator_path_shape,
        test_monte_carlo_convergence,
    ]
    for test in tests:
        test()
        print(f"  PASS: {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
