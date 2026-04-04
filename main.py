#!/usr/bin/env python3
"""
USDJPY Black-Scholes Option Pricer with Spot Path Generator.

Usage:
    python main.py                  # Run with defaults
    python main.py --spot 148.5 --strike 150 --vol 0.12 --T 0.25
    python main.py --mc             # Include Monte Carlo comparison
    python main.py --plot           # Plot spot paths and Greeks
"""

import argparse

import numpy as np

from black_scholes import BlackScholesFX
from spot_generator import SpotGenerator


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_pricer(spot: float, strike: float, T: float, vol: float, r_d: float, r_f: float) -> None:
    """Price call and put, display results and Greeks."""
    pricer = BlackScholesFX(spot=spot, strike=strike, T=T, vol=vol, r_d=r_d, r_f=r_f)

    print_section("USDJPY Black-Scholes (Garman-Kohlhagen) Pricer")
    print(f"  Spot:   {spot:.2f}")
    print(f"  Strike: {strike:.2f}")
    print(f"  Expiry: {T:.4f} years")
    print(f"  Vol:    {vol*100:.1f}%")
    print(f"  r_d(JPY): {r_d*100:.2f}%  |  r_f(USD): {r_f*100:.2f}%")

    for opt_type in ("call", "put"):
        s = pricer.summary(opt_type)
        print_section(f"{opt_type.upper()} Option")
        print(f"  Price:  {s['price']:.4f} JPY")
        print(f"  Delta:  {s['delta']:+.6f}")
        print(f"  Gamma:  {s['gamma']:.6f}")
        print(f"  Vega:   {s['vega']:.4f}  (per 1% vol)")
        print(f"  Theta:  {s['theta']:.4f}  (per day)")
        print(f"  Rho_d:  {s['rho_domestic']:.4f}  (per 1% r_d)")
        print(f"  Rho_f:  {s['rho_foreign']:.4f}  (per 1% r_f)")


def run_monte_carlo(spot: float, strike: float, T: float, vol: float, r_d: float, r_f: float) -> None:
    """Compare analytical BS price with Monte Carlo simulation."""
    pricer = BlackScholesFX(spot=spot, strike=strike, T=T, vol=vol, r_d=r_d, r_f=r_f)
    gen = SpotGenerator(spot=spot, drift=r_d - r_f, volatility=vol, seed=42)

    n_sims = 100_000
    S_T = gen.generate_terminal_spots(n_samples=n_sims, T=T)

    call_payoffs = np.maximum(S_T - strike, 0)
    put_payoffs = np.maximum(strike - S_T, 0)
    df = np.exp(-r_d * T)

    mc_call = df * np.mean(call_payoffs)
    mc_put = df * np.mean(put_payoffs)
    bs_call = pricer.call_price()
    bs_put = pricer.put_price()

    print_section(f"Monte Carlo vs Analytical ({n_sims:,} paths)")
    print(f"  {'':12s} {'BS Analytical':>14s} {'Monte Carlo':>14s} {'Diff':>10s}")
    print(f"  {'Call':12s} {bs_call:14.4f} {mc_call:14.4f} {mc_call-bs_call:+10.4f}")
    print(f"  {'Put':12s} {bs_put:14.4f} {mc_put:14.4f} {mc_put-bs_put:+10.4f}")


def run_plot(spot: float, strike: float, T: float, vol: float, r_d: float, r_f: float) -> None:
    """Generate plots: spot paths, terminal distribution, and Greeks surfaces."""
    import matplotlib.pyplot as plt

    gen = SpotGenerator(spot=spot, drift=r_d - r_f, volatility=vol, seed=42)

    # --- Spot paths ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t, paths = gen.generate_paths(n_paths=50, T=T, steps=252)
    ax = axes[0, 0]
    for i in range(50):
        ax.plot(t * 252, paths[i], alpha=0.4, linewidth=0.7)
    ax.axhline(strike, color="red", linestyle="--", label=f"Strike={strike}")
    ax.set_title("USDJPY Simulated Spot Paths")
    ax.set_xlabel("Business Days")
    ax.set_ylabel("Spot (JPY/USD)")
    ax.legend()

    # --- Terminal distribution ---
    S_T = gen.generate_terminal_spots(n_samples=50_000, T=T)
    ax = axes[0, 1]
    ax.hist(S_T, bins=100, density=True, alpha=0.7, color="steelblue")
    ax.axvline(spot, color="green", linestyle="--", label=f"Spot={spot}")
    ax.axvline(strike, color="red", linestyle="--", label=f"Strike={strike}")
    ax.set_title("Terminal Spot Distribution")
    ax.set_xlabel("USDJPY at Expiry")
    ax.set_ylabel("Density")
    ax.legend()

    # --- Delta vs spot ---
    spots_range = np.linspace(spot * 0.85, spot * 1.15, 200)
    call_deltas = []
    put_deltas = []
    for s in spots_range:
        p = BlackScholesFX(spot=s, strike=strike, T=T, vol=vol, r_d=r_d, r_f=r_f)
        call_deltas.append(p.delta("call"))
        put_deltas.append(p.delta("put"))

    ax = axes[1, 0]
    ax.plot(spots_range, call_deltas, label="Call Delta", color="blue")
    ax.plot(spots_range, put_deltas, label="Put Delta", color="orange")
    ax.axvline(spot, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Delta vs Spot")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Delta")
    ax.legend()

    # --- Price vs spot ---
    call_prices = []
    put_prices = []
    for s in spots_range:
        p = BlackScholesFX(spot=s, strike=strike, T=T, vol=vol, r_d=r_d, r_f=r_f)
        call_prices.append(p.call_price())
        put_prices.append(p.put_price())

    ax = axes[1, 1]
    ax.plot(spots_range, call_prices, label="Call", color="blue")
    ax.plot(spots_range, put_prices, label="Put", color="orange")
    ax.axvline(strike, color="red", linestyle="--", alpha=0.5, label="Strike")
    ax.set_title("Option Price vs Spot")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Price (JPY)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("usdjpy_bs_analysis.png", dpi=150)
    print("\n  Plot saved to usdjpy_bs_analysis.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="USDJPY Black-Scholes FX Option Pricer")
    parser.add_argument("--spot", type=float, default=150.0, help="USDJPY spot rate")
    parser.add_argument("--strike", type=float, default=151.0, help="Strike price")
    parser.add_argument("--T", type=float, default=0.25, help="Time to expiry (years)")
    parser.add_argument("--vol", type=float, default=0.10, help="Implied volatility")
    parser.add_argument("--r_d", type=float, default=0.001, help="JPY risk-free rate")
    parser.add_argument("--r_f", type=float, default=0.05, help="USD risk-free rate")
    parser.add_argument("--mc", action="store_true", help="Run Monte Carlo comparison")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    run_pricer(args.spot, args.strike, args.T, args.vol, args.r_d, args.r_f)

    if args.mc:
        run_monte_carlo(args.spot, args.strike, args.T, args.vol, args.r_d, args.r_f)

    if args.plot:
        run_plot(args.spot, args.strike, args.T, args.vol, args.r_d, args.r_f)


if __name__ == "__main__":
    main()
