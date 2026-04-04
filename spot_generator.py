"""
USDJPY spot price generator using Geometric Brownian Motion (GBM).

Simulates realistic FX spot paths for use with the Black-Scholes pricer.
"""

import numpy as np


class SpotGenerator:
    """Generates USDJPY spot price paths via GBM."""

    def __init__(
        self,
        spot: float = 150.0,
        drift: float = 0.02,
        volatility: float = 0.10,
        seed: int | None = None,
    ):
        self.spot = spot
        self.drift = drift
        self.volatility = volatility
        self.rng = np.random.default_rng(seed)

    def generate_path(
        self, T: float = 1.0, steps: int = 252
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a single spot price path.

        Args:
            T: Time horizon in years.
            steps: Number of time steps (252 = business days in a year).

        Returns:
            Tuple of (time_grid, spot_path).
        """
        dt = T / steps
        t = np.linspace(0, T, steps + 1)

        Z = self.rng.standard_normal(steps)
        log_returns = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * np.sqrt(dt) * Z
        log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
        path = self.spot * np.exp(log_path)

        return t, path

    def generate_paths(
        self, n_paths: int = 1000, T: float = 1.0, steps: int = 252
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate multiple spot price paths (Monte Carlo).

        Args:
            n_paths: Number of simulation paths.
            T: Time horizon in years.
            steps: Number of time steps.

        Returns:
            Tuple of (time_grid, paths) where paths has shape (n_paths, steps+1).
        """
        dt = T / steps
        t = np.linspace(0, T, steps + 1)

        Z = self.rng.standard_normal((n_paths, steps))
        log_returns = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * np.sqrt(dt) * Z
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
        )
        paths = self.spot * np.exp(log_paths)

        return t, paths

    def generate_terminal_spots(
        self, n_samples: int = 10000, T: float = 1.0
    ) -> np.ndarray:
        """Generate terminal spot prices (single-step GBM to maturity).

        Efficient for option pricing — skips intermediate steps.
        """
        Z = self.rng.standard_normal(n_samples)
        S_T = self.spot * np.exp(
            (self.drift - 0.5 * self.volatility**2) * T + self.volatility * np.sqrt(T) * Z
        )
        return S_T
