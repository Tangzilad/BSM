"""
Black-Scholes pricer for USDJPY FX options (Garman-Kohlhagen model).

The Garman-Kohlhagen model extends Black-Scholes to FX options by
incorporating domestic and foreign risk-free rates.

For USDJPY:
  - Domestic currency: JPY (r_d = JPY risk-free rate)
  - Foreign currency:  USD (r_f = USD risk-free rate)
  - Spot S is quoted as JPY per 1 USD
"""

import numpy as np
from scipy.stats import norm


class BlackScholesFX:
    """Garman-Kohlhagen Black-Scholes pricer for FX options."""

    def __init__(
        self,
        spot: float,
        strike: float,
        T: float,
        vol: float,
        r_d: float,
        r_f: float,
    ):
        """
        Args:
            spot: Current USDJPY spot rate.
            strike: Option strike price.
            T: Time to expiry in years.
            vol: Annualized implied volatility.
            r_d: Domestic (JPY) risk-free rate.
            r_f: Foreign (USD) risk-free rate.
        """
        self.S = spot
        self.K = strike
        self.T = T
        self.vol = vol
        self.r_d = r_d
        self.r_f = r_f

    def _d1_d2(self) -> tuple[float, float]:
        d1 = (
            np.log(self.S / self.K)
            + (self.r_d - self.r_f + 0.5 * self.vol**2) * self.T
        ) / (self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)
        return d1, d2

    # ── Pricing ──────────────────────────────────────────────────────────

    def call_price(self) -> float:
        d1, d2 = self._d1_d2()
        return (
            self.S * np.exp(-self.r_f * self.T) * norm.cdf(d1)
            - self.K * np.exp(-self.r_d * self.T) * norm.cdf(d2)
        )

    def put_price(self) -> float:
        d1, d2 = self._d1_d2()
        return (
            self.K * np.exp(-self.r_d * self.T) * norm.cdf(-d2)
            - self.S * np.exp(-self.r_f * self.T) * norm.cdf(-d1)
        )

    # ── Greeks ───────────────────────────────────────────────────────────

    def delta(self, option_type: str = "call") -> float:
        d1, _ = self._d1_d2()
        if option_type == "call":
            return np.exp(-self.r_f * self.T) * norm.cdf(d1)
        return -np.exp(-self.r_f * self.T) * norm.cdf(-d1)

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        return (
            np.exp(-self.r_f * self.T)
            * norm.pdf(d1)
            / (self.S * self.vol * np.sqrt(self.T))
        )

    def vega(self) -> float:
        d1, _ = self._d1_d2()
        return (
            self.S
            * np.exp(-self.r_f * self.T)
            * norm.pdf(d1)
            * np.sqrt(self.T)
            / 100  # per 1% vol move
        )

    def theta(self, option_type: str = "call") -> float:
        d1, d2 = self._d1_d2()
        common = (
            -self.S
            * np.exp(-self.r_f * self.T)
            * norm.pdf(d1)
            * self.vol
            / (2 * np.sqrt(self.T))
        )
        if option_type == "call":
            return (
                common
                + self.r_f * self.S * np.exp(-self.r_f * self.T) * norm.cdf(d1)
                - self.r_d * self.K * np.exp(-self.r_d * self.T) * norm.cdf(d2)
            ) / 365  # per calendar day
        return (
            common
            - self.r_f * self.S * np.exp(-self.r_f * self.T) * norm.cdf(-d1)
            + self.r_d * self.K * np.exp(-self.r_d * self.T) * norm.cdf(-d2)
        ) / 365

    def rho_domestic(self, option_type: str = "call") -> float:
        _, d2 = self._d1_d2()
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r_d * self.T) * norm.cdf(d2) / 100
        return -self.K * self.T * np.exp(-self.r_d * self.T) * norm.cdf(-d2) / 100

    def rho_foreign(self, option_type: str = "call") -> float:
        d1, _ = self._d1_d2()
        if option_type == "call":
            return -self.S * self.T * np.exp(-self.r_f * self.T) * norm.cdf(d1) / 100
        return self.S * self.T * np.exp(-self.r_f * self.T) * norm.cdf(-d1) / 100

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self, option_type: str = "call") -> dict:
        price = self.call_price() if option_type == "call" else self.put_price()
        return {
            "option_type": option_type,
            "spot": self.S,
            "strike": self.K,
            "T": self.T,
            "vol": self.vol,
            "r_d (JPY)": self.r_d,
            "r_f (USD)": self.r_f,
            "price": round(price, 4),
            "delta": round(self.delta(option_type), 6),
            "gamma": round(self.gamma(), 6),
            "vega": round(self.vega(), 4),
            "theta": round(self.theta(option_type), 4),
            "rho_domestic": round(self.rho_domestic(option_type), 4),
            "rho_foreign": round(self.rho_foreign(option_type), 4),
        }
