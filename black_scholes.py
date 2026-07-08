"""
Black-Scholes option pricing, implemented from scratch.

The only outside math help is scipy.stats.norm for the standard normal
CDF/PDF - everything else (d1, d2, prices, greeks, implied vol) is done
by hand from the closed-form equations.

Notation used everywhere in this project:
    S     - spot price of the underlying
    K     - strike price
    T     - time to expiry in YEARS (30 days -> 30/365)
    r     - annualized risk-free interest rate (0.05 = 5%)
    sigma - annualized volatility of the underlying (0.20 = 20%)
"""

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma):
    """
    The two probability-ish terms at the heart of Black-Scholes.

    d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Loosely: N(d2) is the risk-neutral probability the option finishes
    in the money, and N(d1) is that probability adjusted for the size
    of the payoff when it does.
    """
    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # avoid division by zero when T or sigma is 0 (option at expiry /
    # no movement) - clip to a tiny positive number instead
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    """C = S*N(d1) - K*e^(-rT)*N(d2)"""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """P = K*e^(-rT)*N(-d2) - S*N(-d1)  (same thing via put-call parity)"""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price(option_type, S, K, T, r, sigma):
    """Convenience wrapper: option_type is 'call' or 'put'."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma)
    return put_price(S, K, T, r, sigma)


# ---------------------------------------------------------------------------
# Greeks - the partial derivatives of the price formula.
# Each one answers "how much does the option price move if X changes a bit?"
# ---------------------------------------------------------------------------

def delta(option_type, S, K, T, r, sigma):
    """
    dPrice/dSpot. Call delta is N(d1) (0 to 1), put delta is N(d1)-1
    (-1 to 0). A delta of 0.6 means the option gains ~$0.60 when the
    stock gains $1.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma):
    """
    dDelta/dSpot - how fast delta itself changes. Same for calls and
    puts. Highest for at-the-money options near expiry.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    dPrice/dVol, quoted per 1 percentage point of volatility (hence the
    /100). Same for calls and puts.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    T = np.maximum(T, 1e-10)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(option_type, S, K, T, r, sigma):
    """
    dPrice/dTime, quoted per calendar DAY (hence the /365). Almost
    always negative for long options - they lose value as time passes.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    T = np.maximum(T, 1e-10)
    common = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == "call":
        yearly = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        yearly = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return yearly / 365.0


def rho(option_type, S, K, T, r, sigma):
    """
    dPrice/dRate, quoted per 1 percentage point of interest rate.
    Calls gain when rates rise, puts lose.
    """
    _, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def all_greeks(option_type, S, K, T, r, sigma):
    """All five greeks in one dict - used by the UI."""
    return {
        "Delta": float(delta(option_type, S, K, T, r, sigma)),
        "Gamma": float(gamma(S, K, T, r, sigma)),
        "Vega": float(vega(S, K, T, r, sigma)),
        "Theta": float(theta(option_type, S, K, T, r, sigma)),
        "Rho": float(rho(option_type, S, K, T, r, sigma)),
    }


# ---------------------------------------------------------------------------
# Implied volatility - invert the formula: given a market price, find the
# sigma that makes Black-Scholes spit out that price. There is no closed
# form, so we use simple bisection (robust, no derivatives needed).
# ---------------------------------------------------------------------------

def implied_vol(option_type, market_price, S, K, T, r,
                lo=1e-4, hi=5.0, tol=1e-6, max_iter=100):
    """
    Bisection search: price is monotonically increasing in sigma, so we
    keep halving the [lo, hi] interval until the model price matches the
    market price. Returns np.nan if the market price is outside what any
    volatility in [lo, hi] could produce (e.g. price below intrinsic).
    """
    f_lo = price(option_type, S, K, T, r, lo) - market_price
    f_hi = price(option_type, S, K, T, r, hi) - market_price
    if f_lo * f_hi > 0:            # no sign change -> no root in range
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = price(option_type, S, K, T, r, mid) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
