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
    q     - annualized continuous dividend yield (0.02 = 2%), default 0

Dividends matter because the stock price drops by the dividend on each
ex-dividend date. The standard fix (Merton's extension) is to discount
the spot by e^(-qT) everywhere it appears: you effectively price the
option on the stock minus the dividends you won't receive by holding
the option instead of the stock. Set q=0 and every formula collapses
back to plain Black-Scholes.
"""

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma, q=0.0):
    """
    The two probability-ish terms at the heart of Black-Scholes.

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
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

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma, q=0.0):
    """C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)"""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q=0.0):
    """P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)  (put-call parity form)"""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def price(option_type, S, K, T, r, sigma, q=0.0):
    """Convenience wrapper: option_type is 'call' or 'put'."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)
    return put_price(S, K, T, r, sigma, q)


# ---------------------------------------------------------------------------
# Greeks - the partial derivatives of the price formula.
# Each one answers "how much does the option price move if X changes a bit?"
# ---------------------------------------------------------------------------

def delta(option_type, S, K, T, r, sigma, q=0.0):
    """
    dPrice/dSpot. Call delta is e^(-qT)*N(d1) (0 to 1), put delta is
    e^(-qT)*(N(d1)-1) (-1 to 0). A delta of 0.6 means the option gains
    ~$0.60 when the stock gains $1.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """
    dDelta/dSpot - how fast delta itself changes. Same for calls and
    puts. Highest for at-the-money options near expiry.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """
    dPrice/dVol, quoted per 1 percentage point of volatility (hence the
    /100). Same for calls and puts.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-10)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(option_type, S, K, T, r, sigma, q=0.0):
    """
    dPrice/dTime, quoted per calendar DAY (hence the /365). Almost
    always negative for long options - they lose value as time passes.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-10)
    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == "call":
        yearly = (common
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        yearly = (common
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return yearly / 365.0


def rho(option_type, S, K, T, r, sigma, q=0.0):
    """
    dPrice/dRate, quoted per 1 percentage point of interest rate.
    Calls gain when rates rise, puts lose.
    """
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def all_greeks(option_type, S, K, T, r, sigma, q=0.0):
    """All five greeks in one dict - used by the UI."""
    return {
        "Delta": float(delta(option_type, S, K, T, r, sigma, q)),
        "Gamma": float(gamma(S, K, T, r, sigma, q)),
        "Vega": float(vega(S, K, T, r, sigma, q)),
        "Theta": float(theta(option_type, S, K, T, r, sigma, q)),
        "Rho": float(rho(option_type, S, K, T, r, sigma, q)),
    }


# ---------------------------------------------------------------------------
# Implied volatility - invert the formula: given a market price, find the
# sigma that makes Black-Scholes spit out that price. There is no closed
# form, so we use simple bisection (robust, no derivatives needed).
# ---------------------------------------------------------------------------

def implied_vol(option_type, market_price, S, K, T, r, q=0.0,
                lo=1e-4, hi=5.0, tol=1e-6, max_iter=100):
    """
    Bisection search: price is monotonically increasing in sigma, so we
    keep halving the [lo, hi] interval until the model price matches the
    market price. Returns np.nan if the market price is outside what any
    volatility in [lo, hi] could produce (e.g. price below intrinsic).
    """
    f_lo = price(option_type, S, K, T, r, lo, q) - market_price
    f_hi = price(option_type, S, K, T, r, hi, q) - market_price
    if f_lo * f_hi > 0:            # no sign change -> no root in range
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = price(option_type, S, K, T, r, mid, q) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
