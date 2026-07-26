"""
Monte Carlo option pricing - the same answer as Black-Scholes, reached a
completely different way.

Black-Scholes assumes the stock follows geometric Brownian motion (GBM):
random walk in log-space with drift. Under the risk-neutral measure the
terminal price after time T is

    S_T = S * exp( (r - q - sigma^2/2) * T  +  sigma * sqrt(T) * Z )

where Z is a standard normal draw. So instead of solving the math in
closed form, we can literally simulate that equation many thousands of
times, average the option payoffs, and discount back to today:

    price ~ e^(-rT) * mean( payoff(S_T) )

By the law of large numbers this converges to the Black-Scholes price as
the number of simulations grows - watching that happen on the Monte Carlo
page is the best evidence the closed-form model is doing what it claims.
"""

import numpy as np


def terminal_prices(S, T, r, sigma, n_paths, seed=0, q=0.0):
    """
    Draw n_paths terminal stock prices from the risk-neutral GBM
    distribution in one vectorized shot. A fixed seed keeps the page
    reproducible - change it to see a different random sample.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def mc_price(option_type, S, K, T, r, sigma, n_paths, seed=0, q=0.0):
    """
    Monte Carlo price plus its standard error. The standard error
    (std of the discounted payoffs / sqrt(n)) tells you how much the
    estimate would wiggle if you re-ran with a different seed - the
    true price should sit within ~2 standard errors 95% of the time.
    """
    s_t = terminal_prices(S, T, r, sigma, n_paths, seed, q)
    if option_type == "call":
        payoffs = np.maximum(s_t - K, 0.0)
    else:
        payoffs = np.maximum(K - s_t, 0.0)
    discounted = np.exp(-r * T) * payoffs
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def sample_paths(S, T, r, sigma, n_paths=50, n_steps=100, seed=0, q=0.0):
    """
    Full price paths (not just endpoints) for the chart - each path is
    built step by step with the same GBM equation applied over small
    time slices dt. Returns (time grid, paths matrix [n_paths x n_steps+1]).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return np.linspace(0.0, T, n_steps + 1), paths


def convergence_curve(option_type, S, K, T, r, sigma, max_paths, seed=0, q=0.0):
    """
    MC estimate as a function of sample size, computed by taking running
    means over ONE big draw (so the curve is smooth and cheap). Returns
    (sample sizes, estimates) for the convergence chart.
    """
    s_t = terminal_prices(S, T, r, sigma, max_paths, seed, q)
    if option_type == "call":
        payoffs = np.maximum(s_t - K, 0.0)
    else:
        payoffs = np.maximum(K - s_t, 0.0)
    discounted = np.exp(-r * T) * payoffs
    running_mean = np.cumsum(discounted) / np.arange(1, max_paths + 1)
    # log-spaced sample points so the x-axis reads nicely from 100 to max
    start = min(100, max_paths)
    idx = np.unique(np.geomspace(start, max_paths, 200).astype(int)) - 1
    return idx + 1, running_mean[idx]
