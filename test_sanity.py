"""
Quick sanity checks for the pricing math. Run with:  python test_sanity.py

These aren't exhaustive unit tests, just the classic identities every
Black-Scholes implementation must satisfy. If any of these break, the
model is wrong.
"""

import numpy as np

import black_scholes as bs
import strategies as strat

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

failures = []


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


# 1. Known textbook value: S=K=100, T=1, r=5%, vol=20%
#    call = 10.4506, put = 5.5735 (quoted in most derivatives textbooks)
c = float(bs.call_price(S, K, T, r, sigma))
p = float(bs.put_price(S, K, T, r, sigma))
check(f"textbook call 10.4506 (got {c:.4f})", abs(c - 10.4506) < 1e-3)
check(f"textbook put   5.5735 (got {p:.4f})", abs(p - 5.5735) < 1e-3)

# 2. Put-call parity: C - P must equal S - K*e^(-rT), always.
parity = c - p - (S - K * np.exp(-r * T))
check(f"put-call parity (residual {parity:.2e})", abs(parity) < 1e-9)

# 3. Greeks vs numerical differentiation (bump-and-reprice)
eps = 1e-4
num_delta = (bs.call_price(S + eps, K, T, r, sigma)
             - bs.call_price(S - eps, K, T, r, sigma)) / (2 * eps)
check("call delta matches numerical derivative",
      abs(float(bs.delta('call', S, K, T, r, sigma)) - float(num_delta)) < 1e-5)

num_vega = (bs.call_price(S, K, T, r, sigma + eps)
            - bs.call_price(S, K, T, r, sigma - eps)) / (2 * eps) / 100
check("vega matches numerical derivative",
      abs(float(bs.vega(S, K, T, r, sigma)) - float(num_vega)) < 1e-5)

num_theta = (bs.call_price(S, K, T - eps, r, sigma)
             - bs.call_price(S, K, T + eps, r, sigma)) / (2 * eps) / 365
check("theta matches numerical derivative",
      abs(float(bs.theta('call', S, K, T, r, sigma)) - float(num_theta)) < 1e-5)

# 4. Implied vol round-trip: price at 20% vol, invert, get 20% back
iv = bs.implied_vol("call", c, S, K, T, r)
check(f"implied vol round-trip (got {iv:.4%})", abs(iv - sigma) < 1e-4)

# 5. Deep in/out of the money limits
check("deep ITM call ~ S - K*e^(-rT)",
      abs(float(bs.call_price(200, 100, T, r, sigma))
          - (200 - 100 * np.exp(-r * T))) < 0.01)
check("deep OTM call ~ 0",
      float(bs.call_price(20, 100, T, r, sigma)) < 1e-6)

# 6. Strategy payoffs at expiry make sense
grid = np.linspace(50, 150, 101)
for name in strat.STRATEGY_NAMES:
    legs, _ = strat.build_strategy(name, S, K)
    payoff = strat.strategy_payoff(legs, grid)
    cost = strat.strategy_cost(legs, S, T, r, sigma)
    check(f"{name}: finite cost/payoff",
          np.isfinite(cost) and np.all(np.isfinite(payoff)))

# straddle payoff at strike should be exactly 0 (both legs expire worthless)
legs, _ = strat.build_strategy("Long Straddle", S, K)
check("straddle payoff at strike = 0",
      abs(float(strat.strategy_payoff(legs, np.array([K]))[0])) < 1e-9)

# iron condor opens for a credit (you get paid to take the trade)
legs, _ = strat.build_strategy("Iron Condor", S, K)
check("iron condor opens for a credit",
      strat.strategy_cost(legs, S, T, r, sigma) < 0)

# strategy_value at T=0 must equal the expiry payoff
legs, _ = strat.build_strategy("Bull Call Spread", S, K)
v0 = strat.strategy_value(legs, grid, 0.0, r, sigma)
pay = strat.strategy_payoff(legs, grid)
check("strategy value at T=0 equals expiry payoff",
      np.allclose(v0, pay, atol=1e-4))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    raise SystemExit(1)
print("All sanity checks passed.")
