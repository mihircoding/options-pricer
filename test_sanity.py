"""
Quick sanity checks for the pricing math. Run with:  python test_sanity.py

These aren't exhaustive unit tests, just the classic identities every
Black-Scholes implementation must satisfy. If any of these break, the
model is wrong.
"""

import numpy as np

import binomial as bino
import black_scholes as bs
import monte_carlo as mc
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

num_rho = (bs.call_price(S, K, T, r + eps, sigma)
           - bs.call_price(S, K, T, r - eps, sigma)) / (2 * eps) / 100
check("rho matches numerical derivative",
      abs(float(bs.rho('call', S, K, T, r, sigma)) - float(num_rho)) < 1e-5)

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

# 7. Dividend-adjusted model (q > 0)
q = 0.03
cq = float(bs.call_price(S, K, T, r, sigma, q))
pq = float(bs.put_price(S, K, T, r, sigma, q))
# put-call parity with dividends: C - P = S*e^(-qT) - K*e^(-rT)
parity_q = cq - pq - (S * np.exp(-q * T) - K * np.exp(-r * T))
check(f"dividend put-call parity (residual {parity_q:.2e})",
      abs(parity_q) < 1e-9)
check("dividends make calls cheaper and puts dearer", cq < c and pq > p)

num_delta_q = (bs.call_price(S + eps, K, T, r, sigma, q)
               - bs.call_price(S - eps, K, T, r, sigma, q)) / (2 * eps)
check("dividend delta matches numerical derivative",
      abs(float(bs.delta('call', S, K, T, r, sigma, q))
          - float(num_delta_q)) < 1e-5)

num_theta_q = (bs.call_price(S, K, T - eps, r, sigma, q)
               - bs.call_price(S, K, T + eps, r, sigma, q)) / (2 * eps) / 365
check("dividend theta matches numerical derivative",
      abs(float(bs.theta('call', S, K, T, r, sigma, q))
          - float(num_theta_q)) < 1e-5)

iv_q = bs.implied_vol("call", cq, S, K, T, r, q)
check(f"dividend implied vol round-trip (got {iv_q:.4%})",
      abs(iv_q - sigma) < 1e-4)

# 8. Monte Carlo agrees with the closed form (within 4 standard errors -
#    a ~1 in 16,000 false-failure rate at 500k paths)
for opt in ("call", "put"):
    mc_val, mc_err = mc.mc_price(opt, S, K, T, r, sigma, 500_000, seed=0)
    bs_val = float(bs.price(opt, S, K, T, r, sigma))
    check(f"MC {opt} {mc_val:.4f} within 4 SE of BS {bs_val:.4f}",
          abs(mc_val - bs_val) < 4 * mc_err)

mc_q, mc_q_err = mc.mc_price("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
check("MC with dividends matches dividend-adjusted BS",
      abs(mc_q - cq) < 4 * mc_q_err)

# 9. Binomial tree (American/European) - two checks that don't depend on
#    each other agreeing by construction, since they easily could if I'd
#    made a copy-paste mistake.
#
#    a) European binomial -> Black-Scholes as steps grows. Different model
#       (discrete tree vs. closed-form integral), same no-arbitrage
#       argument, should land in the same place.
euro_tree = bino.crr_price("call", "european", S, K, T, r, sigma, steps=500)
check(f"European binomial -> Black-Scholes ({euro_tree:.4f} vs {c:.4f})",
      abs(euro_tree - c) < 0.02)

#    b) No dividend -> American call should never be worth exercising early
#       (a well-known result: you'd throw away remaining time value for
#       nothing, since there's no dividend to capture). American and
#       European calls should price identically when q=0.
prem_call_no_div = bino.early_exercise_premium("call", S, K, T, r, sigma, q=0.0, steps=300)
check(f"American call = European call when q=0 (premium {prem_call_no_div:.2e})",
      abs(prem_call_no_div) < 1e-6)

#    c) Puts are different: even with no dividend, it can be worth
#       exercising a deep ITM put early to start earning interest on the
#       strike now instead of waiting. Premium should be strictly positive
#       for a put that's meaningfully in the money.
prem_put_itm = bino.early_exercise_premium("put", 70.0, 100.0, T, r, sigma, q=0.0, steps=300)
check(f"American put has positive early-exercise premium when deep ITM "
      f"({prem_put_itm:.4f})", prem_put_itm > 0.01)

#    d) With a dividend, American calls CAN be worth exercising early too
#       (to capture the dividend before the stock drops on ex-div date) -
#       premium should turn positive once q > 0.
prem_call_div = bino.early_exercise_premium("call", S, K, T, r, sigma, q=0.05, steps=300)
check(f"American call premium turns positive with a dividend ({prem_call_div:.4f})",
      prem_call_div > 0.0)

# 10. Monte Carlo Greeks (pathwise delta/vega, likelihood-ratio gamma)
#     agree with the closed-form Black-Scholes Greeks, within 4 standard
#     errors - same bar as the MC price check above (#8), and the same
#     reason: these are estimates with real sampling noise, so "matches
#     within its own reported uncertainty" is the honest check, not
#     "matches exactly."
for opt in ("call", "put"):
    d_bs = float(bs.delta(opt, S, K, T, r, sigma))
    v_bs = float(bs.vega(S, K, T, r, sigma))
    g_bs = float(bs.gamma(S, K, T, r, sigma))

    d_mc, d_se = mc.pathwise_delta(opt, S, K, T, r, sigma, 500_000, seed=0)
    v_mc, v_se = mc.pathwise_vega(opt, S, K, T, r, sigma, 500_000, seed=0)
    g_mc, g_se = mc.likelihood_ratio_gamma(opt, S, K, T, r, sigma, 500_000, seed=0)

    check(f"MC pathwise {opt} delta {d_mc:.4f} within 4 SE of BS {d_bs:.4f}",
          abs(d_mc - d_bs) < 4 * d_se)
    check(f"MC pathwise {opt} vega {v_mc:.4f} within 4 SE of BS {v_bs:.4f}",
          abs(v_mc - v_bs) < 4 * v_se)
    check(f"MC likelihood-ratio {opt} gamma {g_mc:.4f} within 4 SE of BS {g_bs:.4f}",
          abs(g_mc - g_bs) < 4 * g_se)

# same three, with a dividend - checks that q flows correctly into all
# three estimators, not just into mc_price
d_bs_q = float(bs.delta("call", S, K, T, r, sigma, q))
v_bs_q = float(bs.vega(S, K, T, r, sigma, q))
g_bs_q = float(bs.gamma(S, K, T, r, sigma, q))
d_mc_q, d_se_q = mc.pathwise_delta("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
v_mc_q, v_se_q = mc.pathwise_vega("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
g_mc_q, g_se_q = mc.likelihood_ratio_gamma("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
check("MC dividend delta matches dividend-adjusted BS delta",
      abs(d_mc_q - d_bs_q) < 4 * d_se_q)
check("MC dividend vega matches dividend-adjusted BS vega",
      abs(v_mc_q - v_bs_q) < 4 * v_se_q)
check("MC dividend gamma matches dividend-adjusted BS gamma",
      abs(g_mc_q - g_bs_q) < 4 * g_se_q)

# _implied_z must actually be the inverse of terminal_prices' own formula -
# if this ever drifted out of sync, every Greek above would be silently
# wrong in a way none of the "matches BS" checks would clearly point to.
s_t_check = mc.terminal_prices(S, T, r, sigma, 10_000, seed=1)
import numpy as _np
z_recovered = mc._implied_z(s_t_check, S, T, r, sigma)
s_t_rebuilt = S * _np.exp((r - 0.5 * sigma**2) * T + sigma * _np.sqrt(T) * z_recovered)
check("_implied_z round-trips through terminal_prices' own formula",
      _np.allclose(s_t_rebuilt, s_t_check, rtol=1e-9))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    raise SystemExit(1)
print("All sanity checks passed.")
