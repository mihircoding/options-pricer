---
title: Options Pricer
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# Options Pricer

**[Live site &rarr;](https://mihircoding.github.io/options-pricer/)** — prices an option three independent ways (closed form, Monte Carlo, binomial tree) live in the browser, with all five Greeks, the live SPY volatility surface, and the delta-hedging study below.

![tests](https://github.com/mihircoding/options-pricer/actions/workflows/ci.yml/badge.svg)

A Black-Scholes options pricing tool with an interactive Streamlit interface.
Prices calls and puts from scratch (with dividend adjustment), computes the
Greeks, builds ten classic multi-leg strategies, cross-checks the closed form
against a Monte Carlo simulator, and compares model prices against live
S&P 500 option quotes from Yahoo Finance - including the volatility smile
backed out of real market prices.

It also runs the model's own hedge. `hedging_study.py` sells a one-month
at-the-money SPY call every month from 2007 to 2024, delta-hedges it daily, and
decomposes the result: implied vol averaged 20.0% against 16.2% realized, the
seller made 0.405 per $100 a month at a Sharpe of 2.20, and lost 2.675 in March
2020. The gamma decomposition reproduces the realized P&L with r = 0.99, which
is what makes "a delta-hedged option is a bet on variance" a measurement rather
than a slogan. See below.

Built with Python, NumPy, SciPy, Matplotlib, Plotly and Streamlit.

## Running it locally

On Windows, just double-click `run.bat`. Or from a terminal:

```
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Your browser opens at http://localhost:8501 - that IS the website, served
from your machine. `streamlit_app.py` is the landing page; the five tool
pages live in `pages/` and show up in the left sidebar automatically.

Sanity-check the math (textbook values, put-call parity, greeks vs numerical
derivatives, implied-vol round trip):

```
python test_sanity.py
```

83 checks, including the arbitrage conditions above run against a chain
generated from Black-Scholes (where they must all hold exactly) and against
the same chain with one quote bent (where the right one must fire).

(`python -m pytest test_sanity.py` runs the same checks.)

The project page in `docs/` computes most of itself in the browser. The
volatility surface and the hedging study are real data, written to
`docs/data.js` by `python docs/build_data.py` (or `... build_data.py hedging`
to redo just the hedging part and keep the surface as it is).

## Putting it on the internet (free, no server needed)

Streamlit Community Cloud hosts Streamlit apps straight from a GitHub repo:

1. Make this repo **public** on GitHub (Settings -> General -> Danger Zone
   -> Change visibility). Streamlit's free tier only deploys public repos.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "Create app" -> "Deploy a public app from GitHub".
4. Pick this repo, branch `main`, main file `streamlit_app.py`. Deploy.

A couple of minutes later you get a permanent public URL like
`https://<yourname>-options-pricer.streamlit.app` that anyone can open -
live market data included. It redeploys itself every time you push.

## What's on each page

**1. Options Pricer (main page)** - set spot, strike, expiry, rate and
volatility in the sidebar. You get the fair call/put value up top
(green/red), price heatmaps over a spot x volatility grid, PnL heatmaps
(green = profit, red = loss, relative to what you paid for the option), and
the five Greeks. Hover any heatmap cell for exact numbers; every sidebar
input has a (?) blurb explaining the parameter.

**2. Strategy Builder** - pick one of ten strategies (covered call, married
put, bull call spread, bear put spread, protective collar, long straddle,
long strangle, long call butterfly, iron condor, iron butterfly). The page
shows the legs, the net debit/credit, the payoff diagram at expiry (green
above breakeven, red below), max profit / max loss / break-evens, and a
spot x volatility PnL heatmap you can slide through time.

**3. Market Comparison** - pick any S&P 500 stock and expiration. The page
pulls the live option chain, prices every near-the-money strike with this
project's own Black-Scholes code (using 1-year historical volatility and the
stock's actual trailing dividend yield), and shows model vs. market side by
side. Green rows: the market pays more than history justifies; red: less. It
backs implied volatility out of market prices with the project's own
bisection solver and plots it per strike - the volatility smile.

**4. Monte Carlo** - prices the same option by simulating thousands of
random price paths under the model's own assumption (geometric Brownian
motion) and shows the estimate converging onto the closed-form price, along
with the simulated paths themselves and the terminal price distribution
split into in-the-money (green) and worthless (red) outcomes.

**5. Vol Surface** - builds the implied volatility surface from a live
chain: the smile per expiry plotted in log-moneyness, the term structure of
at-the-money vol, 25-delta risk reversal and butterfly per expiry, a
calendar no-arbitrage check, and a bar chart of what pricing the whole chain
with one flat volatility costs strike by strike. See below for what makes
this different from plotting Yahoo's IV column.

## How the code works

### `black_scholes.py` - the model

The Black-Scholes formula prices a European option under the assumption
that the stock follows a random walk with constant volatility. The whole
formula hangs on two numbers:

```
d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

`N(d2)` is (roughly) the risk-neutral probability the option finishes in the
money; `N(d1)` weights that probability by how big the payoff is when it
happens. The prices are then:

```
call = S N(d1) - K e^(-rT) N(d2)        # what you get - what you pay, discounted
put  = K e^(-rT) N(-d2) - S N(-d1)
```

The only library math used is `scipy.stats.norm` for the normal CDF/PDF -
`d1`, `d2`, prices, all five Greeks and the implied-vol solver are hand-written.

The **Greeks** are the derivatives of that formula, i.e. sensitivity of the
price to each input: delta (per $1 of stock), gamma (delta's own rate of
change), vega (per vol point), theta (per day), rho (per rate point). The
test file verifies each one against a numerical bump-and-reprice derivative,
so the closed forms are provably consistent with the pricing function.

**Implied volatility** goes the other way: given a market price, find the
sigma that reproduces it. There's no closed form, so `implied_vol()` uses
bisection - price is monotonically increasing in volatility, so we keep
halving an interval [0.0001, 5.0] until model price matches market price.

### `strategies.py` - one representation, ten strategies

Every strategy is just a list of legs, e.g. an iron condor is:

```python
[ {kind: put,  strike: 80,  qty: +1},   # buy the far put   (wing)
  {kind: put,  strike: 90,  qty: -1},   # sell the near put
  {kind: call, strike: 110, qty: -1},   # sell the near call
  {kind: call, strike: 120, qty: +1} ]  # buy the far call  (wing)
```

Because of that one representation, three short generic functions do all the
work for all ten strategies:

- `strategy_cost()` - price each option leg with Black-Scholes, stock legs
  at spot, sum with signs. Negative total = you opened the trade for a credit.
- `strategy_payoff()` - value at expiry: `max(S-K, 0)` per call,
  `max(K-S, 0)` per put, times quantity, summed over legs.
- `strategy_value()` - value *before* expiry, repricing every leg with
  Black-Scholes at the remaining time. This is what makes the strategy PnL
  heatmap respond to volatility: an iron condor that's "safe" at expiry can
  still be underwater halfway there if volatility spikes.

PnL everywhere is just `value - cost`.

### Dividends (the `q` parameter)

Real stocks pay dividends, and holding an option doesn't entitle you to
them. The standard fix (Merton's extension) is to discount the spot price by
`e^(-qT)` everywhere it appears in the formulas, where q is the continuous
dividend yield. Every function in `black_scholes.py` takes `q` with a
default of 0, so the rest of the project works unchanged; the market page
estimates q from the stock's actual trailing 12 months of dividends.
Dividends drag the forward price down, so they make calls cheaper and puts
dearer - the test suite checks the dividend-adjusted put-call parity
`C - P = S e^(-qT) - K e^(-rT)` and re-verifies delta and theta against
numerical derivatives with q > 0.

### `monte_carlo.py` - the model priced a second way

Black-Scholes assumes the stock follows geometric Brownian motion. Under
the risk-neutral measure the terminal price is

```
S_T = S * exp( (r - q - sigma^2/2) T + sigma sqrt(T) Z ),   Z ~ N(0,1)
```

`terminal_prices()` draws thousands of those in one vectorized NumPy call;
`mc_price()` averages the payoffs, discounts by `e^(-rT)`, and reports a
standard error (payoff std / sqrt(n)) so you know how tight the estimate
is. `convergence_curve()` shows the running mean homing in on the closed
form - same assumption, completely different method, same answer. The test
suite requires the two prices to agree within 4 standard errors at 500k
paths, with and without dividends.

**Greeks by simulation, two different techniques.** `pathwise_delta()` and
`pathwise_vega()` differentiate the simulated PATH: `S_T` is a smooth
function of both spot and volatility (`dS_T/dS = S_T/S`,
`dS_T/dsigma = S_T*(sqrt(T)*Z - sigma*T)`), so the derivative can be pushed
inside the expectation and estimated straight from the same paths used for
pricing. That trick breaks for gamma - it needs the derivative of the
payoff's *slope*, and a call/put's slope jumps at the strike, so pathwise
differentiation would need to differentiate a discontinuity.
`likelihood_ratio_gamma()` sidesteps this by differentiating the
*probability density* of `S_T` instead of the payoff (the "score function"
method): the density stays smooth even where the payoff doesn't, so the
same trick that fails for gamma via one route works via the other. All
three are estimated from the same underlying draws (`_implied_z()` recovers
each path's `Z` algebraically from its `S_T` rather than redrawing it, so
there's no risk of the Greek estimators quietly using different randomness
than the price they're being compared against) and reported with their own
standard errors via `mc_greeks()`, same "estimate, not exact number"
discipline as `mc_price()`. Live on the Monte Carlo page, and in
`test_sanity.py` checked against the closed-form Greeks within 4 SE, with
and without dividends.

### `binomial.py` - American exercise, priced a third way

Black-Scholes and the Monte Carlo engine above both price *European*
exercise only - the model can't ask "what if I exercised early?" because
the closed-form solution assumes you can't. Most listed US equity options
are American-style, so this module builds a Cox-Ross-Rubinstein binomial
tree, which can: at every node, walking backward from expiry, it compares
holding the option against exercising it immediately and keeps whichever
is worth more.

Two things worth knowing from running it:

- **European binomial converges to Black-Scholes** as the tree gets more
  steps - a third method (discrete tree vs. Monte Carlo vs. closed form)
  landing in the same place, which is the whole point of cross-checking a
  pricing model three different ways instead of trusting one derivation.
- **The early-exercise premium is real and it isn't the same for calls and
  puts.** With no dividend, an American call is worth exactly the same as
  its European twin - there's nothing to gain by exercising early and
  giving up remaining time value for free. A deep in-the-money American
  put is a different story even with no dividend (locking in the strike
  early starts earning interest on it sooner), and once a dividend is
  added, American calls pick up a premium too. `test_sanity.py` checks
  all three of those directly instead of assuming they hold.

### `market_data.py` - live data


- S&P 500 tickers are scraped from Wikipedia with `pandas.read_html`
  (hardcoded 60-ticker fallback if offline).
- `historical_volatility()` computes the classic realized-vol estimate:
  standard deviation of daily log returns, annualized by sqrt(252 trading
  days). That's the sigma the model uses on the market page.
- Spot prices and option chains come from `yfinance`.

### `heatmaps.py` - the grids

`price_grid()` evaluates Black-Scholes over every (spot, vol) combination -
one vectorized NumPy call per row. `heatmap_figure()` renders it with Plotly
so cells respond to mouse-over. PnL mode uses a red-yellow-green colorscale
with the color range forced symmetric around zero, so yellow always sits
exactly on break-even and green/red always mean profit/loss.

### `vol_surface.py` - the smile, measured

Page 3 has always said the market prices a smile while the model uses one
flat volatility. This module turns that sentence into numbers, and three
decisions in it are most of the difference between a surface and a plot of
Yahoo's `impliedVolatility` column.

**The forward comes from put-call parity, not from the spot.** Black-Scholes
needs a forward, and a forward needs a dividend yield and a borrow rate that
nobody publishes. The options are already quoting both: `C - P = e^(-rT)(F - K)`
is a straight line in K, so regressing call-minus-put on strike gives the
discount factor as the slope and the forward from the intercept. No dividend
estimate, no borrow assumption. The fit's R² comes back with the answer, and
a poor fit means the chain is refused rather than quietly built on.

**Only out-of-the-money quotes are used** - puts below the forward, calls
above. In theory both legs carry the same information; in practice the OTM
one is liquid, tighter, and almost all time value, so its price is mostly a
statement about volatility rather than about intrinsic value.

**Mids, and the quotes are filtered first.** `lastPrice` is whenever that
contract last traded, which on a far strike can be days ago at a different
spot. Zero bids, crossed markets and spreads wider than half the mid are
dropped: a quote whose bid-ask straddles ten volatility points does not pin
down a volatility, and averaging it in is how a surface grows spikes that
get explained as skew.

What comes out of SPY on a normal day: ATM vol rising from about 14% at a
week to 17% at two years, a 25-delta risk reversal of +4 to +5 volatility
points at every expiry (downside protection is dearer than upside, which is
the equity skew), a positive butterfly, and forwards that rise with maturity
at roughly the financing rate. The calendar check - total variance must not
fall as maturity rises at fixed moneyness - is run and reported rather than
assumed.

The honest caveat, stated in the module too: US single-name and ETF options
are American and this inverts a European formula. Restricting to OTM quotes
keeps the early-exercise premium small, but it is not zero on deep strikes
and long maturities, so these IVs read slightly high. The fix is inverting
the binomial tree instead, at roughly 500x the cost per quote - a real
trade-off, not an oversight.

### `arbitrage.py` - does the chain price a distribution at all?

`vol_surface.py` checks the time direction: total variance must not fall as
maturity rises, or a calendar spread is free money. `arbitrage.py` checks the
strike direction, where three conditions follow from the payoff alone and need
no model: the call price falls as the strike rises, a call spread never costs
more than it can pay, and the call price is convex in the strike. Puts are
converted to calls through the parity forward first, so one continuous curve
spans the whole strike range.

Convexity is the interesting one, because the second derivative of the call
price *is* the risk-neutral density (Breeden-Litzenberger). A butterfly quoted
at a negative price is the market assigning negative probability to a range of
prices, which never happens - what happened is that three quotes were not alive
at the same instant, or one leg is stale, or the forward is off.

Which is why every violation here is measured against the bid-ask spread of the
legs you would have to trade. On SPY across six expiries:

```
expiry        days  strikes      verticals        butterflies      worst  density
                                raw  tradable     raw  tradable        $     mass
2026-10-01       8      110       1         0      34         0    -0.02    1.000
2026-10-16      22      157       4         0      48         4    -0.10    1.000
2026-11-20      58       82       0         0      11         1    -0.97    0.996
2027-01-29     128      169       8         8      44         5    -1.22    0.983
2027-09-17     358      120       1         1       9         1    -2.03    0.952
2029-01-19     848       80       1         0      28         0    -1.54    0.859
```

**189 conditions violated by the mid prices; 20 by more than the spread.** That
gap is what a screen built on mids with no spread filter reports as
opportunities, and it is the reason those screens produce no trades.

The density column is the implied distribution integrated over the quoted
strikes. It is 1.000 at a week and 0.859 at two years: the far-dated chain
simply does not quote enough of the tails to account for the distribution, so
anything computed from it - an expected value, a tail probability - is missing
14% of its mass, and would read as a confident number if nobody checked.

`arbitrage_study.py` also asks what happens between the quoted strikes, since a
surface quoted at 40 strikes gets used at any strike. Interpolating total
variance and interpolating volatility produce almost identical numbers of bad
butterflies (61 vs 54 on a 200-strike grid at one expiry) - the choice hardly
matters, because a straight line between two quoted IVs inherits whatever
non-convexity the quotes already had. A surface you can price a book with has
to be *fitted* under the convexity constraint rather than joined up dot to dot.
That is the next thing to build here.

```
python arbitrage_study.py          # SPY by default, or pass tickers
```

### The implied-vol solver

`bs.implied_vol` was bisection. It is now Newton-Raphson on vega with a
guarded bisection fallback: **6.4 pricing calls per solve instead of 31**,
same tolerance, which matters once a surface means a thousand inversions per
chain. Every Newton step is checked against the bracket and thrown away if
it lands outside it, so the guarantee bisection gives you is never given up.

Two things changed with it that are worth more than the speed:

- **It converges on sigma, not on price.** Stopping when the price error is
  small sounds right and is wrong for exactly the options where vega is
  small - a deep in-the-money call is worth intrinsic-plus-epsilon at 5% vol
  and at 40% vol alike, so "the price matches to a millionth" can be true a
  long way from the right volatility. The old solver returned the top of its
  search range in those cases, silently.
- **It returns `nan` when the quote determines no volatility at all.** Below
  intrinsic, above the underlying, or vega too small to invert. Real chains
  produce all three constantly, and a fabricated number is how a garbage IV
  ends up plotted as a spike on a surface.

`test_sanity.py` round-trips 400 random inputs across strikes from 60 to 160
and maturities from a week to two years (worst error ~1e-8), checks each
refusal case, and rebuilds a synthetic chain from a known smile to confirm
the surface code recovers the forward, the rate and every strike's
volatility from nothing but prices.

### `hedging.py` - what the price is actually worth

Every price in this repo rests on a claim buried in the derivation: that an
option can be replicated by continuously trading the underlying, so its value
is the cost of that replication and nothing more. Nobody trades continuously.
`hedging.py` runs the replication for real - discretely, on actual price paths -
and `hedging_study.py` does it on eighteen years of SPY.

The setup: short one at-the-money SPY call on the first trading day of every
month, sell it at VIX, delta-hedge daily to expiry 21 trading days later, repeat.
215 non-overlapping trades, 2007 to 2024. Everything below is per $100 of
underlying, so 2008 and 2024 are comparable.

```
average P&L                 0.405        average implied vol        20.01%
average premium sold        2.304        average realized vol       16.20%
std dev of P&L              0.640        implied above realized      82.8% of months
annualized Sharpe            2.20
months profitable           81.4%
worst month                -2.675  (March 2020)
best month                  3.692  (December 2008)
```

That is the variance risk premium, measured rather than cited: implied
volatility averaged 20.0% against 16.2% realized, and selling that gap
systematically returned 0.405 per $100 a month at a Sharpe of 2.20. Before
reading that as a strategy, look at the worst month, and note that the study
sells exactly one option a month with no leverage, no position sizing and no
stop. Selling variance is selling insurance: you are paid a small amount very
reliably, and the occasional bill is enormous. The four worst months here are
March 2020, September 2008, November 2008 and August 2011, which is the same
list a credit desk would give you.

**The point of the exercise is the decomposition, not the Sharpe.** In
continuous time the P&L of a delta-hedged option is exactly

```
integral of  1/2 * Gamma * S^2 * (implied_vol^2 - realized_vol^2) dt
```

so once the delta is hedged away, the direction of the stock is gone and what
remains is a bet on the *difference between two volatilities*, weighted by
gamma. `gamma_pnl()` computes the discrete version term by term. On the real
SPY trades it reproduces the simulated hedge P&L with a correlation of **0.9895**
and a mean absolute error of 0.066 against a 2.30 average premium - the residual
being the third-order terms the expansion drops. This is why traders say "long
gamma" instead of "long calls".

**And the gamma weighting is not a technicality.** Regress each month's P&L on
that month's variance gap alone - implied² minus realized², no gamma - and you
get r² = 0.51 with an intercept of 0.358, which is most of the average P&L.
Add the gamma weighting back and r² goes to 0.98. Half the variation in the
outcome is *not* about how much the stock moved; it is about *when* it moved,
because an option whose spot has drifted away from the strike has almost no
gamma left and stops caring. A delta-hedged option is a path-dependent
approximation to a variance bet, which is the entire reason variance swaps
exist.

**Hedging less often doesn't cost money, it costs certainty.** Boyle & Emanuel
(1980) say the error a discrete hedge adds should have zero mean and a standard
deviation growing like the square root of the rebalancing interval. Measured
against each month's own daily hedge, so the variance premium common to all
frequencies is differenced out:

```
   rebalance   mean P&L   vs daily  error std  / sqrt(n)     worst
    every 1d      0.405          -          -          -    -2.675
    every 2d      0.422     +0.016      0.327      0.231    -2.351
    every 5d      0.378     -0.028      0.659      0.295    -4.382
   every 10d      0.365     -0.041      0.932      0.295    -7.783
   every 21d      0.412     +0.007      1.057      0.231    -4.523
```

The mean change stays inside the noise at every frequency - hedging weekly
instead of daily is not a worse trade, it is the same trade with wider error
bars and a worst case three times as bad. The last column divides out
sqrt(interval) and is flat from 2 to 10 days, then falls off at 21 because the
option is then hedged once at inception and the error has nowhere left to grow.
At 1bp of transaction cost the average P&L goes from 0.405 to 0.382, and at 5bp
to 0.288 - so on this trade, at this size, costs matter less than the choice of
hedging frequency does.

**Caveats, because this one has more than most.** VIX is SPX's implied
volatility, not SPY's, and it is a variance-swap-style index across the whole
strike range rather than the at-the-money vol the study treats it as - it runs
roughly a point above ATM vol, so the measured premium is a touch generous.
Prices are dividend-adjusted and the risk-free rate is zero, folding both carry
terms into the path. The hedge uses the vol the option was sold at and never
re-marks, where a desk would re-hedge on current implied, which damps the tails.
None of these change the shape of any result above; all of them would move the
second decimal place.

The checks in `test_sanity.py` are the part worth reading. A motionless stock
pays the seller the entire premium to within 1e-9. A hedged call and a hedged
put on the same strike earn identical P&L on every path, because with zero rates
their deltas differ by exactly one share held statically - put-call parity
restated as a statement about hedging. Selling at 30% into a stock that realizes
10% wins on 100% of simulated paths and selling at 20% into a stock that realizes
40% loses on 100% of them. And hedging at the path's own volatility has a mean
P&L of zero to within three standard errors, which is the replication argument
itself, checked rather than assumed.

## Things to notice when comparing to the market

- Market prices rarely match the model exactly. The model uses one flat
  historical volatility; the market prices each strike with its own implied
  volatility (the "smile/skew" - downside puts usually carry higher IV
  because crash insurance is in demand). Page 5 measures that gap in dollars
  per contract rather than leaving it as a remark.
- `lastPrice` on illiquid strikes can be hours old - check volume before
  concluding an option is mispriced.
- The closed form prices European exercise only; US single stock options are
  American-style, so deep in-the-money puts on dividend payers will show the
  largest model-vs-market gaps. `binomial.py` prices the American version.
