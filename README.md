# Options Pricer

![tests](https://github.com/mihircoding/options-pricer/actions/workflows/ci.yml/badge.svg)

A Black-Scholes options pricing tool with an interactive Streamlit interface.
Prices calls and puts from scratch (with dividend adjustment), computes the
Greeks, builds ten classic multi-leg strategies, cross-checks the closed form
against a Monte Carlo simulator, and compares model prices against live
S&P 500 option quotes from Yahoo Finance - including the volatility smile
backed out of real market prices.

Built with Python, NumPy, SciPy, Matplotlib, Plotly and Streamlit.

## Running it locally

On Windows, just double-click `run.bat`. Or from a terminal:

```
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Your browser opens at http://localhost:8501 - that IS the website, served
from your machine. `streamlit_app.py` is the landing page; the three tool
pages live in `pages/` and show up in the left sidebar automatically.

Sanity-check the math (textbook values, put-call parity, greeks vs numerical
derivatives, implied-vol round trip):

```
python test_sanity.py
```

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

> Preparing to talk about this project in an interview? See
> [INTERVIEW_PREP.md](INTERVIEW_PREP.md) - a line-by-line walkthrough of the
> important code, the math behind it, and the interview questions each part
> prepares you for.

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

## Things to notice when comparing to the market

- Market prices rarely match the model exactly. The model uses one flat
  historical volatility; the market prices each strike with its own implied
  volatility (the "smile/skew" - downside puts usually carry higher IV
  because crash insurance is in demand).
- `lastPrice` on illiquid strikes can be hours old - check volume before
  concluding an option is mispriced.
- Black-Scholes prices European exercise and ignores dividends; US single
  stock options are American-style, so deep in-the-money puts on dividend
  payers will show the largest model-vs-market gaps.
