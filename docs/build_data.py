"""Builds docs/data.js for the GitHub Pages site.

Everything else on that page is computed in the browser from a JavaScript
port of the pricers, so it needs no data at all. The volatility surface is
the exception: it is a statement about what the market was quoting, and the
only honest way to put one on a page is to go and get it.

So this pulls a live SPY chain, runs it through vol_surface.py - the same
code the Streamlit app uses - and writes the result out with the timestamp
it was taken at. Re-run it and the page updates:

    python docs/build_data.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import vol_surface as vsurf  # noqa: E402

TICKER = "SPY"
N_EXPIRIES = 6
OUT = Path(__file__).resolve().parent / "data.js"


def main():
    print(f"pulling {TICKER} chains...")
    surface = vsurf.build_surface(TICKER, max_expiries=N_EXPIRIES, verbose=True)
    if surface.empty:
        raise SystemExit("no usable quotes - Yahoo is probably rate limiting")

    terms = vsurf.term_structure(surface)

    smiles = []
    for expiry, smile in surface.groupby("expiry", sort=False):
        s = smile.sort_values("moneyness")
        smiles.append({
            "expiry": expiry,
            "days": round(float(s["T"].iloc[0]) * 365),
            "forward": round(float(s["forward"].iloc[0]), 2),
            "points": [[round(float(m), 4), round(float(v), 4)]
                       for m, v in zip(s["moneyness"], s["iv"])],
        })

    front = surface[surface["expiry"] == terms.iloc[0]["expiry"]]
    flat = float(front["iv"].median())
    err = vsurf.flat_vol_error(front, flat)

    data = {
        "ticker": TICKER,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_quotes": int(len(surface)),
        "smiles": smiles,
        "terms": [{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                   for k, v in row.items()} for row in terms.to_dict("records")],
        "flat": {
            "expiry": terms.iloc[0]["expiry"],
            "sigma": round(flat, 4),
            "points": [[round(float(m), 4), round(float(e), 3)]
                       for m, e in zip(err["moneyness"], err["error"])],
            "worst": round(float(err["error"].abs().max()), 2),
            "median_abs": round(float(err["error"].abs().median()), 3),
        },
        "calendar_violations": int(len(vsurf.calendar_arbitrage(surface))),
    }

    OUT.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  {data['n_quotes']} quotes over {len(smiles)} expiries, "
          f"{data['calendar_violations']} calendar violations")
    for row in data["terms"]:
        print(f"  {row['expiry']}  atm {row['atm']:.2%}  "
              f"RR {row['risk_reversal']:+.2%}  fly {row['butterfly']:+.2%}")


if __name__ == "__main__":
    main()
