#!/usr/bin/env python
"""Run this against the REAL Schwab token before trusting live/schwab_options.py
in run_daily.py -- it has never been called against a live account, only
exercised via the synthetic fixture in --dry-run. Prints what it got so a
human can eyeball whether the shape matches what's assumed (bid/ask/delta/
symbol per contract). Fails loudly and specifically rather than silently,
unlike fetch_chain's own doctrine of returning None on any error -- this
script's whole job is to surface exactly what breaks, if anything.

Usage: cd code/gap-bot && ../etf-bot/.venv-live/bin/python scripts/smoke_test_options.py [TICKER]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.schwab_data import get_client
from live.schwab_options import fetch_chain, pick_by_delta_dte, pick_by_strike_near, quote_by_symbol

ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"

print(f"Requesting Schwab client...")
client = get_client()

print(f"Fetching live option chain for {ticker}...")
chain = fetch_chain(client, ticker)
if chain is None:
    print("FAILED: fetch_chain returned None -- either the HTTP call failed, "
          "the response had no callExpDateMap, or nothing had a two-sided quote. "
          "Add a print of r.status_code / r.json() keys directly in fetch_chain "
          "to see which.")
    sys.exit(1)

print(f"OK: underlying={chain['underlying']}, {len(chain['calls'])} quotable calls")
print("Sample contracts (first 5):")
for c in chain["calls"][:5]:
    print(f"  strike={c['strike']} expiry={c['expiry']} dte={c['dte']} "
          f"bid={c['bid']} ask={c['ask']} delta={c['delta']} symbol={c['symbol']}")

print("\nTesting pick_by_delta_dte(target_dte=14, target_delta=0.30)...")
picked = pick_by_delta_dte(chain, 14, 0.30)
print(f"  -> {picked}")
if picked is None:
    print("  WARNING: no contract had a delta field -- check the JSON key name "
          "schwab-py actually returns (may not be 'delta').")

print("\nTesting pick_by_strike_near(target_dte=45, target_strike=underlying*1.02)...")
target_strike = (chain["underlying"] or 100) * 1.02
picked2 = pick_by_strike_near(chain, 45, target_strike)
print(f"  -> {picked2}")

if picked is not None:
    print(f"\nRe-fetching and looking up the same symbol via quote_by_symbol...")
    chain2 = fetch_chain(client, ticker)
    requoted = quote_by_symbol(chain2, picked["symbol"])
    print(f"  -> {requoted}")
    if requoted is None:
        print("  WARNING: couldn't re-find the same contract by symbol on a fresh "
              "pull -- check whether schwab-py's 'symbol' field is stable across calls.")

print("\nDone. If every section above printed real numbers (not None/WARNING), "
      "schwab_options.py is safe to enable on the VPS.")
