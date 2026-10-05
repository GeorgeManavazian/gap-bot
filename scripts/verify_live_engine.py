"""Regression check: replay the 2yr held-out window day-by-day through
live/engine.step_one_day() (the same code run_daily.py calls) and confirm
it reproduces the anchor backtest's numbers exactly. If this doesn't
match, the live port differs from the backtest and is not trusted with
real data; this is the one test that has to pass before the live bot is real.

Reference run (results/priority_lookahead_check.csv, row "rest_top_n +
resting limit semantics", adopted 2026-09-08): 174 trades, +11.45% return,
-14.59% maxDD, 64.4% win rate. Fills book only where low <= entry <= high,
and orders rest on the top-N gaps at the open.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView, ADV_LOOKBACK
from live.engine import step_one_day
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler

WINDOW_YEARS = 2


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    tickers_data = {}
    longest_dates = None
    for p in files:
        d = load_ticker(p)
        if d is None:
            continue
        tickers_data[p.stem] = TickerView(d)
        if longest_dates is None or len(d) > len(longest_dates):
            longest_dates = d["Date"]
    calendar = pd.DatetimeIndex(sorted(longest_dates))
    cutoff = calendar[-1] - pd.DateOffset(years=WINDOW_YEARS)
    calendar = calendar[calendar >= cutoff]
    print(f"{len(tickers_data)} tickers, {len(calendar)} trading days, "
          f"{calendar[0].date()} -> {calendar[-1].date()}")

    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    # The stock filler IS the anchor economics (2026-09-08 three-variant
    # refactor moved them out of engine.py); the anchor numbers above are
    # stock-only, so that's the only variant this parity check can score.
    filler = make_filler("stock", VARIANTS["stock"])
    all_trades = []
    equity_curve = []

    for day in calendar:
        bars = {}
        for tk, tv in tickers_data.items():
            row = tv.get(day)
            if row is None:
                continue
            o, h, l, c = row
            pr = tv.get_prior_close(day)
            adv = tv.get_adv(day)
            bars[tk] = {"o": float(o), "h": float(h), "l": float(l), "c": float(c),
                       "prior_close": None if pr is None else float(pr), "adv": adv}
        result = step_one_day(state, day.isoformat()[:10], bars, filler)
        all_trades.extend(result["trades"])
        equity_curve.append(result["equity"])

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    tdf = pd.DataFrame(all_trades)
    total_return = (eq.iloc[-1] / CAPITAL - 1) * 100
    if not len(tdf):
        # 0 trades is a data-load failure, not a result: the cached parquets
        # store Date as datetime64[ms] and, where np.datetime64(Timestamp)
        # yields microseconds, every TickerView lookup silently misses.
        print("\nNO TRADES -- bars did not load (Date dtype mismatch, ms vs us). Cast Date to "
              "datetime64[us] in a wrapper around slot_and_priority_sweep.load_ticker and rerun "
              "(see scripts/never_stop_expand_slots.py). Not a MATCH, not a MISMATCH: unverified.")
        return 2
    win_rate = 100 * (tdf["pnl_pct"] > 0).mean()

    print(f"\nlive-engine replay: n={len(tdf)}  win%={win_rate:.1f}  "
          f"return={total_return:.2f}%  maxDD={dd.min()*100:.2f}%")
    print("anchor to match:    n=174  win%=64.4  return=11.45%  maxDD=-14.59%")

    ok = (len(tdf) == 174 and abs(total_return - 11.45) < 0.05
          and abs(dd.min() * 100 - (-14.59)) < 0.05)
    print("\n" + ("MATCH -- live engine reproduces the anchor exactly." if ok
                  else "MISMATCH -- live engine has a bug, do not trust it yet."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
