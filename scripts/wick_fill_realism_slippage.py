"""Part 2 of the wick-fill realism check: extra-slippage sensitivity. The
wick fill assumes a resting order fills at exactly gap_open the instant
High touches it; in reality a single-print touch might not have been a
real fillable level. This doesn't change the fill LOGIC (that would be a
mechanic change, out of scope here) -- it stress-tests how much the
already-reported anchor result degrades if you assume progressively worse
execution than the 10bps used everywhere else this session.

Reuses the 2yr held-out window and the 8yr-derived stop table from
confirm_2yr_heldout_stops.py verbatim -- this is a stress-diagnostic on
the already-reported anchor number (explicitly asked for by name), not a
new tuning pass: no parameter is being chosen based on how this comes
out, nothing here can improve on or replace the anchor.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import slot_and_priority_sweep as sps
from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView, run_sim, MAX_SLOTS

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"

EIGHT_YR_STOPS = {"1-2%": 7.0, "2-3%": 11.0, "3-5%": 16.0, "5-10%": 26.0, "10%+": 37.0}
BUCKET_RANGES = [("1-2%", 1.0, 2.0), ("2-3%", 2.0, 3.0), ("3-5%", 3.0, 5.0),
                  ("5-10%", 5.0, 10.0), ("10%+", 10.0, np.inf)]


def _stop_for_abs_gap_8yr(abs_gap):
    for name, lo, hi in BUCKET_RANGES:
        if lo <= abs_gap < hi:
            return EIGHT_YR_STOPS[name]
    return None


sps.stop_for_abs_gap = _stop_for_abs_gap_8yr

TEST_WINDOW_YEARS = 2
SLIPPAGE_LEVELS_BPS = [10.0, 20.0, 35.0, 50.0]  # 10 = the existing anchor, for reference


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
    full_calendar = pd.DatetimeIndex(sorted(longest_dates))
    cutoff = full_calendar[-1] - pd.DateOffset(years=TEST_WINDOW_YEARS)
    calendar = full_calendar[full_calendar >= cutoff]
    print(f"2yr held-out window (reused for a stress test, not re-tuned): "
          f"{calendar[0].date()} -> {calendar[-1].date()}")

    rows = []
    for bps in SLIPPAGE_LEVELS_BPS:
        r = run_sim(tickers_data, calendar, "wick", (2.0, np.inf),
                    max_slots=MAX_SLOTS, priority="gap_desc",
                    commission_per_trade=0.0, slippage_bps=bps)
        ret_dd = r["total_return_pct"] / abs(r["max_drawdown_pct"]) if r["max_drawdown_pct"] else None
        rows.append({
            "slippage_bps": bps,
            "n_trades": r["n_trades"],
            "win_rate_%": round(r["win_rate_pct"], 1) if r["win_rate_pct"] is not None else None,
            "total_return_%": round(r["total_return_pct"], 2),
            "max_drawdown_%": round(r["max_drawdown_pct"], 2),
            "ret_per_maxdd": round(ret_dd, 3) if ret_dd else None,
        })
        print(f"  {bps:5.0f}bps: n={r['n_trades']}  win%={r['win_rate_pct']:.1f}  "
              f"return={r['total_return_pct']:.2f}%  maxDD={r['max_drawdown_pct']:.2f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "wick_fill_realism_slippage_sensitivity.csv", index=False)
    print("\n" + "=" * 90)
    print("SLIPPAGE SENSITIVITY (10bps = existing anchor, others = new TOTAL, not additive)")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
