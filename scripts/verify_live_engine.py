"""Regression check: replay the 2yr held-out window day-by-day through
live/engine.step_one_day() (the same code run_daily.py calls) and confirm
it reproduces the committed 2-year run exactly. A mismatch means the
live port does not match the backtest and should not run against real
data until it does.

Reference run: results/confirm2yr_heldout_stops_exclude_1-2pct_friction.csv
(the expected trade count, return and drawdown are the constants below).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView, ADV_LOOKBACK
from live.engine import step_one_day
from live.config import CAPITAL

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
        result = step_one_day(state, day.isoformat()[:10], bars)
        all_trades.extend(result["trades"])
        equity_curve.append(result["equity"])

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    tdf = pd.DataFrame(all_trades)
    total_return = (eq.iloc[-1] / CAPITAL - 1) * 100
    win_rate = 100 * (tdf["pnl_pct"] > 0).mean() if len(tdf) else None

    print(f"\nlive-engine replay: n={len(tdf)}  win%={win_rate:.1f}  "
          f"return={total_return:.2f}%  maxDD={dd.min()*100:.2f}%")
    print("reference run:      n=508  win%=81.1  return=72.34%  maxDD=-18.33%")

    ok = (len(tdf) == 508 and abs(total_return - 72.34) < 0.05
          and abs(dd.min() * 100 - (-18.33)) < 0.05)
    print("\n" + ("MATCH -- live engine matches the anchor exactly." if ok
                  else "MISMATCH -- live engine does not match the backtest."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
