"""The final config -- wick entry, 20 slots, gap_desc priority, 1-2% bucket
excluded -- re-run on the clean 2yr window (2024-09 -> 2026-09), with and
without the 10bps slippage / $0 commission friction model. Every number on
record before this predates the priority-fix + bucket-exclusion decisions
and the friction model; this is the one trustworthy anchor number
(2yr = negligible survivorship bias, unlike the 10yr confirm).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from slot_and_priority_sweep import (
    BARS_DIR, load_ticker, TickerView, run_sim, MAX_SLOTS,
)

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
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
          f"{calendar[0].date()} -> {calendar[-1].date()} ({WINDOW_YEARS}yr window)")

    arms = [
        ("blend_all_no_friction", "all", 0.0, 0.0),
        ("exclude_1-2pct_no_friction", (2.0, np.inf), 0.0, 0.0),
        ("blend_all_friction", "all", 0.0, 10.0),
        ("exclude_1-2pct_friction", (2.0, np.inf), 0.0, 10.0),
    ]

    rows = []
    for label, scope, commission, slippage_bps in arms:
        print(f"\nrunning {label} (commission=${commission}/fill, slippage={slippage_bps}bps/fill) ...")
        r = run_sim(tickers_data, calendar, "wick", scope,
                    max_slots=MAX_SLOTS, priority="gap_desc",
                    commission_per_trade=commission, slippage_bps=slippage_bps)
        r["trades_df"].to_csv(RESULTS / f"confirm2yr_final_{label}.csv", index=False)
        r["equity_curve"].to_csv(RESULTS / f"confirm2yr_final_{label}_equity.csv")
        ret_dd = (r["total_return_pct"] / abs(r["max_drawdown_pct"])
                  if r["max_drawdown_pct"] else None)
        rows.append({
            "arm": label,
            "n_trades": r["n_trades"],
            "win_rate_%": round(r["win_rate_pct"], 1) if r["win_rate_pct"] is not None else None,
            "total_return_%": round(r["total_return_pct"], 2),
            "max_drawdown_%": round(r["max_drawdown_pct"], 2),
            "ret_per_maxdd": round(ret_dd, 3) if ret_dd else None,
            "final_equity_$": round(r["final_equity"], 2),
        })
        print(f"  n={r['n_trades']}  win%={r['win_rate_pct']}  "
              f"return={r['total_return_pct']:.2f}%  maxDD={r['max_drawdown_pct']:.2f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "confirm2yr_final_summary.csv", index=False)
    print("\n" + "=" * 100)
    print(f"2-YEAR FINAL CONFIG -- wick/gap_desc/20 slots -- {calendar[0].date()} -> {calendar[-1].date()}")
    print("=" * 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
