"""10-year confirm of the 2yr headline result (wick entry, 20 slots,
priority=gap_desc), plus the capital-allocation design question: keep
blending the 1-2% bucket in (soft-bias via priority, filler on quiet
days) vs hard-exclude it from the pool entirely -- now with friction.

Four arms, same window, same everything else:
  A) wick/all (1%+, all 5 buckets), no friction        -- 2026-09-07 baseline
  B) wick/2%+ (4 buckets, 1-2% excluded), no friction   -- 2026-09-07 baseline
  C) same as A, + $0 commission (Schwab) + 10bps slippage per fill
  D) same as B, + $0 commission (Schwab) + 10bps slippage per fill

Slippage is a stand-in for spread/market-impact cost on a real fill;
Schwab charges $0 commission on stock trades, so commission is modeled
explicitly at $0 rather than skipped -- the friction lever that actually
moves these numbers is slippage.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from slot_and_priority_sweep import (
    BARS_DIR, load_ticker, TickerView, run_sim, MAX_SLOTS,
)

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"


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
    print(f"{len(tickers_data)} tickers, {len(calendar)} trading days, "
          f"{calendar[0].date()} -> {calendar[-1].date()} (full window)")

    arms = [
        ("A_all_1pct+", "all", 0.0, 0.0),
        ("B_excl_1-2pct", (2.0, np.inf), 0.0, 0.0),
        ("C_all_1pct+_friction", "all", 0.0, 10.0),
        ("D_excl_1-2pct_friction", (2.0, np.inf), 0.0, 10.0),
    ]

    rows = []
    for label, scope, commission, slippage_bps in arms:
        print(f"\nrunning {label} (commission=${commission}/fill, slippage={slippage_bps}bps/fill) ...")
        r = run_sim(tickers_data, calendar, "wick", scope,
                    max_slots=MAX_SLOTS, priority="gap_desc",
                    commission_per_trade=commission, slippage_bps=slippage_bps)
        r["trades_df"].to_csv(RESULTS / f"confirm10yr_{label}.csv", index=False)
        r["equity_curve"].to_csv(RESULTS / f"confirm10yr_{label}_equity.csv")
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
              f"return={r['total_return_pct']:.2f}%  "
              f"maxDD={r['max_drawdown_pct']:.2f}%  "
              f"ret/maxDD={ret_dd:.3f}" if ret_dd else "")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "confirm10yr_summary.csv", index=False)
    print("\n" + "=" * 100)
    print("10-YEAR CONFIRM -- wick/gap_desc/20 slots -- blend-all vs exclude-1-2%")
    print("=" * 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
