"""Sensitivity check on the two hand-picked portfolio knobs: max_slots
{10, 20, 30, 40} x priority {alpha, gap_desc}, 8 cells, on the FULL 10yr
window. Everything else is the anchor: wick entry, 1-2% bucket excluded,
8yr-derived stops from confirm_2yr_heldout_stops.py, $0
commission, 10bps slippage, flat 63-day horizon on both clocks.

Why 10yr and not the 2yr held-out window: the 2yr window has already been
used twice (stop check, horizon-B check) and a parameter grid on it
would turn the exam back into the playground. The 10yr window is inflated
for magnitude (survivorship: today's S&P 500 projected backward) but that
bias is shared by every cell on the same universe, so it's usable for the
relative question "is 20/gap_desc a plateau or a spike" -- and for nothing
else. Do not quote any return% from this file as a headline number.

Note the stop table is derived on the first 8yr, so on this 10yr window the
stops are in-sample for 8 of the 10 years. That's the same for every cell.

Sim: run_sim_bucket_horizon with an empty horizon map, which was asserted
equal to run_sim on the 2yr anchor (508 / 72.34% / -18.33%) in
confirm_heldout_horizon_B.py; used here only because it takes stop_fn as a
parameter instead of needing a monkeypatch.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView
from confirm_2yr_heldout_stops import BUCKETS, find_down_gaps_with_mae
from bucket_timeout_sweep import run_sim_bucket_horizon, summarize

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
PREFIX = "slot_priority_sensitivity_10yr"
TRAIN_CUTOFF_YEARS = 2
EXPECTED_STOPS = {"1-2%": 7.0, "2-3%": 11.0, "3-5%": 16.0, "5-10%": 26.0, "10%+": 37.0}

SLOTS = [10, 20, 30, 40]
PRIORITIES = ["alpha", "gap_desc"]


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    tickers_data, dfs, longest_dates = {}, {}, None
    for p in files:
        d = load_ticker(p)
        if d is None:
            continue
        dfs[p.stem] = d
        tickers_data[p.stem] = TickerView(d)
        if longest_dates is None or len(d) > len(longest_dates):
            longest_dates = d["Date"]
    calendar = pd.DatetimeIndex(sorted(longest_dates))
    train_cutoff = calendar[-1] - pd.DateOffset(years=TRAIN_CUTOFF_YEARS)
    print(f"{len(tickers_data)} tickers, {len(calendar)} trading days, "
          f"{calendar[0].date()} -> {calendar[-1].date()} (FULL window)")

    rows = []
    for tk, df in dfs.items():
        rows.extend(find_down_gaps_with_mae(df, tk, train_cutoff))
    g = pd.DataFrame(rows)
    resolved = g[g["fill_day"].notna()]
    stops = {name: float(round(abs(resolved[resolved["bucket"] == name]["mae_pct"].quantile(0.10))))
             for name, _, _ in BUCKETS}
    assert stops == EXPECTED_STOPS, f"stop table drifted: {stops} != {EXPECTED_STOPS}"
    print(f"stops (8yr-derived, train < {train_cutoff.date()}): {stops}")

    def stop_fn(abs_gap):
        for name, lo, hi in BUCKETS:
            if lo <= abs_gap < hi:
                return stops[name]
        return None

    scope = (2.0, np.inf)
    out = []
    for slots in SLOTS:
        for pri in PRIORITIES:
            label = f"slots{slots}_{pri}"
            print(f"\nrunning {label} ...", flush=True)
            r = run_sim_bucket_horizon(tickers_data, calendar, "wick", scope, {},
                                       max_slots=slots, priority=pri,
                                       commission_per_trade=0.0, slippage_bps=10.0,
                                       stop_fn=stop_fn)
            r["trades_df"].to_csv(RESULTS / f"{PREFIX}_{label}.csv", index=False)
            r["equity_curve"].to_csv(RESULTS / f"{PREFIX}_{label}_equity.csv")
            row = summarize(label, r)
            row["max_slots"] = slots
            row["priority"] = pri
            eq = r["equity_curve"]
            yearly = eq.resample("YE").last()
            yearly_ret = (yearly / yearly.shift(1).fillna(eq.iloc[0]) - 1) * 100
            row["worst_year_%"] = round(yearly_ret.min(), 2)
            row["n_neg_years"] = int((yearly_ret < 0).sum())
            out.append(row)
            print(f"  n={row['n_trades']}  return={row['total_return_%']}%  "
                  f"maxDD={row['max_drawdown_%']}%  ret/maxdd={row['ret_per_maxdd']}  "
                  f"worst_yr={row['worst_year_%']}%", flush=True)

    cols = ["max_slots", "priority", "n_trades", "win_rate_%", "total_return_%",
            "max_drawdown_%", "ret_per_maxdd", "worst_year_%", "n_neg_years",
            "n_stop", "n_tp", "n_time_exit"]
    summary = pd.DataFrame(out)[cols]
    summary.to_csv(RESULTS / f"{PREFIX}_summary.csv", index=False)
    print("\n" + "=" * 120)
    print(f"SLOT x PRIORITY SENSITIVITY -- 10yr {calendar[0].date()} -> {calendar[-1].date()}, "
          f"wick / excl 1-2% / 8yr stops / $0 + 10bps / flat 63")
    print("MAGNITUDES ARE SURVIVORSHIP-INFLATED. Relative shape only.")
    print("=" * 120)
    print(summary.to_string(index=False))
    print("\nret/maxdd grid (rows=slots, cols=priority):")
    print(summary.pivot(index="max_slots", columns="priority", values="ret_per_maxdd").to_string())
    print("\ntotal_return_% grid:")
    print(summary.pivot(index="max_slots", columns="priority", values="total_return_%").to_string())
    print("\nmax_drawdown_% grid:")
    print(summary.pivot(index="max_slots", columns="priority", values="max_drawdown_%").to_string())


if __name__ == "__main__":
    main()
