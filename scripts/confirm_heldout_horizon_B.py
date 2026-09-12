"""Confirmation check: the per-bucket horizon-B timeout (2-3%->30d, 3-5%->35d,
5-10%->50d, 10%+->63d) scored on the 2yr held-out window with the 8yr-derived
stop table from confirm_2yr_heldout_stops.py (1-2%=7, 2-3%=11, 3-5%=16,
5-10%=26, 10%+=37).

Horizon-B's values were fixed in advance, so this is a one-shot confirmation
of a pre-committed config, not a tune. It is the second touch of the 2yr
held-out window with these stops (the first was the flat-63 anchor run in
confirm_2yr_heldout_stops.py), recorded here so nobody mistakes it for a
clean first look.

The stop table is rebuilt by calling the other script's own
find_down_gaps_with_mae() on the same train cutoff (imported, not re-derived
by hand), then asserted against the expected table, so a drift in either
would fail loudly instead of silently scoring the wrong stops.

Reference run (flat 63-day timeout, 8yr-derived stops, 2yr, friction on):
  results/confirm2yr_heldout_stops_exclude_1-2pct_friction.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView, MAX_SLOTS
from confirm_2yr_heldout_stops import BUCKETS, find_down_gaps_with_mae
from bucket_timeout_sweep import run_sim_bucket_horizon, summarize, WINDOW_YEARS

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"

EXPECTED_STOPS = {"1-2%": 7.0, "2-3%": 11.0, "3-5%": 16.0, "5-10%": 26.0, "10%+": 37.0}
HORIZON_B = {"2-3%": 30, "3-5%": 35, "5-10%": 50, "10%+": 63}
ANCHOR = {"n_trades": 508, "total_return_%": 72.34,
          "max_drawdown_%": -18.33, "ret_per_maxdd": 3.947}


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
    full_calendar = pd.DatetimeIndex(sorted(longest_dates))
    test_cutoff = full_calendar[-1] - pd.DateOffset(years=WINDOW_YEARS)
    test_calendar = full_calendar[full_calendar >= test_cutoff]

    rows = []
    for tk, df in dfs.items():
        rows.extend(find_down_gaps_with_mae(df, tk, test_cutoff))
    g = pd.DataFrame(rows)
    resolved = g[g["fill_day"].notna()]
    stops = {name: float(round(abs(resolved[resolved["bucket"] == name]["mae_pct"].quantile(0.10))))
             for name, _, _ in BUCKETS}
    print(f"8yr-derived stops (train < {test_cutoff.date()}): {stops}")
    assert stops == EXPECTED_STOPS, f"stop table drifted from hub's report: {stops} != {EXPECTED_STOPS}"

    def stop_fn(abs_gap):
        for name, lo, hi in BUCKETS:
            if lo <= abs_gap < hi:
                return stops[name]
        return None

    common = dict(max_slots=MAX_SLOTS, priority="gap_desc",
                  commission_per_trade=0.0, slippage_bps=10.0, stop_fn=stop_fn)
    scope = (2.0, np.inf)

    # Sanity: this variant sim with an empty horizon map must match the
    # flat-63 anchor from confirm_2yr_heldout_stops.py exactly, or the
    # comparison is invalid.
    print(f"\nreproducing the anchor (flat 63) through run_sim_bucket_horizon ...")
    ra = run_sim_bucket_horizon(tickers_data, test_calendar, "wick", scope, {}, **common)
    anchor = summarize("anchor_flat63", ra)
    print(f"  n={anchor['n_trades']}  return={anchor['total_return_%']}%  "
          f"maxDD={anchor['max_drawdown_%']}%  ret/maxdd={anchor['ret_per_maxdd']}")
    for k, v in ANCHOR.items():
        assert anchor[k] == v, f"anchor reproduction failed on {k}: {anchor[k]} != {v}"

    print(f"\nrunning horizon-B {HORIZON_B} with 8yr-derived stops -- one shot ...")
    rb = run_sim_bucket_horizon(tickers_data, test_calendar, "wick", scope, HORIZON_B, **common)
    label = "heldout_stops_horizon_B_30_35_50_63"
    rb["trades_df"].to_csv(RESULTS / f"bucket_timeout_{label}.csv", index=False)
    rb["equity_curve"].to_csv(RESULTS / f"bucket_timeout_{label}_equity.csv")
    b = summarize(label, rb)

    summary = pd.DataFrame([anchor, b])
    summary.to_csv(RESULTS / "bucket_timeout_heldout_stops_summary.csv", index=False)
    print("\n" + "=" * 110)
    print(f"HORIZON-B vs ANCHOR -- 8yr-derived stops, wick/gap_desc/20 slots/excl 1-2%/"
          f"$0+10bps, {test_calendar[0].date()} -> {test_calendar[-1].date()}")
    print("=" * 110)
    print(summary.to_string(index=False))

    print("\nper-bucket exit mix, horizon-B:")
    print(pd.crosstab(rb["trades_df"]["bucket"], rb["trades_df"]["exit_reason"]).to_string())
    print("\nper-bucket exit mix, anchor:")
    print(pd.crosstab(ra["trades_df"]["bucket"], ra["trades_df"]["exit_reason"]).to_string())


if __name__ == "__main__":
    main()
