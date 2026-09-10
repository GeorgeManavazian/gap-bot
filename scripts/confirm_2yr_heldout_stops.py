"""Derives the per-bucket stop-loss table on the 8yr window that precedes
the 2yr held-out window, then runs the anchor arm (wick entry, gap_desc
priority, 20 slots, 1-2% bucket excluded, $0 commission + 10bps slippage)
exactly once against the 2yr window.

The MAE table is built from the training era only, and the test era is
scored a single time: no second look, no re-run with a different stop
table after seeing the result.

Does not modify slot_and_priority_sweep.py; monkeypatches its
stop_for_abs_gap at call time so run_sim() picks up the 8yr-derived table.

Requires pandas >= 3 / numpy >= 2.5 so that np.datetime64(day) inside
TickerView.get() matches the datetime64[ms] Date column in the bar
parquets; under older versions every lookup misses and the sim returns
zero trades.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import slot_and_priority_sweep as sps
from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView, run_sim, MAX_SLOTS

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
HORIZON = 63
TEST_WINDOW_YEARS = 2

BUCKETS = [("1-2%", 1.0, 2.0), ("2-3%", 2.0, 3.0), ("3-5%", 3.0, 5.0),
           ("5-10%", 5.0, 10.0), ("10%+", 10.0, np.inf)]


def bucket_of(abs_gap):
    for name, lo, hi in BUCKETS:
        if lo <= abs_gap < hi:
            return name
    return None


def find_down_gaps_with_mae(df, ticker, train_cutoff):
    """Same MAE logic as sizing_stats.py, but an event only counts if BOTH
    its entry day and its full resolution window fall before train_cutoff --
    otherwise its outcome would depend on test-window data and feed into
    the stop we then test on that same window."""
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    dates = df["Date"].values
    n = len(df)
    out = []
    for t in range(1, n - 1):
        entry_date = pd.Timestamp(dates[t])
        if entry_date >= train_cutoff:
            continue
        prior_close = c[t - 1]
        if prior_close <= 0:
            continue
        gap_pct = (o[t] - prior_close) / prior_close * 100
        if gap_pct >= -1.0:
            continue
        entry = o[t]
        end = min(t + HORIZON, n - 1)
        if pd.Timestamp(dates[end]) >= train_cutoff:
            continue  # resolution window would poke into the test era
        fill_day = None
        for k in range(t, end + 1):
            if h[k] >= prior_close:
                fill_day = k - t
                break
        mae_end = t + fill_day if fill_day is not None else end
        worst_low = l[t:mae_end + 1].min()
        mae_pct = (worst_low - entry) / entry * 100
        out.append({"ticker": ticker, "date": entry_date,
                     "bucket": bucket_of(abs(gap_pct)),
                     "fill_day": fill_day, "mae_pct": mae_pct})
    return out


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    tickers_data = {}
    dfs = {}
    longest_dates = None
    for p in files:
        d = load_ticker(p)
        if d is None:
            continue
        dfs[p.stem] = d
        tickers_data[p.stem] = TickerView(d)
        if longest_dates is None or len(d) > len(longest_dates):
            longest_dates = d["Date"]
    full_calendar = pd.DatetimeIndex(sorted(longest_dates))
    test_cutoff = full_calendar[-1] - pd.DateOffset(years=TEST_WINDOW_YEARS)

    print(f"full range {full_calendar[0].date()} -> {full_calendar[-1].date()}")
    print(f"8yr TRAIN era: < {test_cutoff.date()}   |   2yr TEST (held out) era: >= {test_cutoff.date()}")

    # --- derive stop table from TRAIN era only --------------------------------
    all_rows = []
    for tk, df in dfs.items():
        all_rows.extend(find_down_gaps_with_mae(df, tk, test_cutoff))
    g = pd.DataFrame(all_rows)
    resolved = g[g["fill_day"].notna()]

    new_stops = {}
    print("\n8yr-derived MAE / stop table (train era only):")
    print(f"{'bucket':8s} {'n_resolved':>10s} {'p90_mae_%':>10s} {'old_stop':>9s} {'new_stop':>9s}")
    old_stops = {"1-2%": 8.0, "2-3%": 11.0, "3-5%": 15.0, "5-10%": 24.0, "10%+": 35.0}
    for name, lo, hi in BUCKETS:
        sub = resolved[resolved["bucket"] == name]
        p90_mae = sub["mae_pct"].quantile(0.10)  # 10th pctile of a negative number = p90 adverse move
        stop = round(abs(p90_mae))
        new_stops[name] = float(stop)
        print(f"{name:8s} {len(sub):10d} {p90_mae:10.2f} {old_stops[name]:9.1f} {stop:9.1f}")

    def stop_for_abs_gap_heldout(abs_gap):
        for name, lo, hi in BUCKETS:
            if lo <= abs_gap < hi:
                return new_stops[name]
        return None

    # monkeypatch -- run_sim() looks up stop_for_abs_gap as a module global
    # at call time, so this redirects it without touching the shared file.
    sps.stop_for_abs_gap = stop_for_abs_gap_heldout

    # --- run the anchor arm ONCE against the untouched 2yr test era ----------
    test_calendar = full_calendar[full_calendar >= test_cutoff]
    print(f"\nrunning HELD-OUT test: wick / gap_desc / 20 slots / exclude 1-2% / "
          f"$0 commission + 10bps slippage, {test_calendar[0].date()} -> {test_calendar[-1].date()} "
          f"-- ONE SHOT, no re-runs")
    r = run_sim(tickers_data, test_calendar, "wick", (2.0, np.inf),
                max_slots=MAX_SLOTS, priority="gap_desc",
                commission_per_trade=0.0, slippage_bps=10.0)

    r["trades_df"].to_csv(RESULTS / "confirm2yr_heldout_stops_exclude_1-2pct_friction.csv", index=False)
    r["equity_curve"].to_csv(RESULTS / "confirm2yr_heldout_stops_exclude_1-2pct_friction_equity.csv")

    ret_dd = (r["total_return_pct"] / abs(r["max_drawdown_pct"])
              if r["max_drawdown_pct"] else None)
    print("\n" + "=" * 100)
    print("HELD-OUT RESULT (8yr-derived stops, tested once on the 2yr window) "
          "vs the in-sample-stop arm")
    print("=" * 100)
    print(f"{'':28s} {'n_trades':>9s} {'win_%':>7s} {'return_%':>9s} {'maxDD_%':>8s} {'ret/maxdd':>10s}")
    print(f"{'IN-SAMPLE STOPS':28s} {501:9d} {80.0:7.1f} {65.42:9.2f} {-17.69:8.2f} {3.697:10.3f}")
    print(f"{'HELD-OUT (8yr-derived)':28s} {r['n_trades']:9d} "
          f"{r['win_rate_pct']:7.1f} {r['total_return_pct']:9.2f} "
          f"{r['max_drawdown_pct']:8.2f} {ret_dd:10.3f}")


if __name__ == "__main__":
    main()
