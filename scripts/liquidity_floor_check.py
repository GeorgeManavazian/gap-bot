"""Correctness fix, not a parameter hunt: does a 3%-of-ADV liquidity floor
change anything, and by how much? Currently the sim sizes every slot at
1/20th of equity (~$5k-$10k+) into whatever gapped, with zero check on
whether the ticker could actually absorb that size at the assumed 10bps
slippage -- a thin, low-volume name slips far more than 10bps at that size
in real life. The sim couldn't tell the difference until now.

Runs on the full 10yr window (NOT the 2yr held-out window -- that one's
spent, touched twice already for the stop-loss-leakage validation). The
10yr window is already flagged elsewhere as "playground" due to
survivorship bias, so it's fine for a DIRECTIONAL check (does the floor
change results, which way, roughly how much) -- not a precise magnitude
claim.

Stops reused as-is from the 8yr-derived table in confirm_2yr_heldout_stops.py
(not rederived here -- that script is the one source of truth for it).

One threshold only (3% of trailing 20-day dollar volume, look-back only,
today's own volume excluded) -- no grid search.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import slot_and_priority_sweep as sps
from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView, run_sim, MAX_SLOTS

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"

# 8yr-derived stop table -- reused verbatim from confirm_2yr_heldout_stops.py,
# NOT rederived here.
EIGHT_YR_STOPS = {"1-2%": 7.0, "2-3%": 11.0, "3-5%": 16.0, "5-10%": 26.0, "10%+": 37.0}
BUCKET_RANGES = [("1-2%", 1.0, 2.0), ("2-3%", 2.0, 3.0), ("3-5%", 3.0, 5.0),
                  ("5-10%", 5.0, 10.0), ("10%+", 10.0, np.inf)]


def _stop_for_abs_gap_8yr(abs_gap):
    for name, lo, hi in BUCKET_RANGES:
        if lo <= abs_gap < hi:
            return EIGHT_YR_STOPS[name]
    return None


sps.stop_for_abs_gap = _stop_for_abs_gap_8yr

ADV_CAP_PCT = 0.03  # 3% of trailing 20-day dollar volume


def mean_entry_adv(tdf, tickers_data):
    if len(tdf) == 0:
        return None
    advs = []
    for _, row in tdf.iterrows():
        adv = tickers_data[row["ticker"]].get_adv(pd.Timestamp(row["entry_date"]))
        if adv is not None:
            advs.append(adv)
    return float(np.mean(advs)) if advs else None


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
          f"{calendar[0].date()} -> {calendar[-1].date()} (full 10yr, playground)")

    common = dict(entry_style="wick", scope_lo_hi=(2.0, np.inf),
                  max_slots=MAX_SLOTS, priority="gap_desc",
                  commission_per_trade=0.0, slippage_bps=10.0)

    print("\nrunning WITHOUT liquidity floor ...")
    r_off = run_sim(tickers_data, calendar, max_pct_of_adv=None, **common)
    print("\nrunning WITH liquidity floor (3% of trailing 20d $ volume) ...")
    r_on = run_sim(tickers_data, calendar, max_pct_of_adv=ADV_CAP_PCT, **common)

    r_off["trades_df"].to_csv(RESULTS / "liquidity_floor_off_10yr.csv", index=False)
    r_off["equity_curve"].to_csv(RESULTS / "liquidity_floor_off_10yr_equity.csv")
    r_on["trades_df"].to_csv(RESULTS / "liquidity_floor_on_3pct_10yr.csv", index=False)
    r_on["equity_curve"].to_csv(RESULTS / "liquidity_floor_on_3pct_10yr_equity.csv")

    def ret_dd(r):
        return r["total_return_pct"] / abs(r["max_drawdown_pct"]) if r["max_drawdown_pct"] else None

    adv_off = mean_entry_adv(r_off["trades_df"], tickers_data)
    adv_on = mean_entry_adv(r_on["trades_df"], tickers_data)

    summary = pd.DataFrame([
        {"variant": "floor_off", "n_trades": r_off["n_trades"],
         "n_illiquid_rejected": r_off["n_illiquid_rejected"],
         "win_rate_%": round(r_off["win_rate_pct"], 1) if r_off["win_rate_pct"] is not None else None,
         "total_return_%": round(r_off["total_return_pct"], 2),
         "max_drawdown_%": round(r_off["max_drawdown_pct"], 2),
         "ret_per_maxdd": round(ret_dd(r_off), 3) if ret_dd(r_off) else None,
         "mean_entry_adv_$": round(adv_off, 0) if adv_off else None},
        {"variant": "floor_on_3pct", "n_trades": r_on["n_trades"],
         "n_illiquid_rejected": r_on["n_illiquid_rejected"],
         "win_rate_%": round(r_on["win_rate_pct"], 1) if r_on["win_rate_pct"] is not None else None,
         "total_return_%": round(r_on["total_return_pct"], 2),
         "max_drawdown_%": round(r_on["max_drawdown_pct"], 2),
         "ret_per_maxdd": round(ret_dd(r_on), 3) if ret_dd(r_on) else None,
         "mean_entry_adv_$": round(adv_on, 0) if adv_on else None},
    ])
    summary.to_csv(RESULTS / "liquidity_floor_summary_10yr.csv", index=False)
    print("\n" + "=" * 110)
    print("LIQUIDITY FLOOR CHECK -- 10yr, wick/gap_desc/20 slots/exclude 1-2%/8yr-derived stops/10bps slip")
    print("=" * 110)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
