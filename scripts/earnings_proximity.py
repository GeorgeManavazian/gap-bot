"""Read-only analysis: do gap trades placed near an earnings announcement
behave differently from the rest? Informs whether an earnings-blackout gate
is worth considering. Does NOT implement a gate.

For each trade, the ticker's nearest earnings_date to gap_date (the decision
point, not the later entry_date) gives a signed calendar-day distance:
negative = gap happened before that report, positive = after. Trades are
bucketed by |distance|: 0-1d (the earnings reaction gap itself), 2-3d,
4-10d, far (11+d or nothing within +-60d). Tickers absent from the calendar
(renames / spinoffs) are reported as their own "no calendar" row, never
folded into "far".

Logs analysed:
  2yr anchor (trust for magnitude):
    results/confirm2yr_heldout_stops_exclude_1-2pct_friction.csv
  10yr same config (bigger N, survivorship-inflated -- direction only):
    results/slot_priority_sensitivity_10yr_slots20_gap_desc.csv

Calendar: etf-bot/data/earnings/calendar.parquet (ticker, earnings_date).
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
CALENDAR = Path("/home/user/Documents/Trading/code/etf-bot/data/earnings/calendar.parquet")
FAR_LIMIT = 60

LOGS = {
    "2yr_anchor": RESULTS / "confirm2yr_heldout_stops_exclude_1-2pct_friction.csv",
    "10yr_same_config": RESULTS / "slot_priority_sensitivity_10yr_slots20_gap_desc.csv",
}
PROX_ORDER = ["0-1d", "2-3d", "4-10d", "far", "no calendar"]
GAP_ORDER = ["2-3%", "3-5%", "5-10%", "10%+"]


def attach_proximity(trades, cal):
    by_tk = {tk: np.sort(g["earnings_date"].values) for tk, g in cal.groupby("ticker")}
    dist = np.full(len(trades), np.nan)
    for i, (tk, gd) in enumerate(zip(trades["ticker"], trades["gap_date"].values)):
        dates = by_tk.get(tk)
        if dates is None or len(dates) == 0:
            continue
        j = np.searchsorted(dates, gd)
        cands = [dates[k] for k in (j - 1, j) if 0 <= k < len(dates)]
        nearest = min(cands, key=lambda d: abs((gd - d) / np.timedelta64(1, "D")))
        dist[i] = (gd - nearest) / np.timedelta64(1, "D")
    trades = trades.copy()
    trades["days_from_earnings"] = dist
    a = np.abs(dist)
    prox = np.where(np.isnan(dist), "no calendar",
            np.where(a <= 1, "0-1d",
            np.where(a <= 3, "2-3d",
            np.where(a <= 10, "4-10d", "far"))))
    trades["proximity"] = pd.Categorical(prox, categories=PROX_ORDER)
    side = np.where(np.isnan(dist), "n/a", np.where(dist < 0, "before", np.where(dist > 0, "after", "same day")))
    trades["side"] = side
    return trades


def bucket_table(t):
    rows = []
    for p in PROX_ORDER:
        s = t[t["proximity"] == p]
        if not len(s):
            continue
        ex = s["exit_reason"].value_counts(normalize=True) * 100
        gs = s["bucket"].value_counts(normalize=True) * 100
        rows.append({
            "proximity": p, "n": len(s), "pct_of_trades": round(100 * len(s) / len(t), 1),
            "win_%": round(100 * (s["pnl_pct"] > 0).mean(), 1),
            "avg_pnl_%": round(s["pnl_pct"].mean(), 2),
            "median_pnl_%": round(s["pnl_pct"].median(), 2),
            "sum_pnl_$": round(s["pnl_dollar"].sum(), 0),
            "pct_of_total_pnl_$": round(100 * s["pnl_dollar"].sum() / t["pnl_dollar"].sum(), 1),
            "stop_%": round(ex.get("stop", 0), 1),
            "tp_%": round(ex.get("tp_gap_filled", 0), 1),
            "time_%": round(ex.get("time_exit", 0), 1),
            "avg_days_held": round(s["days_held"].mean(), 1),
            "gap_2-3%_%": round(gs.get("2-3%", 0), 1),
            "gap_3-5%_%": round(gs.get("3-5%", 0), 1),
            "gap_5-10%_%": round(gs.get("5-10%", 0), 1),
            "gap_10%+_%": round(gs.get("10%+", 0), 1),
            "dominant_gap_bucket": gs.idxmax(),
        })
    return pd.DataFrame(rows)


def gap_by_prox_table(t):
    """Within each gap-size bucket: what share of trades and $P&L sits in each
    proximity bucket, and how near-earnings vs far trades perform there."""
    rows = []
    for gb in GAP_ORDER:
        s = t[t["bucket"] == gb]
        if not len(s):
            continue
        near = s[s["proximity"] == "0-1d"]
        far = s[s["proximity"] == "far"]
        rows.append({
            "gap_bucket": gb, "n": len(s), "sum_pnl_$": round(s["pnl_dollar"].sum(), 0),
            "n_0-1d": len(near), "pct_trades_0-1d": round(100 * len(near) / len(s), 1),
            "pct_pnl$_0-1d": round(100 * near["pnl_dollar"].sum() / s["pnl_dollar"].sum(), 1) if s["pnl_dollar"].sum() else np.nan,
            "win_%_0-1d": round(100 * (near["pnl_pct"] > 0).mean(), 1) if len(near) else np.nan,
            "avg_pnl_%_0-1d": round(near["pnl_pct"].mean(), 2) if len(near) else np.nan,
            "n_far": len(far),
            "win_%_far": round(100 * (far["pnl_pct"] > 0).mean(), 1) if len(far) else np.nan,
            "avg_pnl_%_far": round(far["pnl_pct"].mean(), 2) if len(far) else np.nan,
        })
    return pd.DataFrame(rows)


def side_table(t):
    s = t[t["proximity"].isin(["2-3d", "4-10d"])]
    rows = []
    for (p, side), g in s.groupby(["proximity", "side"], observed=True):
        rows.append({"proximity": p, "side": side, "n": len(g),
                     "win_%": round(100 * (g["pnl_pct"] > 0).mean(), 1),
                     "avg_pnl_%": round(g["pnl_pct"].mean(), 2)})
    return pd.DataFrame(rows)


def main():
    cal = pd.read_parquet(CALENDAR)
    cal["earnings_date"] = pd.to_datetime(cal["earnings_date"])

    for label, path in LOGS.items():
        t = pd.read_csv(path, parse_dates=["gap_date", "entry_date", "exit_date"])
        t = attach_proximity(t, cal)
        t.to_csv(RESULTS / f"earnings_proximity_{label}_trades.csv", index=False)

        bt = bucket_table(t)
        gt = gap_by_prox_table(t)
        st = side_table(t)
        bt.to_csv(RESULTS / f"earnings_proximity_{label}_by_proximity.csv", index=False)
        gt.to_csv(RESULTS / f"earnings_proximity_{label}_by_gap_bucket.csv", index=False)
        st.to_csv(RESULTS / f"earnings_proximity_{label}_before_after.csv", index=False)

        far_none = int(((t["proximity"] == "far") & (t["days_from_earnings"].abs() > FAR_LIMIT)).sum())
        print("\n" + "=" * 120)
        print(f"{label}: {len(t)} trades, {t['gap_date'].min().date()} -> {t['gap_date'].max().date()}, "
              f"total pnl ${t['pnl_dollar'].sum():,.0f}, "
              f"'far' includes {far_none} with no earnings within +-{FAR_LIMIT}d")
        print("=" * 120)
        print("\nBY EARNINGS PROXIMITY (|gap_date - nearest earnings_date|):")
        print(bt.to_string(index=False))
        print("\nBY GAP-SIZE BUCKET -- how much of each sits in the 0-1d earnings gap, and 0-1d vs far performance within it:")
        print(gt.to_string(index=False))
        print("\nBEFORE vs AFTER earnings, 2-3d and 4-10d buckets:")
        print(st.to_string(index=False))


if __name__ == "__main__":
    main()
