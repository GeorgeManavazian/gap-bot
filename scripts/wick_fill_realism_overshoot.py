"""Part 1 of the wick-fill realism check: how far past gap_open did the
fill day's High actually go? A day where High just barely brushes
gap_open (overshoot near 0%) is a much shakier "the resting order filled"
assumption than a day where price moved cleanly through the level.

Uses only data already on disk (the 502 daily-bar parquets + the anchor
508-trade log) -- no download.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
BARS_DIR = HERE.parent / "data" / "daily_bars"
ANCHOR = RESULTS / "confirm2yr_heldout_stops_exclude_1-2pct_friction.csv"


def main():
    trades = pd.read_csv(ANCHOR, parse_dates=["gap_date", "entry_date", "exit_date"])
    rows = []
    for _, t in trades.iterrows():
        p = BARS_DIR / f"{t['ticker']}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])

        gap_row = df[df["Date"] == t["gap_date"]]
        fill_row = df[df["Date"] == t["entry_date"]]
        if gap_row.empty or fill_row.empty:
            continue
        gap_open = float(gap_row["Open"].iloc[0])
        fill_high = float(fill_row["High"].iloc[0])
        overshoot_pct = (fill_high - gap_open) / gap_open * 100
        rows.append({
            "ticker": t["ticker"], "gap_date": t["gap_date"].date(),
            "entry_date": t["entry_date"].date(), "gap_open": round(gap_open, 4),
            "fill_day_high": round(fill_high, 4),
            "overshoot_pct": round(overshoot_pct, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "wick_fill_realism_overshoot.csv", index=False)

    os_pct = out["overshoot_pct"]
    print(f"n = {len(out)} (of {len(trades)} anchor trades; "
          f"{len(trades) - len(out)} missing parquet rows for gap/entry date)")
    print(f"min:    {os_pct.min():.4f}%")
    print(f"p10:    {os_pct.quantile(0.10):.4f}%")
    print(f"p25:    {os_pct.quantile(0.25):.4f}%")
    print(f"median: {os_pct.median():.4f}%")
    print(f"p75:    {os_pct.quantile(0.75):.4f}%")
    print(f"max:    {os_pct.max():.4f}%")
    under_01 = (os_pct < 0.1).mean() * 100
    under_02 = (os_pct < 0.2).mean() * 100
    under_05 = (os_pct < 0.5).mean() * 100
    print(f"\n% of fills with overshoot < 0.10% (hair's-breadth touch): {under_01:.1f}%")
    print(f"% of fills with overshoot < 0.20%: {under_02:.1f}%")
    print(f"% of fills with overshoot < 0.50%: {under_05:.1f}%")

    summary = pd.DataFrame([{
        "n": len(out), "min_%": round(os_pct.min(), 4), "p10_%": round(os_pct.quantile(0.10), 4),
        "p25_%": round(os_pct.quantile(0.25), 4), "median_%": round(os_pct.median(), 4),
        "p75_%": round(os_pct.quantile(0.75), 4), "max_%": round(os_pct.max(), 4),
        "pct_under_0.10%": round(under_01, 1), "pct_under_0.20%": round(under_02, 1),
        "pct_under_0.50%": round(under_05, 1),
    }])
    summary.to_csv(RESULTS / "wick_fill_realism_overshoot_summary.csv", index=False)


if __name__ == "__main__":
    main()
