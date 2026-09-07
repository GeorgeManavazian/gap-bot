"""Answers the sizing questions directly from data already on disk (no new
download): how many down-gap signals actually fire per day (and how
correlated are they across tickers -- do they cluster on panic days), and
how far does price move AGAINST an eventually-winning trade before it hits
TP (max adverse excursion -- the number that should set the stop, not a
guess)."""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
HORIZON = 63

BUCKETS = [("1-2%", 1.0, 2.0), ("2-3%", 2.0, 3.0), ("3-5%", 3.0, 5.0),
           ("5-10%", 5.0, 10.0), ("10%+", 10.0, np.inf)]


def bucket_of(abs_gap):
    for name, lo, hi in BUCKETS:
        if lo <= abs_gap < hi:
            return name
    return None


def load_ticker(path):
    df = pd.read_parquet(path)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    need = {"Date", "Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < HORIZON + 5:
        return None
    return df


def find_down_gaps_with_mae(df, ticker):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    dates = df["Date"].values
    n = len(df)
    out = []
    for t in range(1, n - 1):
        prior_close = c[t - 1]
        if prior_close <= 0:
            continue
        gap_pct = (o[t] - prior_close) / prior_close * 100
        if gap_pct >= -1.0:   # down gaps only, >=1% down
            continue
        entry = o[t]
        end = min(t + HORIZON, n - 1)
        fill_day = None
        for k in range(t, end + 1):
            if h[k] >= prior_close:
                fill_day = k - t
                break
        # MAE: worst drawdown from entry, measured over [t, fill_day] if
        # resolved, else over the full horizon window (what you'd have sat
        # through holding to the quarter mark with no exit at all).
        mae_end = t + fill_day if fill_day is not None else end
        worst_low = l[t:mae_end + 1].min()
        mae_pct = (worst_low - entry) / entry * 100  # negative number
        out.append({
            "ticker": ticker, "date": pd.Timestamp(dates[t]),
            "abs_gap_pct": abs(gap_pct), "bucket": bucket_of(abs(gap_pct)),
            "fill_day": fill_day, "mae_pct": mae_pct,
        })
    return out


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    all_rows = []
    for p in files:
        df = load_ticker(p)
        if df is None:
            continue
        all_rows.extend(find_down_gaps_with_mae(df, p.stem))
    g = pd.DataFrame(all_rows)
    print(f"{len(g)} down-gap events (>=1%) across {g['ticker'].nunique()} tickers\n")

    # --- Q1: how many signals per day, by bucket -----------------------------
    print("=" * 100)
    print("Q1: HOW MANY DOWN-GAP SIGNALS FIRE ON A TYPICAL TRADING DAY (across all 503 names)")
    print("=" * 100)
    rows = []
    for name, lo, hi in BUCKETS:
        sub = g[g["bucket"] == name]
        per_day = sub.groupby("date").size()
        all_days = pd.date_range(sub["date"].min(), sub["date"].max(), freq="B")
        per_day = per_day.reindex(all_days, fill_value=0)
        rows.append({
            "bucket": name,
            "mean_per_day": round(per_day.mean(), 2),
            "median_per_day": per_day.median(),
            "p90_per_day": per_day.quantile(0.90),
            "max_per_day": per_day.max(),
            "days_with_zero_%": round(100 * (per_day == 0).mean(), 1),
        })
    freq_table = pd.DataFrame(rows)
    print(freq_table.to_string(index=False))
    freq_table.to_csv((HERE.parent / "results" / "signals_per_day.csv"), index=False)

    # --- Q2: clustering -- do signals correlate across tickers on the same day
    print("\n" + "=" * 100)
    print("Q2: CLUSTERING -- when signals DO fire, do they fire together (macro) or "
          "alone (idiosyncratic)?")
    print("=" * 100)
    print("(using the 3-5% bucket as the representative size)")
    sub = g[g["bucket"] == "3-5%"]
    per_day = sub.groupby("date").size()
    per_day = per_day[per_day > 0]
    print(f"  of {len(per_day)} days with >=1 signal:")
    print(f"    1 signal only:      {(per_day == 1).mean()*100:.1f}% of days")
    print(f"    2-5 signals:        {((per_day >= 2) & (per_day <= 5)).mean()*100:.1f}% of days")
    print(f"    6-20 signals:       {((per_day >= 6) & (per_day <= 20)).mean()*100:.1f}% of days")
    print(f"    20+ signals (panic-day cluster): {(per_day > 20).mean()*100:.1f}% of days")
    top10 = per_day.sort_values(ascending=False).head(10)
    print(f"\n  worst 10 clustering days (n signals that day):")
    for d, n in top10.items():
        print(f"    {d.date()}  {n} tickers gapped down 3-5%+ simultaneously")

    # --- Q3: max adverse excursion -- what should the stop actually be -------
    print("\n" + "=" * 100)
    print("Q3: MAX ADVERSE EXCURSION -- how far against you before it recovers "
          "(for trades that DO eventually fill within a quarter)")
    print("=" * 100)
    resolved = g[g["fill_day"].notna()]
    rows2 = []
    for name, lo, hi in BUCKETS:
        sub = resolved[resolved["bucket"] == name]
        if len(sub) == 0:
            continue
        rows2.append({
            "bucket": name,
            "n_resolved": len(sub),
            "median_mae_%": round(sub["mae_pct"].median(), 2),
            "p75_mae_%": round(sub["mae_pct"].quantile(0.25), 2),  # 25th pctile of a negative number = worse tail
            "p90_mae_%": round(sub["mae_pct"].quantile(0.10), 2),
            "worst_mae_%": round(sub["mae_pct"].min(), 2),
            "pct_that_never_dip_below_entry_%": round(100 * (sub["mae_pct"] >= -0.05).mean(), 1),
        })
    mae_table = pd.DataFrame(rows2)
    print(mae_table.to_string(index=False))
    mae_table.to_csv((HERE.parent / "results" / "mae_table.csv"), index=False)


if __name__ == "__main__":
    main()
