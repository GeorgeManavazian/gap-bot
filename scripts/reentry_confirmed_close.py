"""STRONGER confirmation: don't enter on a mere intraday wick back up to the
gap-day open (that barely differed from buying blind -- see
reentry_analysis.py). Require a full day's CLOSE back above the gap-day's
open -- proof buyers held it through end of day, not a fakeout wick that
rolled back over.

  gap day t:          Open[t] < Close[t-1]  (down gap >= 1%)
  confirm day k>t:    first day where Close[k] >= Open[t]
  entry:              next day's OPEN after confirmation (k+1) -- you can
                       only act on a close once you've seen it; entering
                       AT that close would be untradeable information.
  target:             Close[t-1], unchanged.
  horizon:             63 trading days from the GAP DAY t, same budget as
                       every other test -- confirmation eats into it, so a
                       trade that confirms late has less runway left to fill.
"""
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
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if len(df) < HORIZON + 5:
        return None
    return df


def analyze(df, ticker):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    dates = df["Date"].values
    n = len(df)
    out = []
    for t in range(1, n - 2):  # need room for confirm day AND an entry day after it
        prior_close = c[t - 1]
        if prior_close <= 0:
            continue
        gap_pct = (o[t] - prior_close) / prior_close * 100
        if gap_pct >= -1.0:
            continue
        gap_open = o[t]
        end = min(t + HORIZON, n - 1)

        confirm_day = None
        for k in range(t + 1, end + 1):
            if c[k] >= gap_open:
                confirm_day = k
                break

        if confirm_day is None or confirm_day + 1 > end:
            out.append({"ticker": ticker, "date": pd.Timestamp(dates[t]),
                        "abs_gap_pct": abs(gap_pct), "bucket": bucket_of(abs(gap_pct)),
                        "confirmed": False, "days_to_confirm": None,
                        "entry_price": None, "trade_pnl_pct": None,
                        "fill_day_from_entry": None})
            continue

        entry_day = confirm_day + 1
        entry_price = o[entry_day]
        fill_day = None
        for k in range(entry_day, end + 1):
            if h[k] >= prior_close:
                fill_day = k - entry_day
                break
        pnl = (prior_close - entry_price) / entry_price * 100 if fill_day is not None else None
        out.append({"ticker": ticker, "date": pd.Timestamp(dates[t]),
                    "abs_gap_pct": abs(gap_pct), "bucket": bucket_of(abs(gap_pct)),
                    "confirmed": True, "days_to_confirm": confirm_day - t,
                    "entry_price": entry_price, "trade_pnl_pct": pnl,
                    "fill_day_from_entry": fill_day})
    return out


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    all_rows = []
    for p in files:
        df = load_ticker(p)
        if df is None:
            continue
        all_rows.extend(analyze(df, p.stem))
    g = pd.DataFrame(all_rows)
    g.to_parquet((HERE.parent / "results" / "confirmed_close_gaps.parquet"), index=False)
    print(f"{len(g)} down-gap events (>=1%) across {g['ticker'].nunique()} tickers\n")

    print("=" * 110)
    print("STEP 1: does price ever CLOSE back above the gap-day open, within a quarter?")
    print("=" * 110)
    rows = []
    for name, lo, hi in BUCKETS:
        sub = g[g["bucket"] == name]
        n = len(sub)
        conf = sub[sub["confirmed"]]
        rows.append({
            "bucket": name, "n_gaps": n,
            "confirmed_%": round(100 * len(conf) / n, 1),
            "never_confirmed_%": round(100 * (1 - len(conf) / n), 1),
            "median_days_to_confirm": conf["days_to_confirm"].median(),
        })
    t1 = pd.DataFrame(rows)
    print(t1.to_string(index=False))
    t1.to_csv((HERE.parent / "results" / "confirmed_step1.csv"), index=False)

    print("\n" + "=" * 110)
    print("STEP 2 (end-to-end): confirms with a real close, THEN goes on to fill")
    print("(this is the disciplined-entry hit rate, and the entry price is now")
    print(" HIGHER than the gap open -- so profit per trade should shrink)")
    print("=" * 110)
    rows2 = []
    for name, lo, hi in BUCKETS:
        sub = g[g["bucket"] == name]
        n = len(sub)
        both = sub[sub["trade_pnl_pct"].notna()]
        rows2.append({
            "bucket": name, "n_gaps": n,
            "end_to_end_hit_rate_%": round(100 * len(both) / n, 1),
            "avg_profit_%_per_successful_trade": round(both["trade_pnl_pct"].mean(), 3) if len(both) else None,
            "median_days_confirm_to_fill": both["fill_day_from_entry"].median() if len(both) else None,
        })
    t2 = pd.DataFrame(rows2)
    print(t2.to_string(index=False))
    t2.to_csv((HERE.parent / "results" / "confirmed_step2_end_to_end.csv"), index=False)

    print("\n" + "=" * 110)
    print("COMPARISON TABLE -- naive open entry vs wick re-entry vs real close confirmation")
    print("=" * 110)
    naive = {"1-2%": (95.0, 1.393), "2-3%": (92.2, 2.467), "3-5%": (87.6, 3.885),
            "5-10%": (80.8, 7.173), "10%+": (67.6, 15.835)}
    wick = {"1-2%": (94.0, 1.393), "2-3%": (91.5, 2.467), "3-5%": (87.5, 3.885),
           "5-10%": (80.8, 7.175), "10%+": (67.6, 15.846)}
    print(f"{'bucket':8s}{'naive-open hit%':>18s}{'naive-open avg%':>18s}"
          f"{'wick-reentry hit%':>20s}{'wick avg%':>12s}"
          f"{'CLOSE-confirm hit%':>20s}{'CLOSE-confirm avg%':>20s}")
    for name, lo, hi in BUCKETS:
        row = t2[t2["bucket"] == name].iloc[0]
        n_h, n_a = naive[name]
        w_h, w_a = wick[name]
        print(f"{name:8s}{n_h:>18.1f}{n_a:>18.3f}{w_h:>20.1f}{w_a:>12.3f}"
              f"{row['end_to_end_hit_rate_%']:>20.1f}{row['avg_profit_%_per_successful_trade']:>20.3f}")


if __name__ == "__main__":
    main()
