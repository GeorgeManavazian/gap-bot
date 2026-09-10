"""The gap trade as specified: we do NOT buy at the Open on the gap day
(that's buying blind, right when it gaps down). We wait
for price to climb back UP and RE-ENTER the gap zone -- cross back above the
gap day's own Open -- and only then do we buy, on a LATER day than the gap
itself. If price never comes back up to that level, no trade ever happens.

  gap day t:        Open[t] < Close[t-1]   (a down gap >= 1%)
  re-entry day k>t: the first day where High[k] >= Open[t]
  entry price:      Open[t] itself if day k's Open already opened at/above
                     it (gapped straight through a resting order), else
                     Open[t] (the resting-order fill level) if only the
                     High reached it intraday.
  target:           Close[t-1] (unchanged -- full gap fill)
  never re-enters:  counted and reported explicitly, not folded into "loss"
                     or dropped silently -- these are trades that structurally
                     never happen under this rule.

Same 63-trading-day search horizon as every other test in this study, timed
from the GAP DAY (not the re-entry day) so results stay comparable to the
earlier "does it fill within a quarter" framing.
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
    for t in range(1, n - 1):
        prior_close = c[t - 1]
        if prior_close <= 0:
            continue
        gap_pct = (o[t] - prior_close) / prior_close * 100
        if gap_pct >= -1.0:
            continue
        gap_open = o[t]  # the level we're waiting to re-enter
        end = min(t + HORIZON, n - 1)

        # find re-entry: first day STRICTLY AFTER t where High reaches gap_open
        reentry_day = None
        for k in range(t + 1, end + 1):
            if h[k] >= gap_open:
                reentry_day = k
                break

        if reentry_day is None:
            out.append({"ticker": ticker, "date": pd.Timestamp(dates[t]),
                        "abs_gap_pct": abs(gap_pct), "bucket": bucket_of(abs(gap_pct)),
                        "reentered": False, "days_to_reentry": None,
                        "fill_day_from_reentry": None, "trade_pnl_pct": None})
            continue

        entry_price = gap_open  # resting order fills exactly at the level
        # now: does it go on to fill the TP (prior_close) within what's left
        # of the horizon, measured from the re-entry day?
        fill_day = None
        for k in range(reentry_day, end + 1):
            if h[k] >= prior_close:
                fill_day = k - reentry_day
                break
        pnl = (prior_close - entry_price) / entry_price * 100 if fill_day is not None else None
        out.append({"ticker": ticker, "date": pd.Timestamp(dates[t]),
                    "abs_gap_pct": abs(gap_pct), "bucket": bucket_of(abs(gap_pct)),
                    "reentered": True, "days_to_reentry": reentry_day - t,
                    "fill_day_from_reentry": fill_day, "trade_pnl_pct": pnl})
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
    g.to_parquet((HERE.parent / "results" / "reentry_gaps.parquet"), index=False)
    print(f"{len(g)} down-gap events (>=1%) across {g['ticker'].nunique()} tickers\n")

    print("=" * 110)
    print("STEP 1: does price even come back to re-enter the gap at all, within a quarter?")
    print("=" * 110)
    rows = []
    for name, lo, hi in BUCKETS:
        sub = g[g["bucket"] == name]
        n = len(sub)
        reentered = sub[sub["reentered"]]
        rows.append({
            "bucket": name, "n_gaps": n,
            "reentered_%": round(100 * len(reentered) / n, 1),
            "never_reentered_%": round(100 * (1 - len(reentered) / n), 1),
            "median_days_to_reentry": reentered["days_to_reentry"].median(),
            "same_next_day_%": round(100 * (reentered["days_to_reentry"] == 1).mean(), 1) if len(reentered) else None,
        })
    t1 = pd.DataFrame(rows)
    print(t1.to_string(index=False))
    t1.to_csv((HERE.parent / "results" / "reentry_step1.csv"), index=False)

    print("\n" + "=" * 110)
    print("STEP 2: of the trades that DID re-enter, does the full gap then go on to fill?")
    print("(entry = re-entry price = the gap-day open; target = prior close, same as always)")
    print("=" * 110)
    rows2 = []
    for name, lo, hi in BUCKETS:
        sub = g[(g["bucket"] == name) & (g["reentered"])]
        n = len(sub)
        if n == 0:
            continue
        filled = sub[sub["trade_pnl_pct"].notna()]
        rows2.append({
            "bucket": name, "n_reentered_trades": n,
            "then_fills_%": round(100 * len(filled) / n, 1),
            "avg_profit_%_when_filled": round(filled["trade_pnl_pct"].mean(), 3) if len(filled) else None,
            "median_days_reentry_to_fill": filled["fill_day_from_reentry"].median() if len(filled) else None,
        })
    t2 = pd.DataFrame(rows2)
    print(t2.to_string(index=False))
    t2.to_csv((HERE.parent / "results" / "reentry_step2.csv"), index=False)

    print("\n" + "=" * 110)
    print("STEP 3: END-TO-END odds, from the moment the gap happens -- reenters AND fills")
    print("(this is the real, honest hit rate of the actual strategy you described)")
    print("=" * 110)
    rows3 = []
    for name, lo, hi in BUCKETS:
        sub = g[g["bucket"] == name]
        n = len(sub)
        both = sub[sub["trade_pnl_pct"].notna()]
        rows3.append({
            "bucket": name, "n_gaps": n,
            "end_to_end_hit_rate_%": round(100 * len(both) / n, 1),
            "avg_profit_%_per_successful_trade": round(both["trade_pnl_pct"].mean(), 3) if len(both) else None,
        })
    t3 = pd.DataFrame(rows3)
    print(t3.to_string(index=False))
    t3.to_csv((HERE.parent / "results" / "reentry_step3_end_to_end.csv"), index=False)


if __name__ == "__main__":
    main()
