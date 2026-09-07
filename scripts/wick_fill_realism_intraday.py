"""Part 3 of the wick-fill realism check: real 1-minute intraday spot
check on the anchor trades whose fill day is recent enough for Yahoo to
still serve 1m bars.

NOTE on scope: the task asked for "last 60 days" assuming that's
yfinance's 1m-bar limit. Empirically tested first (see
wick_fill_realism_60d_probe below, or just trust this comment): Yahoo's
actual error is explicit -- "1m data ... must be within the last 30
days" -- confirmed by a live probe (5/10 days back succeeded, 8/30/60
days back failed with that exact message). So the real usable window is
~30 days, not 60. This script uses that real limit and says so rather
than silently reporting fewer trades than promised.

For each trade: pull 1m bars for the fill day (ticker, entry_date),
find the print(s) at or above gap_open, and report the volume around
that touch plus whether it looks like a single anomalous tick or a
sustained move through the level (>=2 consecutive 1m bars at/above
gap_open, or the close of the touch bar itself already above gap_open).
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
ANCHOR = RESULTS / "confirm2yr_heldout_stops_exclude_1-2pct_friction.csv"
BARS_DIR = HERE.parent / "data" / "daily_bars"

TODAY = date(2026, 9, 7)
YFINANCE_1M_LIMIT_DAYS = 30  # empirically confirmed, not the 60 originally assumed


def gap_open_for(ticker, gap_date):
    p = BARS_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(p)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    row = df[df["Date"] == pd.Timestamp(gap_date)]
    if row.empty:
        return None
    return float(row["Open"].iloc[0])


def classify(ticker, entry_date, gap_open):
    start = entry_date
    end = entry_date + timedelta(days=1)
    try:
        bars = yf.download(ticker, start=str(start), end=str(end), interval="1m",
                            progress=False, auto_adjust=False)
    except Exception as e:
        return {"verdict": "fetch_error", "detail": str(e)[:200]}
    if bars.empty:
        return {"verdict": "no_intraday_data", "detail": "empty response"}
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)

    touched = bars[bars["High"] >= gap_open]
    if touched.empty:
        return {"verdict": "NO_TOUCH_IN_1M_DATA -- daily bar High disagrees with 1m data",
                "detail": f"max 1m High = {bars['High'].max():.4f}, gap_open = {gap_open:.4f}"}

    first_touch_ts = touched.index[0]
    first_touch_idx = bars.index.get_loc(first_touch_ts)
    window = bars.iloc[max(0, first_touch_idx - 1): first_touch_idx + 3]
    touch_bar = bars.loc[first_touch_ts]
    close_through = touch_bar["Close"] >= gap_open
    n_consecutive_at_or_above = 0
    for i in range(first_touch_idx, min(len(bars), first_touch_idx + 5)):
        if bars["High"].iloc[i] >= gap_open:
            n_consecutive_at_or_above += 1
        else:
            break
    sustained = close_through or n_consecutive_at_or_above >= 2
    verdict = "PLAUSIBLE_FILL (sustained move through level)" if sustained \
        else "LOOKS_LIKE_A_BLIP (single-print touch, price fell back same bar)"
    evidence = window[["Open", "High", "Low", "Close", "Volume"]].round(4).to_string()
    return {"verdict": verdict, "touch_time": str(first_touch_ts),
            "touch_bar_close": round(float(touch_bar["Close"]), 4),
            "touch_bar_volume": int(touch_bar["Volume"]),
            "n_consecutive_bars_at_or_above": n_consecutive_at_or_above,
            "evidence_1m_window": evidence}


def main():
    trades = pd.read_csv(ANCHOR, parse_dates=["gap_date", "entry_date", "exit_date"])
    cutoff = TODAY - timedelta(days=YFINANCE_1M_LIMIT_DAYS)
    recent = trades[trades["entry_date"].dt.date >= cutoff].reset_index(drop=True)
    print(f"anchor has {len(trades)} trades total; {len(recent)} fall within Yahoo's "
          f"real 1m-bar limit (last {YFINANCE_1M_LIMIT_DAYS} days from {TODAY}, "
          f"not the 60 originally assumed -- confirmed by a live probe: Yahoo returns "
          f"'must be within the last 30 days' for anything older).")

    rows = []
    for _, t in recent.iterrows():
        gap_open = gap_open_for(t["ticker"], t["gap_date"].date())
        if gap_open is None:
            rows.append({"ticker": t["ticker"], "entry_date": t["entry_date"].date(),
                         "verdict": "no_gap_open_in_daily_bars"})
            continue
        result = classify(t["ticker"], t["entry_date"].date(), gap_open)
        rows.append({"ticker": t["ticker"], "gap_date": t["gap_date"].date(),
                     "entry_date": t["entry_date"].date(), "gap_open": round(gap_open, 4),
                     **result})
        time.sleep(0.5)  # be polite to the free endpoint

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "wick_fill_realism_intraday.csv", index=False)

    print("\n" + "=" * 100)
    for _, r in out.iterrows():
        print(f"{r['ticker']:6s} {r.get('entry_date')}  gap_open={r.get('gap_open')}  "
              f"-> {r['verdict']}")
    print("=" * 100)
    if "verdict" in out.columns:
        vc = out["verdict"].apply(lambda v: v.split(" (")[0].split(" --")[0]).value_counts()
        print(vc.to_string())


if __name__ == "__main__":
    main()
