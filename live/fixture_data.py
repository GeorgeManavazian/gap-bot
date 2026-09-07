"""A local-parquet-backed stand-in for schwab_data.fetch_universe_bars, so
run_daily.py can be exercised end-to-end (--dry-run) without a live Schwab
connection. Reads the same cached data/daily_bars/*.parquet the backtest
uses, truncated to bars up to and including `as_of` -- lets you replay any
historical date as if it were "today" to sanity-check the live engine
against a day the backtest already scored."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from live.schwab_data import FETCH_LOOKBACK_DAYS, ADV_LOOKBACK

BARS_DIR = Path(__file__).parent.parent / "data" / "daily_bars"


def fetch_universe_bars(tickers: list[str], as_of) -> dict:
    end = pd.Timestamp(as_of).normalize()
    # generous calendar-day lookback so ADV_LOOKBACK trading days always fit
    start = end - pd.Timedelta(days=FETCH_LOOKBACK_DAYS + 30)
    out = {}
    for tk in tickers:
        path = BARS_DIR / f"{tk}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        need = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not need.issubset(df.columns):
            continue
        df = df[list(need)].dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        if len(df) >= ADV_LOOKBACK + 1:
            out[tk] = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return out
