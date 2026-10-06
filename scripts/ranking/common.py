"""Shared data helpers for the ranking-feature builders (2026-10-05).

Why this exists: the cached parquet bars store Date as datetime64[ms], which
breaks the engine's TickerView lookups in this env, and the bars dict that
load() hands the engine carries no volume and no history. Feature builders
read raw frames from here instead, so every feature uses the identical gap
event definition.

INFORMATION SET RULE (no look-ahead): a feature for the watch (ticker,
gap_date) may use bars up to and including the CLOSE of gap_date, and nothing
later. The watch is ranked from the next session on, so that is all a live
bot would know.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "daily_bars"
CONSTITUENTS = Path(__file__).parent.parent.parent / "data" / "sp500_constituents.csv"
MIN_ABS_GAP = 2.0   # live.config.EXCLUDE_BELOW: gaps smaller than this are never watched


def load_frames() -> dict[str, pd.DataFrame]:
    """ticker -> DataFrame[Date(datetime64[ns]), Open, High, Low, Close, Volume],
    sorted by Date, duplicates dropped, rows with missing OHLC dropped."""
    out = {}
    for p in sorted(DATA_DIR.glob("*.parquet")):
        d = pd.read_parquet(p)
        d.columns = [c if isinstance(c, str) else c[0] for c in d.columns]
        need = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if not set(need).issubset(d.columns):
            continue
        d = d[need].dropna(subset=["Date", "Open", "High", "Low", "Close"])
        d["Date"] = pd.to_datetime(d["Date"]).astype("datetime64[ns]")
        d = d.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        if len(d) >= 68:   # engine: HORIZON + 5
            out[p.stem] = d
    return out


def gap_events(frames: dict[str, pd.DataFrame], min_abs_gap: float = MIN_ABS_GAP) -> pd.DataFrame:
    """Every down-gap the live engine would register as a watch: open more
    than min_abs_gap % below the ticker's own previous close.
    Columns: ticker, gap_date ('YYYY-MM-DD'), row (index into that ticker's
    frame, the gap-day row), gap_open, prior_close, gap_pct (negative)."""
    rows = []
    for tk, d in frames.items():
        pc = d["Close"].shift(1)
        gp = (d["Open"] - pc) / pc * 100
        m = (gp <= -min_abs_gap) & pc.notna() & (pc > 0)
        idx = np.flatnonzero(m.values)
        for i in idx:
            rows.append((tk, d["Date"].iat[i].strftime("%Y-%m-%d"), int(i),
                         float(d["Open"].iat[i]), float(pc.iat[i]), float(gp.iat[i])))
    ev = pd.DataFrame(rows, columns=["ticker", "gap_date", "row", "gap_open", "prior_close", "gap_pct"])
    return ev.sort_values(["gap_date", "ticker"]).reset_index(drop=True)


def sectors() -> dict[str, str]:
    """ticker -> GICS sector from data/sp500_constituents.csv (today's
    membership; tickers with a '.' in the Symbol are stored as '-' in bars)."""
    c = pd.read_csv(CONSTITUENTS)
    return {s.replace(".", "-"): sec for s, sec in zip(c["Symbol"], c["GICS Sector"])}
