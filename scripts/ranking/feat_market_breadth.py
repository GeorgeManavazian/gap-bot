"""Feature 2: market breadth of the gap day (market-wide vs stock-specific gap).

n_gaps = number of watches (rows of events) registered on the same gap_date
across the whole universe, the stock itself included.
score = -n_gaps: a stock that gapped alone ranks above one that gapped on a
crowded panic day. Scores tie within a day by design (harness breaks ties by
depth). compute_frac() divides by the number of tickers with a bar that date,
to remove drift in universe size.

Information set: only same-day cross-section of gap_date (open vs prior close,
known by the close of gap_date). Nothing after gap_date is read.
"""
from __future__ import annotations
import pandas as pd

NAME = "mkt_breadth"
DESCRIPTION = "-(number of watches registered on the same gap_date across the universe); alone ranks above panic day"


def _counts(events):
    return events.groupby("gap_date").size()


def compute(frames, events):
    n = _counts(events)
    return {(t, d): -float(n[d]) for t, d in zip(events["ticker"], events["gap_date"])}


def _n_tickers(frames):
    s = pd.concat([d["Date"] for d in frames.values()], ignore_index=True)
    c = s.value_counts()
    c.index = c.index.strftime("%Y-%m-%d")
    return c


def compute_frac(frames, events):
    n = _counts(events)
    nt = _n_tickers(frames)
    out = {}
    for t, d in zip(events["ticker"], events["gap_date"]):
        k = nt.get(d)
        out[(t, d)] = -(float(n[d]) / float(k)) if k else None
    return out


def daily_table(frames, events):
    """DataFrame[date, n_gaps, n_tickers, frac, median_open_gap_pct]."""
    parts = []
    for d in frames.values():
        pc = d["Close"].shift(1)
        g = (d["Open"] - pc) / pc * 100
        parts.append(pd.DataFrame({"date": d["Date"], "g": g}))
    a = pd.concat(parts, ignore_index=True)
    a["date"] = a["date"].dt.strftime("%Y-%m-%d")
    grp = a.groupby("date")
    t = pd.DataFrame({"n_tickers": grp.size(), "median_open_gap_pct": grp["g"].median()})
    t["n_gaps"] = _counts(events)
    t["n_gaps"] = t["n_gaps"].fillna(0).astype(int)
    t["frac"] = t["n_gaps"] / t["n_tickers"]
    t = t.reset_index()
    return t[["date", "n_gaps", "n_tickers", "frac", "median_open_gap_pct"]]
