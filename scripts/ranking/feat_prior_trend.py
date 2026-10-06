"""Feature 4: prior trend before the gap.

Score = raw signed number, NOT flipped. Higher score = ranked first.
  compute()       -> 52-week drawdown of the prior close, in percent (<= 0).
                     0 = stock closed at its 52-week high the day before the
                     gap; -40 = prior close sat 40% under its 52-week high.
                     High score = shallow drawdown / near highs.
  compute_mom63() -> 63-session momentum into the prior close, in percent.
                     High score = strong positive momentum; very negative =
                     a stock that was already falling.

Fixed a priori: 252-row high window, 63-row momentum lookback. No tuning.
Information set: rows up to the session BEFORE the gap-day row (the gap-day
open is already in gap_pct, so it is not repeated here). "Sessions" are rows
of the ticker's frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "prior_trend"
DESCRIPTION = ("52-week drawdown of prior close vs max High of last 252 sessions "
               "(pct, <=0; 0 = at 52w high). Raw signed, not flipped.")

DD_WINDOW = 252
MOM_LOOKBACK = 63


def _per_ticker(frames, events, series_fn):
    out = {}
    for tk, ev in events.groupby("ticker", sort=False):
        d = frames[tk]
        vals = series_fn(d)  # array aligned to frame rows: value as of row r-1
        for gd, r in zip(ev["gap_date"], ev["row"]):
            v = vals[int(r)]
            out[(tk, gd)] = None if np.isnan(v) else float(v)
    return out


def _dd_series(d):
    # value at index r uses rows r-252 .. r-1 (window ends at prior session)
    high = d["High"].astype(float)
    close = d["Close"].astype(float)
    hmax = high.rolling(DD_WINDOW, min_periods=DD_WINDOW).max()
    dd = (close / hmax - 1.0) * 100.0
    return dd.shift(1).to_numpy()


def _mom_series(d):
    # value at index r: Close[r-1] / Close[r-1-63] - 1
    close = d["Close"].astype(float)
    mom = (close / close.shift(MOM_LOOKBACK) - 1.0) * 100.0
    return mom.shift(1).to_numpy()


def _fill(events, out):
    for tk, gd in zip(events["ticker"], events["gap_date"]):
        out.setdefault((tk, gd), None)
    return out


def compute(frames, events):
    return _fill(events, _per_ticker(frames, events, _dd_series))


def compute_mom63(frames, events):
    return _fill(events, _per_ticker(frames, events, _mom_series))
