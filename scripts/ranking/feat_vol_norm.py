"""Feature 1: volatility-normalized gap depth.

atr_pct = mean over the 20 sessions strictly BEFORE the gap-day row of
    true_range / that session's close * 100
with true_range = max(High-Low, |High-prevClose|, |Low-prevClose|), prevClose
being the previous session's close (so 21 prior rows are needed).
score = abs(gap_pct) / atr_pct: the gap measured in 'typical daily ranges'.

Why the window ends the day BEFORE the gap: the gap-day range itself is the
shock being measured. Including it would inflate the denominator exactly on
the violent days and shrink the score for the deepest gaps. The window also
keeps the information-set rule (nothing after gap_date's close is touched; in
fact nothing from the gap day at all).

Parameters (fixed a priori): window = 20 sessions.
Higher score = ranked first.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NAME = "vol_norm"
DESCRIPTION = "abs(gap_pct) / 20-session ATR% (window ends the session before the gap day)"
WINDOW = 20


def _atr_pct_before(d: pd.DataFrame) -> np.ndarray:
    """atr_pct_before[i] = mean of tr_pct over rows i-WINDOW..i-1 (NaN if short)."""
    h, l, c = d["High"], d["Low"], d["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1, skipna=False)
    tr_pct = tr / c * 100
    roll = tr_pct.rolling(WINDOW, min_periods=WINDOW).mean()
    return roll.shift(1).to_numpy()


def _scores(frames, events) -> list:
    cache: dict[str, np.ndarray] = {}
    out = []
    for tk, row, gp in zip(events["ticker"], events["row"], events["gap_pct"]):
        a = cache.get(tk)
        if a is None:
            a = cache[tk] = _atr_pct_before(frames[tk])
        atr = a[int(row)]
        if not np.isfinite(atr) or atr <= 0:
            out.append((tk, None))
        else:
            out.append((tk, abs(float(gp)) / float(atr)))
    return [v for _, v in out]


def compute(frames, events) -> dict:
    sc = _scores(frames, events)
    return {(t, g): s for t, g, s in zip(events["ticker"], events["gap_date"], sc)}


def compute_blend(frames, events) -> dict:
    """sqrt(abs(gap_pct) * score): geometric blend of raw and normalized depth."""
    sc = _scores(frames, events)
    return {(t, g): (None if s is None else float(np.sqrt(abs(float(gp)) * s)))
            for t, g, gp, s in zip(events["ticker"], events["gap_date"], events["gap_pct"], sc)}
