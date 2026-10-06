"""Feature 3: gap-day volume spike (ranking builder, 2026-10-05).

ratio = Volume[gap-day row] / mean(Volume of the 20 rows strictly before the gap-day row)

Parameters (fixed a priori): window = 20 sessions. Needs 20 prior rows, all
with non-NaN volume; mean must be finite and > 0, gap-day volume must be
non-NaN. Otherwise None. Score is the RAW ratio (no log, no rank, no sign flip).
HIGHER = ranked first. Direction of the edge is unknown: high volume may mean
capitulation or real news, so the harness should test both signs.

Information set: gap-day full-session volume is known after that day's close.
Nothing after the gap-day row is read.
"""
from __future__ import annotations
import numpy as np

NAME = "vol_spike"
DESCRIPTION = "gap-day volume / mean volume of the prior 20 sessions (raw ratio)"
WINDOW = 20


def compute(frames, events):
    out = {}
    for tk, g in events.groupby("ticker", sort=False):
        vol = frames[tk]["Volume"].astype(float)
        prior_mean = vol.shift(1).rolling(WINDOW).mean()   # rows i-20..i-1, all must be valid
        ratio = (vol / prior_mean).where(prior_mean > 0)
        arr = ratio.to_numpy()
        for gd, r in zip(g["gap_date"], g["row"]):
            v = arr[int(r)]
            out[(tk, gd)] = float(v) if np.isfinite(v) else None
    return out
