"""Feature 5: sector concentration cap (a selection CONSTRAINT, not a score).

Mirrors live/engine.py::resting_orders exactly in ordering -- sorted by
(pending[t]["gap_pct"], t), most negative gap first -- but walks the ranked
list and SKIPS a watch whose sector already holds `max_per_sector` tickers.
The count includes currently held positions (open_tickers) and watches
already chosen earlier in the same call. It keeps walking until `free`
orders are chosen or the list ends, so a skipped watch is replaced by the
next-ranked watch from another sector.

Unmapped tickers (not in data/sp500_constituents.csv): sector 'Unknown',
which is NEVER capped (they neither get skipped nor block anyone; they are
still counted in sector_counts() for description).

HOW THE HARNESS CALLS IT
engine.resting_orders(live, pending, free) has no open-positions argument,
so the harness must supply the held tickers itself. Sketch (NOT implemented
here, never run against the engine by this module):

    f = make_resting_orders(max_per_sector=3)
    orig = engine.resting_orders
    def step_with_cap(state, day, ...):
        held = set(state["open_positions"])          # tickers held at day start
        engine.resting_orders = lambda live, pending, free: f(live, pending, free, held)
        try:
            return orig_step_one_day(state, day, ...)
        finally:
            engine.resting_orders = orig

i.e. wrap engine.step_one_day per day: read the open-position tickers from
the state before the call, monkeypatch engine.resting_orders with a closure
that binds those tickers as open_tickers, call step_one_day, restore. If
resting_orders is called more than once inside a day, `held` must be the
positions open at the time of that call (re-read from state in the closure
if the engine mutates it mid-day).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common

NAME = "sector_cap"
DESCRIPTION = "Selection constraint: depth-ranked resting orders, skipping watches whose GICS sector already holds max_per_sector tickers (open + chosen)."

UNKNOWN = "Unknown"
_SECTORS: dict[str, str] | None = None


def _sector_map() -> dict[str, str]:
    global _SECTORS
    if _SECTORS is None:
        _SECTORS = common.sectors()
    return _SECTORS


def sector_of(ticker: str, smap: dict[str, str] | None = None) -> str:
    smap = _sector_map() if smap is None else smap
    return smap.get(ticker, UNKNOWN)


def sector_counts(tickers, smap: dict[str, str] | None = None) -> dict:
    """sector -> number of distinct tickers (unmapped -> 'Unknown')."""
    out: dict[str, int] = {}
    for t in set(tickers):
        s = sector_of(t, smap)
        out[s] = out.get(s, 0) + 1
    return out


def make_resting_orders(max_per_sector: int, smap: dict[str, str] | None = None):
    """Return f(live, pending, free, open_tickers) -> set of tickers."""
    def f(live, pending, free, open_tickers=()):
        if free <= 0:
            return set()
        counts = sector_counts(open_tickers, smap)
        chosen: set = set()
        for t in sorted(live, key=lambda t: (pending[t]["gap_pct"], t)):
            if len(chosen) >= free:
                break
            s = sector_of(t, smap)
            if s != UNKNOWN and counts.get(s, 0) >= max_per_sector:
                continue
            chosen.add(t)
            counts[s] = counts.get(s, 0) + 1
        return chosen
    return f
