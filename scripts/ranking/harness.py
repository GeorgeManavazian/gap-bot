"""Ranking-feature harness (2026-10-05): which watches get the resting orders.

Swaps live/engine.py::resting_orders (monkeypatch, engine file untouched) so
the free slots go to the top watches by a feature score instead of by gap
depth alone. Everything else is the live config: 20 slots, bucket stops
7/11/16/26/37, 63-day exit, 10 bps slippage, 3% ADV floor, 1-2% bucket
excluded. One full 10-year engine run per variant; metrics are sliced from
the single mark-to-market equity curve.

PRE-REGISTERED PROTOCOL (written before any result was seen):
  TRAIN = sessions through 2022-12-30.   TEST = 2023-01-03 onward.
  1. Baseline = plain depth ranking. It must reproduce the known full-10yr
     row (821 closed trades, +177.81%, maxDD -25.96%) or nothing else counts.
  2. Every variant below is scored on TRAIN ONLY. TEST numbers of the
     variants are never printed or saved, except the single winner (and the
     baseline for comparison).
  3. WINNER = the variant with the highest TRAIN return/maxDD among those
     that satisfy ALL of:
        (a) TRAIN return > baseline TRAIN return
        (b) TRAIN return/maxDD > baseline TRAIN return/maxDD
        (c) beats baseline's return in >= 4 of the 7 TRAIN calendar years
            (2016 is a partial year)
        (d) >= 80% of baseline's closed TRAIN trades (no degenerate selector)
     If nobody qualifies there is NO winner and the TEST slice is not run.
  4. The winner (and baseline) are scored on TEST exactly once.
Caveats that no protocol removes: 14 variants were tried on TRAIN so the
winner's TRAIN edge is optimistic by construction; the universe is today's
S&P 500 (survivorship); the 2024-09..2026-09 stretch inside TEST has already
been used for other checks.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

import common
import feat_vol_norm, feat_market_breadth, feat_volume_spike, feat_prior_trend, sector_cap
from live import engine
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler
import never_stop_expand_slots as nse      # gives load() with the date-unit workaround

RESULTS = Path(__file__).parent.parent.parent / "results"
TRAIN_END = "2022-12-30"
TEST_START = "2023-01-03"
SLOTS = 20
_ORIG_RESTING = engine.resting_orders


def score_ranker(scores: dict, sign: float):
    """resting_orders replacement: highest sign*score first; a missing score
    (None) ranks after every scored watch; ties and missing broken by depth
    then ticker (the engine's own tie-break)."""
    def f(live, pending, free):
        if free <= 0:
            return set()
        def key(t):
            s = scores[(t, pending[t]["gap_date"])]     # KeyError = harness bug, on purpose
            k = (1, 0.0) if s is None else (0, -sign * s)
            return (k, pending[t]["gap_pct"], t)
        return set(sorted(live, key=key)[:free])
    return f


def simulate(days, ranker=None, cap=None):
    """One full run. ranker: replacement resting_orders(live,pending,free).
    cap: max_per_sector (uses sector_cap with live open positions)."""
    engine.MAX_SLOTS = SLOTS
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    if cap is not None:
        capf = sector_cap.make_resting_orders(cap)
        ranker = lambda live, pending, free: capf(live, pending, free, set(state["open_positions"]))
    engine.resting_orders = ranker if ranker is not None else _ORIG_RESTING
    dates, equity, trades = [], [], []
    try:
        for today, bars in days:
            res = engine.step_one_day(state, today, bars, filler)
            trades.extend(res["trades"])
            dates.append(today)
            equity.append(res["equity"])
    finally:
        engine.resting_orders = _ORIG_RESTING
    return pd.Series(equity, index=pd.to_datetime(dates)), pd.DataFrame(trades)


def slice_metrics(eq: pd.Series, trades: pd.DataFrame, start, end, base_eq=None):
    """Return/maxDD of eq between start..end, rebased to the equity just before start."""
    s = eq[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    prev = eq[eq.index < pd.Timestamp(start)]
    base = prev.iloc[-1] if len(prev) else CAPITAL
    curve = pd.concat([pd.Series([base]), s.reset_index(drop=True)])
    dd = ((curve - curve.cummax()) / curve.cummax()).min() * 100
    ret = (s.iloc[-1] / base - 1) * 100
    t = trades[(trades.exit_date >= str(pd.Timestamp(start).date())) & (trades.exit_date <= str(pd.Timestamp(end).date()))]
    return {"return_pct": round(ret, 2), "maxdd_pct": round(dd, 2),
            "ret_over_maxdd": round(ret / -dd, 2) if dd < 0 else None,
            "closed_trades": len(t),
            "win_pct": round(100 * (t.pnl_pct > 0).mean(), 1) if len(t) else None}


def yearly_returns(eq: pd.Series, end) -> pd.Series:
    e = eq[eq.index <= pd.Timestamp(end)]
    last = e.groupby(e.index.year).last()
    prev = pd.concat([pd.Series([CAPITAL]), last.iloc[:-1]]).values
    return (last / prev - 1) * 100


def build_variants(frames, events):
    keys = list(zip(events.ticker, events.gap_date))
    V = {}
    def add(name, scores, sign=1.0):
        V[name] = dict(ranker=score_ranker(scores, sign))
    add("vol_norm", feat_vol_norm.compute(frames, events))
    add("vol_norm_blend", feat_vol_norm.compute_blend(frames, events))
    add("breadth_count", feat_market_breadth.compute(frames, events))
    add("breadth_frac", feat_market_breadth.compute_frac(frames, events))
    sp = feat_volume_spike.compute(frames, events)
    add("vol_spike_hi_first", sp, +1); add("vol_spike_lo_first", sp, -1)
    dd = feat_prior_trend.compute(frames, events)
    add("near_high_first", dd, +1); add("far_below_high_first", dd, -1)
    mo = feat_prior_trend.compute_mom63(frames, events)
    add("mom63_hi_first", mo, +1); add("mom63_lo_first", mo, -1)
    for c in (2, 3, 4):
        V[f"sector_cap_{c}"] = dict(cap=c)
    return V


def main():
    print(__doc__)
    frames = common.load_frames()
    events = common.gap_events(frames)
    days = nse.load(10)
    print(f"\n10yr: {len(days)} sessions {days[0][0]} -> {days[-1][0]}; {len(events)} gap events", flush=True)

    beq, btr = simulate(days)
    full = slice_metrics(beq, btr, days[0][0], days[-1][0])
    print("baseline full 10yr:", full, flush=True)
    ok = (len(btr) == 821 and abs(full["return_pct"] - 177.81) < 0.05 and abs(full["maxdd_pct"] + 25.96) < 0.05)
    # NB: closed trades in the baseline run, not slice_metrics' count of exits in window
    print("baseline reproduces 821 / +177.81% / -25.96%:", "MATCH" if ok else f"MISMATCH (trades={len(btr)})", flush=True)
    if not ok:
        return 1

    base_tr = slice_metrics(beq, btr, days[0][0], TRAIN_END)
    base_yr = yearly_returns(beq, TRAIN_END)
    rows = [{"variant": "depth_baseline", **base_tr}]
    store = {"depth_baseline": (beq, btr)}
    for name, kw in build_variants(frames, events).items():
        eq, tr = simulate(days, **kw)
        m = slice_metrics(eq, tr, days[0][0], TRAIN_END)
        yr = yearly_returns(eq, TRAIN_END)
        m["years_beating_base"] = int((yr.values > base_yr.values).sum())
        rows.append({"variant": name, **m})
        store[name] = (eq, tr)
        print(f"  {name:22s} TRAIN {m}", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "ranking_train.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nTRAIN (2016-09 .. 2022-12), live config, 20 slots:")
    print(df.to_string(index=False))

    b = df[df.variant == "depth_baseline"].iloc[0]
    q = df[(df.variant != "depth_baseline")
           & (df.return_pct > b.return_pct) & (df.ret_over_maxdd > b.ret_over_maxdd)
           & (df.years_beating_base >= 4) & (df.closed_trades >= 0.8 * b.closed_trades)]
    if q.empty:
        print("\nNO WINNER: no variant met all four pre-registered criteria on TRAIN. TEST slice not run.")
        return 0
    win = q.sort_values("ret_over_maxdd", ascending=False).iloc[0].variant
    print(f"\nWINNER by the pre-registered rule: {win}  (qualified: {list(q.variant)})")
    weq, wtr = store[win]
    t_base = slice_metrics(beq, btr, TEST_START, days[-1][0])
    t_win = slice_metrics(weq, wtr, TEST_START, days[-1][0])
    out = pd.DataFrame([{"variant": "depth_baseline", **t_base}, {"variant": win, **t_win}])
    out.to_csv(RESULTS / "ranking_test_winner.csv", index=False)
    print("\nTEST (2023-01 .. 2026-09), scored once:")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
