"""Per-trade logs for the MAX_SLOTS sweep under the 2026-09-08 honest entry
rule (resting-limit fill + rest-top-N priority) -- summary-stats sibling is
scripts/slot_sweep_honest_rule.py (results/slot_sweep_honest_rule.csv, one
row per slot count). That's not enough to see WHICH trades a slot count
change wins or loses; this script re-runs the identical replay and dumps
every trade, tagged with its slot count, so vault-71 can slice by
ticker/bucket/gap size.

Same 2yr held-out window (2024-09 -> 2026-09), same engine.step_one_day()
call path as the summary sweep -- n=20 must reproduce the anchor
(174 trades / +11.45% / -14.59% maxDD) as the sanity check, same as there.

Schema matches results/confirm2yr_heldout_stops_exclude_1-2pct_friction.csv
(the anchor log) plus a leading `slots` column, so a diff across slot counts
is a plain groupby/filter, not a schema reconciliation.
Measurement only -- live/config.py is not changed here.
-> results/slot_sweep_trades.csv (combined, one row per trade per slot count)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from live import engine
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler
from priority_lookahead_check import load

RESULTS = Path(__file__).parent.parent / "results"
SLOTS = [5, 10, 15, 20, 30, 40, 50]
ANCHOR = {"slots": 20, "n_trades": 174, "return_pct": 11.45}
WINDOW_YEARS = 2  # the 2yr held-out window the ruling and the summary sweep both use

# Anchor-log column order (+ leading `slots`); engine.close_position() also
# emits cash_cost/proceeds, which ride along after this fixed prefix rather
# than being dropped -- more columns than the anchor, never fewer.
COLUMNS = ["slots", "ticker", "gap_date", "entry_date", "exit_date", "bucket", "gap_pct",
           "entry_price", "exit_price", "exit_reason", "units", "position_dollars",
           "pnl_pct", "pnl_dollar", "days_held"]


def run(days, n_slots: int) -> pd.DataFrame:
    engine.MAX_SLOTS = n_slots
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades = []
    equity = []
    for today, bars in days:
        res = engine.step_one_day(state, today, bars, filler)
        trades.extend(res["trades"])
        equity.append(res["equity"])
    for t in trades:
        t["slots"] = n_slots
    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity)
    ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    print(f"  slots={n_slots}: {len(tdf)} trades, return {ret:.2f}%")
    return tdf, len(tdf), round(ret, 2)


def main():
    days = load(WINDOW_YEARS)
    print(f"{WINDOW_YEARS}yr window: {len(days)} sessions, {days[0][0]} -> {days[-1][0]}")
    frames = []
    anchor_check = None
    for n in SLOTS:
        tdf, n_trades, ret = run(days, n)
        frames.append(tdf)
        if n == ANCHOR["slots"]:
            anchor_check = (n_trades == ANCHOR["n_trades"] and abs(ret - ANCHOR["return_pct"]) < 0.05)

    combined = pd.concat(frames, ignore_index=True)
    ordered = [c for c in COLUMNS if c in combined.columns]
    ordered += [c for c in combined.columns if c not in ordered]
    combined = combined[ordered]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "slot_sweep_trades.csv"
    combined.to_csv(out, index=False)
    print(f"\nwrote {len(combined)} trade rows across {len(SLOTS)} slot counts -> {out}")
    print("\nanchor reproduction at 20 slots:", "MATCH" if anchor_check else "MISMATCH -- do not trust the other rows")
    return 0 if anchor_check else 1


if __name__ == "__main__":
    sys.exit(main())
