"""Best-lead config trade log (2026-10-06): 10 slots, one flat 50% disaster stop,
63d exit, equity/10 sizing, everything else = live honest rule.

Two runs through live.engine.step_one_day (via ranking/harness.simulate):
  10yr continuous  -> must reproduce +262.6% / -21.7% (2026-10-05 note)
  2yr cold start   -> same window as the +11.45% anchor (2024-09 -> 2026-09)
-> results/best_10slot_50stop_trades_{10yr,2yr}.csv
Run from this dir with the etf-bot venv (needs the datetime64[us] workaround in
never_stop_expand_slots, which harness imports)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "ranking"))
import harness
from harness import engine, nse

RESULTS = Path(__file__).parent.parent / "results"
ORIG = engine.stop_for_abs_gap
harness.SLOTS = 10
engine.stop_for_abs_gap = lambda g: 50.0
try:
    for years in (10, 2):
        days = nse.load(years)
        eq, tr = harness.simulate(days)
        m = harness.slice_metrics(eq, tr, days[0][0], days[-1][0])
        print(f"{years}yr {days[0][0]}->{days[-1][0]}: {m} closed={len(tr)}", flush=True)
        tr.to_csv(RESULTS / f"best_10slot_50stop_trades_{years}yr.csv", index=False)
        eq.to_csv(RESULTS / f"best_10slot_50stop_equity_{years}yr.csv", header=["equity"])
finally:
    engine.stop_for_abs_gap = ORIG
