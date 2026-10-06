"""Parity check for GAPBOT_PROFILE=ramp (2026-10-06): replay 10yr day-by-day through the UNPATCHED live/engine.step_one_day()
and compare with the research backtest of the same rule (scripts/test_ramp_floor.py, "ramp G12 floor 5%", 10 bps):
259 trades, +330.8%, maxDD -17.0%, Sharpe 1.16. The backtest dropped sub-5% watches at order time; the engine never registers
them (EXCLUDE_BELOW=5), which can differ by a re-registration edge case, so small drift is reported, a big one fails.
Also asserts the profile is really ramp, and slot_limit stays in [6, 9]."""
import os, sys
os.environ["GAPBOT_PROFILE"] = "ramp"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from live import config, engine
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler
import never_stop_expand_slots as nse
assert config.PROFILE == "ramp" and engine.PROFILE == "ramp", "profile not ramp"
print(config.PROFILE_BANNER, "floor", config.EXCLUDE_BELOW, "stop(12%)", config.stop_for_abs_gap(12.0), "size_div", config.SIZE_DIV)
days = nse.load(10)
st = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
f = make_filler("stock", VARIANTS["stock"])
eq, tr, lim, op = [], [], [], []
for today, bars in days:
    lim.append(engine.slot_limit(st["open_positions"], st["pending"]))
    r = engine.step_one_day(st, today, bars, f)
    tr.extend(r["trades"]); eq.append(r["equity"]); op.append(r["n_open"])
e = pd.Series(eq); t = pd.DataFrame(tr)
ret = (e.iloc[-1] / CAPITAL - 1) * 100; dd = ((e - e.cummax()) / e.cummax()).min() * 100
r_ = e.pct_change().dropna(); sh = r_.mean() / r_.std() * np.sqrt(252)
print(f"engine ramp 10yr: n={len(t)} return={ret:.1f}% maxDD={dd:.1f}% sharpe={sh:.2f} max_open={max(op)} slot_limit range {min(lim)}-{max(lim)}")
print("backtest ref    : n=259 return=330.8% maxDD=-17.0% sharpe=1.16")
print("min gap traded  :", round((-t.gap_pct).min(), 2), "| stops placed all 50%:", bool(((t.entry_price - t.entry_price) == 0).all()))
ok_range = min(lim) >= 6 and max(lim) <= 9 and max(op) <= 9 and (-t.gap_pct).min() >= 5.0
ok_close = abs(ret - 330.8) < 15 and abs(dd + 17.0) < 1.5 and abs(len(t) - 259) <= 12
print("RAMP RANGE/FLOOR OK" if ok_range else "RAMP RANGE/FLOOR VIOLATED", "|", "CLOSE TO BACKTEST" if ok_close else "DRIFT FROM BACKTEST -- investigate")
sys.exit(0 if (ok_range and ok_close) else 1)
