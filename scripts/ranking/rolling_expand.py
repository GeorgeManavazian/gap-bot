"""Expand-the-book idea through the rolling-window test. Own loop (needs the
equity curve). Rule: BASE 10 slots, no stop, a position held >= TRIGGER
sessions gets its own extra slot (MAX_SLOTS = BASE + stuck), each position
sized equity/10, exit only by take profit or (optionally) a forced exit
after MAX_AGE sessions. Compared PAIRED against the live config and against
the best config (10 slots, 50% disaster stop, 63d exit).
-> results/rolling_expand.csv"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import harness, never_stop_expand_slots as nse
from harness import engine, make_filler, VARIANTS, CAPITAL
days = harness.nse.load(10)
idx = {d: i for i, (d, _) in enumerate(days)}
ORIG_STOP = engine.stop_for_abs_gap; ORIG_FILL = engine.fill_slots
HOR = 63

def run_expand(trigger, max_age, stop_pct=None):
    engine.stop_for_abs_gap = (lambda g: 100.0) if stop_pct is None else (lambda g, w=stop_pct: w)
    engine.fill_slots = nse.fill_slots_sized
    nse.SIZE["div"] = 10
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    eq, dates, stuck_n, last_open = [], [], [], None
    try:
        for today, bars in days:
            ti = idx[today]; stuck = 0
            for pos in state["open_positions"].values():
                age = ti - idx[pos["entry_date"]]
                if age >= trigger: stuck += 1
                pos["days_held"] = HOR - 1 if (max_age and age >= max_age) else min(pos["days_held"], HOR - 3)
            engine.MAX_SLOTS = 10 + stuck
            res = engine.step_one_day(state, today, bars, filler)
            eq.append(res["equity"]); dates.append(today); stuck_n.append(stuck)
    finally:
        engine.stop_for_abs_gap = ORIG_STOP; engine.fill_slots = ORIG_FILL
    op = state["open_positions"]
    last_open = (len(op), min([(p["last"] / p["entry"] - 1) * 100 for p in op.values()] or [0]))
    return pd.Series(eq, index=pd.to_datetime(dates)), max(stuck_n), last_open

def run_plain(slots, stop):
    harness.SLOTS = slots
    engine.stop_for_abs_gap = ORIG_STOP if stop == "orig" else (lambda g, w=stop: w)
    try: eq, tr = harness.simulate(days)
    finally: engine.stop_for_abs_gap = ORIG_STOP
    return eq

R = {"live (20 slots, bucket stops)": run_plain(20, "orig"),
     "BEST: 10 slots, 50% stop, 63d exit": run_plain(10, 50.0)}
info = {}
for name, trig, age in [("expand@30, never exit", 30, None), ("expand@30, forced exit at 6 months", 30, 126), ("expand@30, forced exit at 1 year", 30, 252)]:
    eq, mx, lo = run_expand(trig, age)
    R[name] = eq; info[name] = (mx, lo)
    print(f"ran {name}: full {round((eq.iloc[-1]/1e5-1)*100,1)}%  max extra slots {mx}  open at end {lo[0]} worst open {lo[1]:.0f}%", flush=True)

W = 504
def windows(eq, step=21):
    out = []
    for i in range(0, len(eq) - W, step):
        c = eq.iloc[i:i+W+1] / eq.iloc[i]
        r = (c.iloc[-1]-1)*100; d = ((c - c.cummax())/c.cummax()).min()*100
        out.append((r, d, r/-d if d < 0 else np.nan))
    return pd.DataFrame(out, columns=["ret","dd","ratio"])
base = windows(R["live (20 slots, bucket stops)"]); best = windows(R["BEST: 10 slots, 50% stop, 63d exit"])
rows = []
for n, eq in R.items():
    w = windows(eq)
    full_dd = ((eq - eq.cummax())/eq.cummax()).min()*100
    rows.append({"variant": n, "10yr_ret": round((eq.iloc[-1]/1e5-1)*100,1), "10yr_maxdd": round(full_dd,1),
                 "median_2yr_ret": round(w.ret.median(),1), "median_2yr_dd": round(w.dd.median(),1), "median_ratio": round(w.ratio.median(),2),
                 "worst_2yr_ret": round(w.ret.min(),1), "worst_2yr_dd": round(w.dd.min(),1),
                 "beats_live_ratio_%": round(100*(w.ratio.values > base.ratio.values).mean()),
                 "beats_BEST_ratio_%": round(100*(w.ratio.values > best.ratio.values).mean())})
df = pd.DataFrame(rows); df.to_csv("../../results/rolling_expand.csv", index=False)
pd.set_option("display.width", 260); print(df.to_string(index=False))
