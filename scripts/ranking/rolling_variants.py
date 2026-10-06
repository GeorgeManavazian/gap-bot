"""Fair comparison of stop/slot variants: one CONTINUOUS 10yr run per variant,
then every rolling 2-year window (a new start each ~month, 96 windows) scored
for return / maxDD, compared PAIRED against the live config. Also the 5
non-overlapping 2-year windows (the honest effective sample size).
-> results/rolling_variants.csv"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import harness
from harness import engine
days = harness.nse.load(10)
ORIG_STOP = engine.stop_for_abs_gap
VARIANTS = {
    "live (20 slots, bucket stops)":   dict(slots=20, stop="orig"),
    "15 slots, bucket stops":          dict(slots=15, stop="orig"),
    "10 slots, bucket stops":          dict(slots=10, stop="orig"),
    "20 slots, NO stop (63d exit)":    dict(slots=20, stop=None),
    "10 slots, NO stop (63d exit)":    dict(slots=10, stop=None),
    "20 slots, 50% disaster stop":     dict(slots=20, stop=50.0),
    "10 slots, 50% disaster stop":     dict(slots=10, stop=50.0),
}
W = 504
eqs = {}
for name, v in VARIANTS.items():
    harness.SLOTS = v["slots"]
    engine.stop_for_abs_gap = ORIG_STOP if v["stop"] == "orig" else (lambda g: 100.0) if v["stop"] is None else (lambda g, w=v["stop"]: w)
    try:
        eq, tr = harness.simulate(days)
    finally:
        engine.stop_for_abs_gap = ORIG_STOP
    eqs[name] = eq
    print(f"ran {name}: full {round((eq.iloc[-1]/1e5-1)*100,1)}%", flush=True)

def windows(eq, step):
    out = []
    for i in range(0, len(eq) - W, step):
        seg = eq.iloc[i:i+W+1]; c = seg / seg.iloc[0]
        ret = (c.iloc[-1] - 1) * 100; dd = ((c - c.cummax()) / c.cummax()).min() * 100
        out.append((ret, dd, ret / -dd if dd < 0 else np.nan))
    return pd.DataFrame(out, columns=["ret", "dd", "ratio"])

base = "live (20 slots, bucket stops)"
rows = []
for step, label in ((21, "96 overlapping"), (W, "non-overlapping")):
    b = windows(eqs[base], step)
    for name in VARIANTS:
        w = windows(eqs[name], step)
        rows.append({"windows": label, "n": len(w), "variant": name,
                     "median_ret": round(w.ret.median(), 1), "median_dd": round(w.dd.median(), 1),
                     "median_ratio": round(w.ratio.median(), 2), "worst_ret": round(w.ret.min(), 1), "worst_dd": round(w.dd.min(), 1),
                     "beats_live_ratio_pct": round(100 * (w.ratio.values > b.ratio.values).mean(), 0) if name != base else None,
                     "beats_live_ret_pct": round(100 * (w.ret.values > b.ret.values).mean(), 0) if name != base else None,
                     "median_ret_delta": round((w.ret.values - b.ret.values).mean(), 1) if name != base else None})
df = pd.DataFrame(rows)
df.to_csv("../../results/rolling_variants.csv", index=False)
pd.set_option("display.width", 250)
for label in ("96 overlapping", "non-overlapping"):
    print("\n", label); print(df[df.windows == label].drop(columns="windows").to_string(index=False))
