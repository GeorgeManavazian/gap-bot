import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import feat_volume_spike as f


def frame(vols):
    n = len(vols)
    d = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=n, freq="B"),
                      "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
                      "Volume": np.asarray(vols, dtype=float)})
    return d


def ev(tk, d, row):
    return pd.DataFrame({"ticker": [tk], "gap_date": [d["Date"].iat[row].strftime("%Y-%m-%d")],
                         "row": [row], "gap_open": [97.0], "prior_close": [100.0], "gap_pct": [-3.0]})


# (a) hand value: prior 20 rows volume 100 each, gap-day 350 -> 3.5
vols = [100] * 20 + [350] + [999] * 5
d = frame(vols)
e = ev("A", d, 20)
r = f.compute({"A": d}, e)
assert abs(r[("A", e.gap_date[0])] - 3.5) < 1e-12, r
# window is rows 0..19 strictly before: a spike in row 0 vs row 1 changes mean
vols2 = [200] + [100] * 19 + [350]
r2 = f.compute({"A": frame(vols2)}, ev("A", frame(vols2), 20))
assert abs(list(r2.values())[0] - 350 / 105) < 1e-12

# (b) no look-ahead: scramble everything after gap row
rng = np.random.default_rng(0)
vols3 = list(rng.integers(50, 500, 30)) + [123] * 10
d3 = frame(vols3)
e3 = ev("B", d3, 25)
base = f.compute({"B": d3}, e3)
d3b = d3.copy()
d3b.loc[26:, "Volume"] = rng.integers(1, 10**7, len(d3b) - 26).astype(float)
d3b.loc[26:, "Close"] = rng.random(len(d3b) - 26) * 1000
assert f.compute({"B": d3b}, e3) == base
# gap-day own volume DOES matter (allowed)
d3c = d3.copy(); d3c.loc[25, "Volume"] *= 2
assert f.compute({"B": d3c}, e3) != base

# (c) None cases: only 19 prior rows; zero mean; NaN volume
d4 = frame([100] * 19 + [300])
assert list(f.compute({"C": d4}, ev("C", d4, 19)).values()) == [None]
d5 = frame([0] * 20 + [300])
assert list(f.compute({"D": d5}, ev("D", d5, 20)).values()) == [None]
v6 = [100.0] * 21; v6[7] = np.nan
d6 = frame(v6)
assert list(f.compute({"E": d6}, ev("E", d6, 20)).values()) == [None]
d7 = frame([100.0] * 20 + [np.nan])
assert list(f.compute({"F": d7}, ev("F", d7, 20)).values()) == [None]

# (d) direction: higher volume spike -> higher score (ranked first)
lo, hi = frame([100] * 20 + [150]), frame([100] * 20 + [900])
frames = {"LO": lo, "HI": hi}
events = pd.concat([ev("LO", lo, 20), ev("HI", hi, 20)], ignore_index=True)
s = f.compute(frames, events)
ranked = sorted(s, key=lambda k: -s[k])
assert ranked[0][0] == "HI" and len(s) == len(events)

print("all vol_spike tests passed")
