import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import common
import feat_market_breadth as f

DATES = pd.bdate_range("2024-01-01", periods=5).astype("datetime64[ns]")


def mk(opens, closes):
    return pd.DataFrame({"Date": DATES, "Open": opens, "High": np.maximum(opens, closes) + 1,
                         "Low": np.minimum(opens, closes) - 1, "Close": closes, "Volume": 1000.0})


def frames(late=(100, 100, 100)):
    # day idx 2 (2024-01-03): A,B gap -5%, C flat. day idx 3: C gaps -10%.
    return {
        "A": mk([100, 100, 95, late[0], late[1]], [100, 100, 100, 100, 100]),
        "B": mk([50, 50, 47.5, late[0] / 2, late[1] / 2], [50, 50, 50, 50, 50]),
        "C": mk([10, 10, 10, 9 if late[2] == 100 else 10, 10], [10, 10, 10, 10, 10]),
    }


def ev(fr):
    return common.gap_events(fr, min_abs_gap=2.0)


# (a) hand-computed
fr = frames()
# gap_events requires >=68 rows via load only; gap_events itself does not
e = ev(fr)
d2 = DATES[2].strftime("%Y-%m-%d")
d3 = DATES[3].strftime("%Y-%m-%d")
# idx3: A open 100 vs close 100 -> no gap; C open 9 vs 10 -> -10%
s = f.compute(fr, e)
assert len(s) == len(e) == 3, e
assert s[("A", d2)] == -2.0 and s[("B", d2)] == -2.0 and s[("C", d3)] == -1.0, s
sf = f.compute_frac(fr, e)
assert abs(sf[("A", d2)] - (-2 / 3)) < 1e-12 and abs(sf[("C", d3)] - (-1 / 3)) < 1e-12, sf
t = f.daily_table(fr, e).set_index("date")
assert t.loc[d2, "n_gaps"] == 2 and t.loc[d2, "n_tickers"] == 3
assert abs(t.loc[d2, "frac"] - 2 / 3) < 1e-12
assert abs(t.loc[d2, "median_open_gap_pct"] - (-5.0)) < 1e-9   # gaps -5,-5,0
assert t.loc[d3, "n_gaps"] == 1

# (b) no look-ahead: replace all bars after gap_date d2 (rows 3,4) with junk; scores for d2 unchanged
fr2 = frames()
rng = np.random.default_rng(0)
for k, d in fr2.items():
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = d[c].astype(float)
        d.loc[3:, c] = rng.uniform(1, 200, size=len(d) - 3)
e2 = ev(fr2)
s2 = f.compute(fr2, e2)
for key in [("A", d2), ("B", d2)]:
    assert s2[key] == s[key], key
t2 = f.daily_table(fr2, e2).set_index("date")
assert t2.loc[d2, "n_gaps"] == t.loc[d2, "n_gaps"]
assert abs(t2.loc[d2, "median_open_gap_pct"] - t.loc[d2, "median_open_gap_pct"]) < 1e-12

# (c) None case: ticker with no bar on the date -> n_tickers missing -> frac None
e3 = e.copy()
e3.loc[0, "gap_date"] = "1999-01-01"
assert f.compute_frac(fr, e3)[(e3.loc[0, "ticker"], "1999-01-01")] is None

# (d) ordering: alone (-1) ranks above crowded (-2)
assert s[("C", d3)] > s[("A", d2)]
assert sf[("C", d3)] > sf[("A", d2)]

print("all tests passed")
