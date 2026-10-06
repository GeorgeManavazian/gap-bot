"""Plain-assert tests for feat_prior_trend. Run from gap-bot dir:
   <venv python> scripts/ranking/test_prior_trend.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import feat_prior_trend as f


def make_frame(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    d = pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Open": close, "High": close + rng.uniform(0.1, 2, n),
        "Low": close - 1, "Close": close, "Volume": 1000.0,
    })
    return d


def ev_for(tk, d, row):
    pc = float(d["Close"].iat[row - 1])
    return pd.DataFrame([{
        "ticker": tk, "gap_date": d["Date"].iat[row].strftime("%Y-%m-%d"),
        "row": row, "gap_open": pc * 0.97, "prior_close": pc, "gap_pct": -3.0}])


# (a) hand-computed
n = 300
d = make_frame(n)
row = 280
ev = ev_for("AAA", d, row)
key = (ev.ticker[0], ev.gap_date[0])
hi = d["High"].iloc[row - 252:row].max()          # rows 28..279
pc = d["Close"].iat[row - 1]
exp = (pc / hi - 1) * 100
got = f.compute({"AAA": d}, ev)[key]
assert abs(got - exp) < 1e-9 and got <= 0, (got, exp)
exp_m = (pc / d["Close"].iat[row - 1 - 63] - 1) * 100
got_m = f.compute_mom63({"AAA": d}, ev)[key]
assert abs(got_m - exp_m) < 1e-9, (got_m, exp_m)

# tiny exact synthetic: flat 100 closes, High 100 except one 125 -> -20%
d2 = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=300),
                   "Open": 100.0, "High": 100.0, "Low": 100.0,
                   "Close": 100.0, "Volume": 1.0})
d2.loc[100, "High"] = 125.0
d2.loc[299, "Close"] = 90.0   # gap day close: must not matter
e2 = ev_for("BBB", d2, 290)
assert abs(f.compute({"BBB": d2}, e2)[("BBB", e2.gap_date[0])] - (-20.0)) < 1e-9
# at 52w high -> 0
d3 = d2.copy(); d3.loc[289, "High"] = 100.0; d3["High"] = 100.0
assert f.compute({"BBB": d3}, e2)[("BBB", e2.gap_date[0])] == 0.0

# (b) no look-ahead: scramble everything strictly after gap_date
for fn in (f.compute, f.compute_mom63):
    base = fn({"AAA": d}, ev)[key]
    dz = d.copy()
    rng = np.random.default_rng(99)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        dz.loc[row + 1:, c] = rng.uniform(1, 1000, n - row - 1)
    assert fn({"AAA": dz}, ev)[key] == base
    # gap-day row itself also must not matter (only prior sessions used)
    dg = d.copy()
    dg.loc[row, ["Open", "High", "Low", "Close"]] = 1e6
    assert fn({"AAA": dg}, ev)[key] == base

# (c) None for insufficient history
for r, fn, ok in [(251, f.compute, False), (252, f.compute, True),
                  (63, f.compute_mom63, False), (64, f.compute_mom63, True)]:
    e = ev_for("AAA", d, r)
    v = fn({"AAA": d}, e)[("AAA", e.gap_date[0])]
    assert (v is not None) == ok, (r, fn.__name__, v)

# every event has an entry
e_all = pd.concat([ev_for("AAA", d, 10), ev_for("AAA", d, 280)])
res = f.compute({"AAA": d}, e_all)
assert len(res) == 2 and sorted(v is None for v in res.values()) == [False, True]

# (d) ordering: shallower drawdown => higher score => ranked first
d_near = d2.copy()                      # drawdown -20
d_deep = d2.copy(); d_deep.loc[100, "High"] = 200.0   # drawdown -50
sc_near = f.compute({"BBB": d_near}, e2)[("BBB", e2.gap_date[0])]
sc_deep = f.compute({"BBB": d_deep}, e2)[("BBB", e2.gap_date[0])]
assert sc_near > sc_deep
order = sorted([("near", sc_near), ("deep", sc_deep)], key=lambda x: -x[1])
assert order[0][0] == "near"
# momentum: rising stock scores above falling stock
dr = d2.copy(); dr["Close"] = np.linspace(50, 100, 300)
df_ = d2.copy(); df_["Close"] = np.linspace(100, 50, 300)
assert (f.compute_mom63({"BBB": dr}, e2)[("BBB", e2.gap_date[0])]
        > f.compute_mom63({"BBB": df_}, e2)[("BBB", e2.gap_date[0])])

print("ALL TESTS PASSED")
