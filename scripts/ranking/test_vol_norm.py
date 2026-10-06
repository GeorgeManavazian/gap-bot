import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import feat_vol_norm as f


def mk(n, gap_row=25, seed=0):
    """Flat-ish synthetic frame; close=100 always, daily H=102, L=98 (TR=4 -> 4%)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B").astype("datetime64[ns]")
    d = pd.DataFrame({"Date": dates, "Open": 100.0, "High": 102.0, "Low": 98.0,
                      "Close": 100.0, "Volume": 1e6})
    d.loc[gap_row, "Open"] = 95.0   # gap -5%
    d.loc[gap_row, "High"] = 200.0  # gap-day shock: must NOT enter the ATR
    d.loc[gap_row, "Low"] = 1.0
    return d


def ev(d, row, tk="AAA"):
    pc = d["Close"].iat[row - 1]
    gp = (d["Open"].iat[row] - pc) / pc * 100
    return pd.DataFrame([(tk, d["Date"].iat[row].strftime("%Y-%m-%d"), row,
                          d["Open"].iat[row], pc, gp)],
                        columns=["ticker", "gap_date", "row", "gap_open", "prior_close", "gap_pct"])


# (a) hand value: TR=4 each day, close 100 -> atr 4%, gap -5% -> 1.25; blend sqrt(5*1.25)=2.5
d = mk(40)
e = ev(d, 25)
key = ("AAA", e["gap_date"].iat[0])
assert abs(f.compute({"AAA": d}, e)[key] - 1.25) < 1e-12
assert abs(f.compute_blend({"AAA": d}, e)[key] - 2.5) < 1e-12

# hand value with a gap in prev close: row 10 close 110 -> row 11 TR = max(4, |102-110|, |98-110|)=12
d2 = mk(40)
d2.loc[10, "Close"] = 110.0
d2.loc[10, "High"] = 112.0   # row10 TR vs prev close 100: max(14, 12, 2)=14 -> 14/110*100
e2 = ev(d2, 25)
tr10 = 14.0 / 110 * 100
tr11 = 12.0 / 100 * 100   # close 100 on row 11
others = 4.0
# window rows 5..24: rows 10,11 special, 18 others
atr = (tr10 + tr11 + 18 * others) / 20
gp = e2["gap_pct"].iat[0]
assert abs(f.compute({"AAA": d2}, e2)[("AAA", e2["gap_date"].iat[0])] - abs(gp) / atr) < 1e-9

# (b) no look-ahead: scramble everything after gap row, and the gap row itself
rng = np.random.default_rng(1)
d3 = d.copy()
for col in ["Open", "High", "Low", "Close"]:
    d3.loc[26:, col] = rng.uniform(1, 500, len(d3) - 26)
d3.loc[25, ["High", "Low", "Close"]] = [999.0, 0.5, 3.0]   # gap-day range/close also excluded
assert f.compute({"AAA": d3}, e)[key] == f.compute({"AAA": d}, e)[key]

# (c) insufficient history: row 20 needs rows 0..19 + prev close for row 0 -> None; row 21 ok
d4 = mk(40, gap_row=20)
assert f.compute({"AAA": d4}, ev(d4, 20))[("AAA", d4["Date"].iat[20].strftime("%Y-%m-%d"))] is None
d5 = mk(40, gap_row=21)
assert f.compute({"AAA": d5}, ev(d5, 21))[("AAA", d5["Date"].iat[21].strftime("%Y-%m-%d"))] is not None
assert f.compute_blend({"AAA": d4}, ev(d4, 20))[("AAA", d4["Date"].iat[20].strftime("%Y-%m-%d"))] is None

# (d) direction: same gap, calmer stock scores higher; same vol, deeper gap scores higher
calm = mk(40); calm["High"] = 101.0; calm["Low"] = 99.0
calm.loc[25, ["High", "Low"]] = [200.0, 1.0]
wild = mk(40)
sc_calm = f.compute({"AAA": calm}, ev(calm, 25))[key]
sc_wild = f.compute({"AAA": wild}, ev(wild, 25))[key]
assert sc_calm > sc_wild
deep = mk(40); deep.loc[25, "Open"] = 90.0
assert f.compute({"AAA": deep}, ev(deep, 25))[key] > sc_wild

# every event gets an entry
assert set(f.compute({"AAA": d}, e)) == {key}
print("ALL TESTS PASSED")
