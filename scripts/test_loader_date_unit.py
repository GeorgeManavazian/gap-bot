"""Regression test for the Date-unit bug (found 2026-10-05).

The cached parquets store Date as datetime64[ms]. TickerView keys its index
on df["Date"].values and looks up np.datetime64(day); in this env that is
datetime64[us] for a pandas Timestamp and datetime64[D] for a datetime.date,
so the hash never matches an [ms] key, every getter returns None and a whole
backtest quietly reports 0 trades. The shared loader must make the unit of
the stored Date irrelevant.

Run: .venv/bin/python -m pytest scripts/test_loader_date_unit.py
 or: .venv/bin/python scripts/test_loader_date_unit.py   (prints PASS/FAIL)
"""
import datetime
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from slot_and_priority_sweep import load_ticker, TickerView, ADV_LOOKBACK

N = 80
UNITS = ["ms", "ns", "s"]


def make_parquet(unit, tmpdir):
    dates = pd.bdate_range("2024-01-02", periods=N)
    df = pd.DataFrame({
        "Date": dates.values.astype(f"datetime64[{unit}]"),
        "Open": np.arange(N, dtype=float) + 100.0,
        "High": np.arange(N, dtype=float) + 101.0,
        "Low": np.arange(N, dtype=float) + 99.0,
        "Close": np.arange(N, dtype=float) + 100.5,
        "Volume": np.full(N, 1_000_000.0),
    })
    assert str(df["Date"].dtype) == f"datetime64[{unit}]"
    p = Path(tmpdir) / f"TEST_{unit}.parquet"
    df.to_parquet(p)
    # make sure the unit really survived the round trip, else the test is moot
    # (parquet has no seconds timestamp: pyarrow widens [s] to [ms] on write)
    stored = str(pd.read_parquet(p)["Date"].dtype)
    assert stored == f"datetime64[{'ms' if unit == 's' else unit}]", (unit, stored)
    return p, dates


def check_unit(unit, tmpdir):
    p, dates = make_parquet(unit, tmpdir)
    df = load_ticker(p)
    assert df is not None, f"[{unit}] load_ticker returned None"
    tv = TickerView(df)

    i = 40
    day = dates[i]                       # pandas Timestamp
    expect = (100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i)

    keys = {
        "pd.Timestamp": day,
        "np.datetime64[ns]": np.datetime64(day).astype("datetime64[ns]"),
        "np.datetime64[us]": np.datetime64(day).astype("datetime64[us]"),
        "np.datetime64[ms]": np.datetime64(day).astype("datetime64[ms]"),
        "np.datetime64[s]": np.datetime64(day).astype("datetime64[s]"),
        "np.datetime64[D]": np.datetime64(day).astype("datetime64[D]"),
        "datetime.date": day.date(),
        "datetime.datetime": day.to_pydatetime(),
    }
    for name, k in keys.items():
        got = tv.get(k)
        assert got is not None, f"[{unit}] get({name}) returned None"
        assert tuple(float(x) for x in got) == expect, f"[{unit}] get({name}) -> {got}, want {expect}"

    pc = tv.get_prior_close(day)
    assert pc is not None and float(pc) == 100.5 + (i - 1), f"[{unit}] get_prior_close -> {pc}"
    assert tv.get_prior_close(dates[0]) is None, f"[{unit}] prior close of first bar must be None"

    adv = tv.get_adv(day)
    want_adv = np.mean([(100.5 + j) * 1_000_000.0 for j in range(i - ADV_LOOKBACK, i)])
    assert adv is not None and abs(adv - want_adv) < 1e-6, f"[{unit}] get_adv -> {adv}, want {want_adv}"
    assert tv.get_adv(dates[ADV_LOOKBACK - 1]) is None, f"[{unit}] adv before lookback filled must be None"

    missing = pd.Timestamp("2023-12-25")
    assert tv.get(missing) is None, f"[{unit}] missing day get must be None"
    assert tv.get_prior_close(missing) is None, f"[{unit}] missing day prior close must be None"
    assert tv.get_adv(missing) is None, f"[{unit}] missing day adv must be None"
    assert tv.get(datetime.date(2023, 12, 25)) is None, f"[{unit}] missing date get must be None"


def test_ms():
    with tempfile.TemporaryDirectory() as d:
        check_unit("ms", d)


def test_ns():
    with tempfile.TemporaryDirectory() as d:
        check_unit("ns", d)


def test_s():
    with tempfile.TemporaryDirectory() as d:
        check_unit("s", d)


if __name__ == "__main__":
    failed = []
    with tempfile.TemporaryDirectory() as d:
        for u in UNITS:
            try:
                check_unit(u, d)
                print(f"  [{u}] ok")
            except AssertionError as e:
                failed.append(u)
                print(f"  [{u}] FAIL: {e}")
    print("PASS" if not failed else f"FAIL ({', '.join(failed)})")
    sys.exit(0 if not failed else 1)
