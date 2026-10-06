"""Test 2: does arm E (no stops, 63d exit) beat control D (stops, 63d exit) in most years?

Own copy of never_stop_expand_slots.run() loop that also keeps the dated daily
MTM equity series. 10 slots, size_div=10 passed explicitly. Usage: `... [2|10]`.
-> results/test2_nostop_by_year[_10yr]*.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import never_stop_expand_slots as ns
from live import engine
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler

RESULTS = Path(__file__).parent.parent / "results"
YEARS = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def run_eq(days, stops):
    engine.stop_for_abs_gap = ns._ORIG_STOP if stops else (lambda g: 100.0)
    engine.fill_slots = ns.fill_slots_sized
    ns.SIZE["div"] = 10
    engine.MAX_SLOTS = ns.BASE
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades, eq, dates = [], [], []
    for today, bars in days:
        res = engine.step_one_day(state, today, bars, filler)
        trades.extend(res["trades"])
        eq.append(res["equity"])
        dates.append(pd.Timestamp(today))
    return pd.Series(eq, index=pd.DatetimeIndex(dates)), pd.DataFrame(trades)


def seg_stats(eq, prev_end):
    """return % and maxDD within segment; base = equity at end of previous segment."""
    base = CAPITAL if prev_end is None else prev_end
    s = pd.concat([pd.Series([base]), eq.reset_index(drop=True)])
    return (eq.iloc[-1] / base - 1) * 100, ((s - s.cummax()) / s.cummax()).min() * 100


def table(eqs, groups):
    rows = []
    prev = {k: None for k in eqs}
    for label, key in groups:
        r = {"period": label}
        if int(key(eqs["D"].index).sum()) < 20:   # drop empty / stub trailing block
            continue
        for arm, eq in eqs.items():
            seg = eq[key(eq.index)]
            ret, dd = seg_stats(seg, prev[arm])
            prev[arm] = seg.iloc[-1]
            r[f"{arm}_ret"], r[f"{arm}_dd"] = round(ret, 2), round(dd, 2)
        r["sessions"] = int(len(seg))
        r["d_ret"] = round(r["E_ret"] - r["D_ret"], 2)
        r["d_dd"] = round(r["E_dd"] - r["D_dd"], 2)   # >0 = E shallower DD
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    days = ns.load(YEARS)
    print(f"{YEARS}yr: {len(days)} sessions {days[0][0]} -> {days[-1][0]}")
    eqs, trs = {}, {}
    for arm, stops in (("D", True), ("E", False)):
        eqs[arm], trs[arm] = run_eq(days, stops)
    engine.stop_for_abs_gap, engine.fill_slots = ns._ORIG_STOP, ns._ORIG_FILL
    for arm in eqs:
        e = eqs[arm]
        print(arm, "total ret", round((e.iloc[-1] / CAPITAL - 1) * 100, 2),
              "maxDD", round(((e - e.cummax()) / e.cummax()).min() * 100, 2), "trades", len(trs[arm]))

    idx = eqs["D"].index
    first, last = idx[0], idx[-1]
    cal = []
    for y in sorted(set(idx.year)):
        part = ""
        if y == first.year and first > pd.Timestamp(f"{y}-01-10"): part = " (partial)"
        if y == last.year and last < pd.Timestamp(f"{y}-12-20"): part = " (partial)"
        cal.append((f"{y}{part}", (lambda ix, y=y: ix.year == y)))
    cal_df = table(eqs, cal)

    # 12-month blocks from window start
    start = idx[0]
    nb = int(((idx[-1].year - start.year) * 12 + idx[-1].month - start.month) // 12) + 1
    blocks = []
    for b in range(nb):
        lo, hi = start + pd.DateOffset(months=12 * b), start + pd.DateOffset(months=12 * (b + 1))
        blocks.append((f"blk{b+1} {lo.date()}", (lambda ix, lo=lo, hi=hi: (ix >= lo) & (ix < hi))))
    blk_df = table(eqs, blocks)

    # stop-exit realized losses in D by year
    td = trs["D"].copy()
    td["exit_date"] = pd.to_datetime(td["exit_date"])
    td["yr"] = td["exit_date"].dt.year
    st = td[td["exit_reason"] == "stop"]
    stop_df = st.groupby("yr").agg(stop_n=("pnl_dollar", "size"), stop_pnl_usd=("pnl_dollar", "sum")).reindex(sorted(set(idx.year)), fill_value=0).reset_index()
    stop_df["stop_pnl_usd"] = stop_df["stop_pnl_usd"].round(0)
    allp = td.groupby("yr")["pnl_dollar"].sum().reindex(stop_df["yr"], fill_value=0).round(0).values
    stop_df["all_exits_pnl_usd"] = allp

    sfx = "" if YEARS == 2 else f"_{YEARS}yr"
    cal_df.to_csv(RESULTS / f"test2_nostop_by_year{sfx}.csv", index=False)
    blk_df.to_csv(RESULTS / f"test2_nostop_by_year_blocks{sfx}.csv", index=False)
    stop_df.to_csv(RESULTS / f"test2_nostop_by_year_stoplosses{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    for nm, df in (("CALENDAR", cal_df), ("12M BLOCKS", blk_df)):
        print(f"\n{nm}\n{df.to_string(index=False)}")
        print(f"E>D return: {(df.d_ret > 0).sum()}/{len(df)} | E shallower DD: {(df.d_dd > 0).sum()}/{len(df)}"
              f" | best {df.loc[df.d_ret.idxmax(), 'period']} {df.d_ret.max()} | worst {df.loc[df.d_ret.idxmin(), 'period']} {df.d_ret.min()}")
        tot = df.d_ret.sum(); top2 = df.d_ret.nlargest(2).sum()
        print(f"sum of yearly deltas {tot:.2f}; top2 {top2:.2f} = {100*top2/tot:.0f}% of sum" if tot else "")
    print("\nSTOP EXITS in D by exit year\n" + stop_df.to_string(index=False))


if __name__ == "__main__":
    main()
