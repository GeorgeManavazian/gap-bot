"""Test 4: how dependent is arm E's edge on SMCI / top contributors?

Drop tickers from every day's bars dict (freed slots go to other names), run
control D and arm E at 10 slots (size_div=10) on 2yr and 10yr.
Universes: full, minus SMCI, minus top-3 E contributors, minus top-5
(ranked by total pnl_dollar in E's full-universe trades on that window).
10yr = direction only (survivorship-biased). -> results/test4_drop_smci*.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import never_stop_expand_slots as ns
from live import engine

RESULTS = Path(__file__).parent.parent / "results"
D = dict(stops=True, time_exit=True, expand=False, size_div=10)
E = dict(stops=False, time_exit=True, expand=False, size_div=10)


def drop(days, tickers):
    if not tickers:
        return days
    return [(d, {k: v for k, v in bars.items() if k not in tickers}) for d, bars in days]


def main():
    out = []
    try:
        for yrs in (2, 10):
            days = ns.load(yrs)
            print(f"{yrs}yr: {len(days)} sessions {days[0][0]} -> {days[-1][0]}", flush=True)
            ns.BASE = 10
            rd, td, _ = ns.run(days, "D", **D)
            re_, te, _ = ns.run(days, "E", **E)
            if yrs == 2:
                ok = rd["closed_trades"] == 76 and abs(rd["return_pct"] - 25.16) < .05 and abs(rd["maxdd_pct"] + 11.09) < .05
                ok2 = re_["closed_trades"] == 74 and abs(re_["return_pct"] - 32.15) < .05
                print("control D 2yr:", rd["closed_trades"], rd["return_pct"], rd["maxdd_pct"], "E:", re_["closed_trades"], re_["return_pct"], re_["maxdd_pct"], "MATCH" if ok and ok2 else "MISMATCH")
                if not (ok and ok2):
                    return 1
            else:
                print("control 10yr D:", rd["return_pct"], rd["maxdd_pct"], "E:", re_["return_pct"], re_["maxdd_pct"])
            pnl = te.groupby("ticker")["pnl_dollar"].sum().sort_values(ascending=False)
            print("E top contributors:", pnl.head(6).round(0).to_dict(), flush=True)
            top = list(pnl.index)
            unis = [("full", []), ("-SMCI", ["SMCI"]), ("-top3", top[:3]), ("-top5", top[:5])]
            for uname, drops in unis:
                dd = drop(days, set(drops))
                res = {}
                for arm, kw in (("D", D), ("E", E)):
                    row, tdf, _ = ns.run(dd, arm, **kw)
                    res[arm] = row
                    out.append(dict(window=f"{yrs}yr", universe=uname, dropped=",".join(drops), arm=arm,
                                    closed_trades=row["closed_trades"], return_pct=row["return_pct"],
                                    maxdd_pct=row["maxdd_pct"], ret_over_maxdd=row["ret_over_maxdd"]))
                for r in out[-2:]:
                    r["E_minus_D_ret"] = round(res["E"]["return_pct"] - res["D"]["return_pct"], 2)
                    r["E_minus_D_retdd"] = round((res["E"]["ret_over_maxdd"] or 0) - (res["D"]["ret_over_maxdd"] or 0), 2)
    finally:
        engine.stop_for_abs_gap = ns._ORIG_STOP
        engine.fill_slots = ns._ORIG_FILL
    df = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "test4_drop_smci.csv", index=False)
    for w in ("2yr", "10yr"):
        df[df.window == w].to_csv(RESULTS / f"test4_drop_smci_{w}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
