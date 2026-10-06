"""Test 1: is arm E's (no stop, keep 63d exit) gain over control D a 10-slot artifact?
Runs D and E at BASE = 10/15/20/30 (size_div = BASE) on 2yr and 10yr.
10yr = direction only (survivorship-biased universe).
-> results/test1_nostop_20slots_{2yr,10yr}.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.argv = [sys.argv[0]]  # never_stop_expand_slots reads argv[1] at import
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import never_stop_expand_slots as m
from live import engine

SLOTS = [10, 15, 20, 30]
CHECKS = {(2, 10): (76, 25.16, -11.09), (2, 20): (174, 11.45, -14.59)}


def main():
    pd.set_option("display.width", 250)
    try:
        for yrs in (2, 10):
            days = m.load(yrs)
            print(f"\n=== {yrs}yr: {len(days)} sessions {days[0][0]} -> {days[-1][0]}")
            rows = []
            for n in SLOTS:
                m.BASE = n
                res = {}
                for arm, stops in (("D", True), ("E", False)):
                    row, _, _ = m.run(days, arm, stops=stops, time_exit=True, expand=False, size_div=n)
                    res[arm] = row
                    rows.append({"slots": n, "arm": arm, "closed_trades": row["closed_trades"],
                                 "return_pct": row["return_pct"], "maxdd_pct": row["maxdd_pct"],
                                 "ret_over_maxdd": row["ret_over_maxdd"]})
                    if arm == "D" and (yrs, n) in CHECKS:
                        t, r, d = CHECKS[(yrs, n)]
                        ok = row["closed_trades"] == t and abs(row["return_pct"] - r) < .05 and abs(row["maxdd_pct"] - d) < .05
                        print(f"control check {yrs}yr/{n}: {'MATCH' if ok else 'MISMATCH'} {row['closed_trades']} {row['return_pct']} {row['maxdd_pct']}")
                        if not ok:
                            print("STOP: control mismatch"); return 1
                D, E = res["D"], res["E"]
                rows.append({"slots": n, "arm": "E-D",
                             "closed_trades": E["closed_trades"] - D["closed_trades"],
                             "return_pct": round(E["return_pct"] - D["return_pct"], 2),
                             "maxdd_pct": round(E["maxdd_pct"] - D["maxdd_pct"], 2),
                             "ret_over_maxdd": round(E["ret_over_maxdd"] - D["ret_over_maxdd"], 2)})
            df = pd.DataFrame(rows)
            df.to_csv(m.RESULTS / f"test1_nostop_20slots_{yrs}yr.csv", index=False)
            print(df.to_string(index=False))
    finally:
        engine.stop_for_abs_gap = m._ORIG_STOP
        engine.fill_slots = m._ORIG_FILL
    return 0


if __name__ == "__main__":
    sys.exit(main())
