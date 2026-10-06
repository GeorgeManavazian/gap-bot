"""Test 3: does a wide flat 'disaster-only' stop cost anything vs no stop? 10 slots, size_div=10, 63d exit kept.
Arms: D (bucket stops), E (no stop), E+W for W in 30,40,50,60,75 (flat stop W% below entry, all buckets).
Run: python scripts/test3_disaster_stop.py [2|10]. 10yr = direction only (survivorship-biased)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import never_stop_expand_slots as nse
from live import engine
import pandas as pd

WS = [30, 40, 50, 60, 75]
ORIG = nse._ORIG_STOP


def arm(days, name, stops, W=None):
    nse._ORIG_STOP = (lambda g, W=W: float(W)) if W else ORIG
    try:
        row, tdf, _ = nse.run(days, name, stops=stops, time_exit=True, expand=False, size_div=10)
    finally:
        nse._ORIG_STOP = ORIG
    return row, tdf


def main():
    yrs = nse.WINDOW_YEARS
    days = nse.load(yrs)
    print(f"{yrs}yr: {len(days)} sessions")
    out, rows = [], []
    for name, stops, W in [("D", True, None), ("E", False, None)] + [(f"E+W{w}", True, w) for w in WS]:
        row, t = arm(days, name, stops, W)
        if name == "D":
            print("exit col candidates:", [c for c in t.columns if "reason" in c or "exit" in c])
        rc = next((c for c in ("exit_reason", "reason", "exit_type") if c in t.columns), None)
        stop_n = int(t[rc].astype(str).str.contains("stop", case=False).sum()) if rc else None
        print(name, rc, t[rc].value_counts().to_dict() if rc else None)
        rows.append(dict(arm=name, W=W, closed_trades=row["closed_trades"], n_stops=stop_n,
                         return_pct=row["return_pct"], maxdd_pct=row["maxdd_pct"],
                         ret_over_maxdd=row["ret_over_maxdd"], worst_trade_pct=round(t["pnl_pct"].min(), 2),
                         open_at_end=row["open_at_end"], worst_open_pct=row["worst_open_pct"]))
    df = pd.DataFrame(rows)
    d, e = df.iloc[0], df.iloc[1]
    df["ret_vs_D"] = (df.return_pct - d.return_pct).round(2)
    df["ret_vs_E"] = (df.return_pct - e.return_pct).round(2)
    df["dd_vs_E"] = (df.maxdd_pct - e.maxdd_pct).round(2)
    df.to_csv(nse.RESULTS / f"test3_disaster_stop{'' if yrs == 2 else f'_{yrs}yr'}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))
    if yrs == 2:
        print("D check:", d.closed_trades == 76 and abs(d.return_pct - 25.16) < .05 and abs(d.maxdd_pct + 11.09) < .05)
    print("E check:", e.return_pct, e.maxdd_pct, e.closed_trades)

main()
