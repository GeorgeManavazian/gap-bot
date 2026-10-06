"""Follow-ups raised by scripts/slot_sweep_trade_analysis.py (2026-09-08):

1. Is "10 slots" really "only the 10%+ bucket"? 58 of the 10-arm's 75
   trades were 10%+ gaps. Grid: EXCLUDE_BELOW (bucket scope floor) x
   MAX_SLOTS under the honest rule, on the 2yr window AND the 10yr
   direction-only window. If scope beats slot count, the mechanic to
   change is scope, not slots.
2. Per-bucket P&L under the honest rule, both windows, at 20 and 50
   slots -- which buckets carry edge once fills are real.
3. The stop-order variant's April-2025 behaviour: a limit at gap_open
   cannot fill when the rebound gaps UP through it; a buy-stop can.
   How much of the stop-order variant's extra return is that one month.

Every 2yr cell here is another look at the same window (playground, not
exam); the 10yr rows are the only independent check, and they carry the
survivorship caveat. -> results/slot_sweep_scope_grid.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from live import engine
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler
from priority_lookahead_check import load, replay as policy_replay

ROOT = Path(__file__).parent.parent
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DEFAULT_EXCLUDE = engine.EXCLUDE_BELOW


def run(days, n_slots, exclude_below):
    engine.MAX_SLOTS = n_slots
    engine.EXCLUDE_BELOW = exclude_below
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades, equity = [], []
    for today, bars in days:
        res = engine.step_one_day(state, today, bars, filler)
        trades.extend(res["trades"])
        equity.append(res["equity"])
    engine.EXCLUDE_BELOW = DEFAULT_EXCLUDE
    eq = pd.Series(equity, index=[d for d, _ in days])
    return pd.DataFrame(trades), eq


def summarize(tdf, eq):
    ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    d = eq.pct_change().dropna()
    return {"n_trades": len(tdf), "win_pct": round(100 * (tdf.pnl_pct > 0).mean(), 1) if len(tdf) else None,
            "return_pct": round(ret, 2), "maxdd_pct": round(dd, 2), "ret_over_dd": round(ret / -dd, 2) if dd < 0 else None,
            "worst_day_pct": round(d.min() * 100, 2), "mean_pnl_pct": round(tdf.pnl_pct.mean(), 2) if len(tdf) else None}


def halves(eq):
    df = pd.DataFrame({"eq": eq.values}, index=pd.to_datetime(eq.index))
    df["half"] = df.index.year.astype(str) + "H" + ((df.index.month > 6) + 1).astype(str)
    out, prev = {}, CAPITAL
    for h, g in df.groupby("half"):
        out[h] = round((g["eq"].iloc[-1] / prev - 1) * 100, 1)
        prev = g["eq"].iloc[-1]
    return out


def main():
    rows = []
    for years in (2, 10):
        days = load(years)
        print(f"\n===== {years}yr window ({days[0][0]} -> {days[-1][0]}){'  [direction only: survivorship-biased]' if years == 10 else ''} =====")
        # 2. per-bucket P&L at 20 and 50 slots, default scope
        for n in (20, 50):
            t, _ = run(days, n, 2.0)
            g = t.groupby("bucket").agg(n=("pnl_pct", "size"), mean_pnl=("pnl_pct", "mean"), median_pnl=("pnl_pct", "median"),
                                        win=("pnl_pct", lambda s: 100 * (s > 0).mean()), sum_pnl_usd=("pnl_dollar", "sum")).round(2)
            print(f"\n-- per-bucket, {n} slots, scope >2%:"); print(g.to_string())
        # 1. scope x slots grid
        print(f"\n-- scope floor x slots grid ({years}yr):")
        for ex in (2.0, 3.0, 5.0, 10.0):
            for n in (10, 15, 20, 30):
                t, eq = run(days, n, ex)
                r = {"window": f"{years}yr", "exclude_below_pct": ex, "slots": n, **summarize(t, eq)}
                if years == 2:
                    r.update(halves(eq))
                rows.append(r)
        df = pd.DataFrame([r for r in rows if r["window"] == f"{years}yr"])
        print(df.drop(columns="window").to_string(index=False))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "slot_sweep_scope_grid.csv", index=False)

    # 3. stop-order variant vs limit anchor, April 2025 and the rest
    print("\n===== 3. Stop-order variant (rest-top-N, cancelled if open >= tp) vs limit anchor: where the extra return lives (2yr) =====")
    engine.MAX_SLOTS = 20  # the grid above leaves it at its last cell; policy_replay sizes off engine.fill_slots
    days = load(2)
    lim_t, lim_eq = policy_replay(days, "rest_top_n", limit_semantics=True, return_detail=True)
    stp_t, stp_eq = policy_replay(days, "rest_top_n", open_if_above=True, skip_if_open_past_tp=True, return_detail=True)
    for name, t, eq in (("limit (anchor)", lim_t, lim_eq), ("stop-order", stp_t, stp_eq)):
        apr = t[(t.entry_date >= "2025-04-01") & (t.entry_date <= "2025-04-30")]
        w = eq[(eq.index >= "2025-04-01") & (eq.index <= "2025-04-30")]
        pre = eq[eq.index < "2025-04-01"].iloc[-1]
        print(f"  {name:14s} total {(eq.iloc[-1]/CAPITAL-1)*100:+.2f}%  n={len(t)}  | April-2025 entries: {len(apr)}, their eventual P&L ${apr.pnl_dollar.sum():,.0f} "
              f"(mean {apr.pnl_pct.mean() if len(apr) else 0:+.2f}%), window ret {(w.iloc[-1]/pre-1)*100:+.2f}%  | P&L$ from entries outside April: ${t[~t.index.isin(apr.index)].pnl_dollar.sum():,.0f}")
    print("  halves, limit:     ", halves(lim_eq))
    print("  halves, stop-order:", halves(stp_eq))
    stp_t["entry_month"] = stp_t.entry_date.str[:7]; lim_t["entry_month"] = lim_t.entry_date.str[:7]
    m = pd.DataFrame({"limit_n": lim_t.groupby("entry_month").size(), "limit_pnl$": lim_t.groupby("entry_month").pnl_dollar.sum().round(0),
                      "stop_n": stp_t.groupby("entry_month").size(), "stop_pnl$": stp_t.groupby("entry_month").pnl_dollar.sum().round(0)}).fillna(0)
    m["diff$"] = m["stop_pnl$"] - m["limit_pnl$"]
    print(m.to_string())


if __name__ == "__main__":
    main()
