"""Re-derives per-bucket stop widths under the CURRENT entry rule (resting-
limit + rest-top-N-by-gap, live since 2026-09-08), instead of the wick/
gap_desc rule the live 7/11/16/26/37 table was derived under.

Same discipline as confirm_2yr_heldout_stops.py: split the full bar history
into an 8yr TRAIN era and the 2yr HELD-OUT test era (same cutoff
`priority_lookahead_check.load(years=2)` uses internally -- last date minus
2 years), derive a stop table from MAE (max adverse excursion) measured
ONLY on trades that fully resolve inside the train era, then test candidate
tables ONCE against the untouched 2yr era.

Difference from confirm_2yr_heldout_stops.py: that script's MAE proxy
(entry at the gap day's own open, `find_down_gaps_with_mae`) predates the
current entry rule. Here the MAE is measured on trades the CURRENT rule
(rest_top_n + limit_semantics) actually produces in the train era, run
with a stop wide enough (95%) to never fire, so each trade's own price
path to its real exit (tp_gap_filled or time_exit) is intact -- same
"no-stop run" idea as stop_loss_sim.py's Scheme A, just on the current
engine.

MUST run under etf-bot/.venv-live (see gap-bot/README.md; the sibling
script documents a real dtype bug -- silent 0-trade result -- under the
other venv).

Does NOT touch live/config.py or anything under live/. Research only.
-> results/stop_resweep_current_rule.csv (candidate comparison)
-> results/mae_table_current_rule_train8yr.csv (the MAE table itself)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import priority_lookahead_check as plc
from live.config import BUCKETS as LIVE_BUCKETS, stop_for_abs_gap as LIVE_STOP_FOR_ABS_GAP

RESULTS = Path(__file__).parent.parent / "results"
BUCKET_NAMES = [b[0] for b in LIVE_BUCKETS]
LIVE_TABLE = {name: stop for name, _, _, stop in LIVE_BUCKETS}


def split_train_test():
    """Same cutoff load(years=2) uses internally (last date - 2yr), just
    computed once here so we can also keep everything before it."""
    all_days = plc.load(years=10)
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in all_days])
    cutoff = dates[-1] - pd.DateOffset(years=2)
    train_days = [(d, bars) for d, bars in all_days if pd.Timestamp(d) < cutoff]
    test_days = [(d, bars) for d, bars in all_days if pd.Timestamp(d) >= cutoff]
    return train_days, test_days, cutoff


def mae_table_from_train(train_days):
    """Run the current entry rule on the train era with a stop wide enough
    to never fire, then measure each trade's own worst adverse excursion
    (lowest low from entry_date to its actual exit_date) as % of entry."""
    plc.stop_for_abs_gap = lambda abs_gap: 95.0
    tdf, _ = plc.replay(train_days, "rest_top_n", limit_semantics=True, return_detail=True)
    plc.stop_for_abs_gap = LIVE_STOP_FOR_ABS_GAP

    date_index = {d: i for i, (d, _) in enumerate(train_days)}
    rows = []
    for _, tr in tdf.iterrows():
        i0, i1 = date_index[tr["entry_date"]], date_index[tr["exit_date"]]
        lows = [train_days[i][1].get(tr["ticker"], {}).get("l") for i in range(i0, i1 + 1)]
        lows = [l for l in lows if l is not None]
        worst_low = min(lows) if lows else tr["entry_price"]
        mae_pct = (worst_low - tr["entry_price"]) / tr["entry_price"] * 100
        rows.append({"ticker": tr["ticker"], "bucket": tr["bucket"], "entry_date": tr["entry_date"],
                     "exit_date": tr["exit_date"], "exit_reason": tr["exit_reason"], "mae_pct": mae_pct})
    return pd.DataFrame(rows)


def candidate_tables(mae_df):
    print(f"\n{'bucket':8s} {'n':>5s} {'p75_mae':>9s} {'p90_mae':>9s} {'p95_mae':>9s} {'live_stop':>10s}")
    p75, p90, p95 = {}, {}, {}
    for name in BUCKET_NAMES:
        sub = mae_df[mae_df["bucket"] == name]["mae_pct"]
        if len(sub) == 0:
            p75[name] = p90[name] = p95[name] = LIVE_TABLE[name]
            print(f"{name:8s} {0:5d} {'--':>9s} {'--':>9s} {'--':>9s} {LIVE_TABLE[name]:10.1f}")
            continue
        v75, v90, v95 = -sub.quantile(0.25), -sub.quantile(0.10), -sub.quantile(0.05)
        p75[name], p90[name], p95[name] = round(v75), round(v90), round(v95)
        print(f"{name:8s} {len(sub):5d} {v75:9.2f} {v90:9.2f} {v95:9.2f} {LIVE_TABLE[name]:10.1f}")
    return {"p75": p75, "p90": p90, "p95": p95}


def test_candidate(test_days, table, label):
    fn = lambda abs_gap: next((s for n, lo, hi, _ in LIVE_BUCKETS if lo <= abs_gap < hi and n in table
                                for s in [table[n]]), None)
    plc.stop_for_abs_gap = fn
    r = plc.replay(test_days, "rest_top_n", limit_semantics=True)
    plc.stop_for_abs_gap = LIVE_STOP_FOR_ABS_GAP
    ret_dd = r["return_pct"] / abs(r["maxdd_pct"]) if r["maxdd_pct"] else None
    return {"label": label, "n_trades": r["n_trades"], "win_pct": r["win_pct"],
            "return_pct": r["return_pct"], "maxdd_pct": r["maxdd_pct"], "ret_maxdd": round(ret_dd, 3) if ret_dd else None}


def main():
    train_days, test_days, cutoff = split_train_test()
    print(f"train era: {train_days[0][0]} -> {train_days[-1][0]}  ({len(train_days)} days)")
    print(f"test era (held out, 2yr): {test_days[0][0]} -> {test_days[-1][0]}  ({len(test_days)} days), cutoff {cutoff.date()}")

    baseline = test_candidate(test_days, LIVE_TABLE, "LIVE (7/11/16/26/37, derived under old entry rule)")
    ok = baseline["n_trades"] == 174 and abs(baseline["return_pct"] - 11.45) < 0.05 and abs(baseline["maxdd_pct"] - (-14.59)) < 0.05
    print(f"\nbaseline reproduction: {'MATCH' if ok else 'MISMATCH -- stop, numbers below not trustworthy'}  {baseline}")
    if not ok:
        return 1

    mae_df = mae_table_from_train(train_days)
    mae_df.to_csv(RESULTS / "mae_table_current_rule_train8yr.csv", index=False)
    tables = candidate_tables(mae_df)

    rows = [baseline]
    for key in ["p75", "p90", "p95"]:
        t = tables[key]
        rows.append(test_candidate(test_days, t, f"{key} ({'/'.join(str(int(t[n])) for n in BUCKET_NAMES)})"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stop_resweep_current_rule.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
