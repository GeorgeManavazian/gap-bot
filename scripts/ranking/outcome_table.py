"""Phase 1 of PROTOCOL_2026-10-05: per-watch outcome table.

Runs the REAL engine with MAX_SLOTS huge, so every touched watch fills (no
slot contention, no missed wicks); each resulting trade is one watch's
unconstrained outcome under the live exit rules. Features are joined on
(ticker, gap_date). Validation: every trade of the normal 20-slot depth
baseline must reappear here with the same entry/exit/pnl, except watches the
engine's own one-position-per-ticker rule blocks only in the unconstrained
run (a still-open earlier fill on the same ticker); that exception count is
reported.
-> results/outcome_table.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common
import harness
import feat_vol_norm, feat_market_breadth, feat_volume_spike, feat_prior_trend, sector_cap

RESULTS = harness.RESULTS


def main():
    frames = common.load_frames()
    events = common.gap_events(frames)
    days = harness.nse.load(10)

    print("baseline 20-slot depth run ...", flush=True)
    harness.SLOTS = 20
    beq, btr = harness.simulate(days)
    print(f"  {len(btr)} trades", flush=True)

    print("unconstrained run (MAX_SLOTS = 1e6) ...", flush=True)
    harness.SLOTS = 10**6
    ueq, utr = harness.simulate(days)
    harness.SLOTS = 20
    print(f"  {len(utr)} filled watches (of {len(events)} gap events)", flush=True)

    # --- validation against the 20-slot baseline
    k = ["ticker", "gap_date"]
    m = btr.merge(utr[k + ["entry_date", "exit_date", "pnl_pct", "exit_reason"]], on=k, how="left", suffixes=("", "_u"))
    missing = m["entry_date_u"].isna()
    same = (~missing) & (m.entry_date == m.entry_date_u) & (m.exit_date == m.exit_date_u) & ((m.pnl_pct - m.pnl_pct_u).abs() < 1e-6)
    print(f"validation: {int(same.sum())}/{len(btr)} baseline trades reproduce exactly in the unconstrained table; "
          f"{int(missing.sum())} absent (watch blocked by an earlier open fill on the same ticker); "
          f"{int((~missing & ~same).sum())} present but DIFFERENT", flush=True)
    if int((~missing & ~same).sum()) > 0:
        print(m[~missing & ~same][k + ["entry_date", "entry_date_u", "exit_date", "exit_date_u", "pnl_pct", "pnl_pct_u"]].head(10).to_string())

    # --- features
    ev = events.set_index(k)
    tbl = utr[["ticker", "gap_date", "entry_date", "exit_date", "bucket", "gap_pct", "exit_reason",
               "days_held", "pnl_pct", "pnl_dollar"]].copy()
    key = list(zip(tbl.ticker, tbl.gap_date))
    def col(d): return [d.get(x) for x in key]
    tbl["abs_gap"] = tbl.gap_pct.abs()
    tbl["vol_norm"] = col(feat_vol_norm.compute(frames, events))
    tbl["vol_spike"] = col(feat_volume_spike.compute(frames, events))
    tbl["breadth_frac"] = [(-v if v is not None else None) for v in col(feat_market_breadth.compute_frac(frames, events))]
    tbl["breadth_count"] = [(-v if v is not None else None) for v in col(feat_market_breadth.compute(frames, events))]
    tbl["dd52"] = col(feat_prior_trend.compute(frames, events))
    tbl["mom63"] = col(feat_prior_trend.compute_mom63(frames, events))
    smap = common.sectors()
    tbl["sector"] = [smap.get(t, "Unknown") for t in tbl.ticker]
    tbl["pnl_per_day"] = tbl.pnl_pct / tbl.days_held.clip(lower=1)
    tbl["in_baseline_20slot"] = [x in set(zip(btr.ticker, btr.gap_date)) for x in key] if False else False
    bset = set(zip(btr.ticker, btr.gap_date))
    tbl["in_baseline_20slot"] = [x in bset for x in key]
    tbl.to_csv(RESULTS / "outcome_table.csv", index=False)
    print(f"saved results/outcome_table.csv  rows={len(tbl)}  cols={list(tbl.columns)}")
    print("exit mix:", tbl.exit_reason.value_counts().to_dict())
    print("mean pnl_pct", round(tbl.pnl_pct.mean(), 2), " median days_held", tbl.days_held.median())
    return 0


if __name__ == "__main__":
    sys.exit(main())
