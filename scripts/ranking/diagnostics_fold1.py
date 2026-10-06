"""Phase 2 of PROTOCOL_2026-10-05: mechanism check on watches with gap_date
<= 2018-12-31 ONLY (fold-1 train set). Return per slot-day = sum(pnl_pct) /
sum(days_held) (ratio of sums, robust to 1-day trades). Quintiles are formed
WITHIN depth bucket. CI = 95% bootstrap over gap_dates (cluster bootstrap).
-> results/diagnostics_fold1.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).parent.parent.parent / "results"
t = pd.read_csv(R / "outcome_table.csv")
t = t[t.gap_date <= "2018-12-31"].copy()
t["grp"] = np.where(t.abs_gap >= 10, "10%+", np.where(t.abs_gap >= 5, "5-10%", "2-5%"))
print(f"watches: {len(t)}  dates: {t.gap_date.nunique()}  by group: {t.grp.value_counts().to_dict()}")
print(f"crowd days (frac>=0.5) in fold-1 events: {t[t.breadth_frac >= 0.5].gap_date.nunique()} days, {int((t.breadth_frac >= 0.5).sum())} watches")
rng = np.random.default_rng(0)
FEATS = ["abs_gap", "vol_norm", "vol_spike", "breadth_frac", "dd52", "mom63"]
rows = []
for g, sub in t.groupby("grp"):
    for f in FEATS:
        s = sub.dropna(subset=[f]).copy()
        if len(s) < 100:
            continue
        s["q"] = pd.qcut(s[f].rank(method="first"), 5, labels=False) + 1
        out = {"grp": g, "feature": f, "n": len(s)}
        for q in (1, 2, 3, 4, 5):
            x = s[s.q == q]
            out[f"Q{q}_rpsd"] = round(x.pnl_pct.sum() / x.days_held.clip(lower=1).sum(), 3)   # % per slot-day
        # cluster bootstrap of Q5-Q1 rpsd
        dates = s.gap_date.unique()
        di = {d: i for i, d in enumerate(dates)}
        didx = s.gap_date.map(di).values
        def agg(mask):
            p = np.bincount(didx[mask], weights=s.pnl_pct.values[mask], minlength=len(dates))
            d = np.bincount(didx[mask], weights=s.days_held.clip(lower=1).values[mask], minlength=len(dates))
            return p, d
        p5, d5 = agg((s.q == 5).values); p1, d1 = agg((s.q == 1).values)
        diffs = []
        for _ in range(1000):
            b = rng.integers(0, len(dates), len(dates))
            if d5[b].sum() == 0 or d1[b].sum() == 0:
                continue
            diffs.append(p5[b].sum() / d5[b].sum() - p1[b].sum() / d1[b].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["Q5_minus_Q1"] = round(out["Q5_rpsd"] - out["Q1_rpsd"], 3)
        out["ci_lo"], out["ci_hi"] = round(lo, 3), round(hi, 3)
        out["monotone_ups"] = int(sum(out[f"Q{i+1}_rpsd"] > out[f"Q{i}_rpsd"] for i in range(1, 5)))
        rows.append(out)
df = pd.DataFrame(rows)
df.to_csv(R / "diagnostics_fold1.csv", index=False)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
