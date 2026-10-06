"""Candidates A-E of PROTOCOL_2026-10-05 (amendment A1). Engine untouched:
resting_orders is monkeypatched per run. See the protocol for the exact
qualification/selection rules; they are implemented literally below.

Printing discipline: lockbox (2023-01-03 onward) numbers are computed only
for the single final candidate and the baseline, after selection.
-> results/candidates_train.csv, results/candidates_years.csv,
   results/candidates_lockbox.csv (only if someone qualifies)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common
import harness
import feat_vol_norm, feat_market_breadth, feat_volume_spike
import sector_cap
from harness import engine, make_filler, VARIANTS, CAPITAL, slice_metrics, RESULTS

TRAIN_START, TRAIN_END = "2016-09-06", "2022-12-30"
OOS_START, OOS_END = "2019-01-01", "2022-12-31"
LOCK_START, LOCK_END = "2023-01-03", "2026-09-04"
CROWD_FRAC, CROWD_CAP, SECTOR_CAP, RIDGE_ALPHA = 0.50, 4, 4, 10.0


# ----------------------------------------------------------------- features
def load_feature_scores(frames, events):
    S = {
        "vol_norm": feat_vol_norm.compute(frames, events),
        "vol_spike": feat_volume_spike.compute(frames, events),
        "quiet": feat_market_breadth.compute_frac(frames, events),   # = -frac, higher = quieter day
    }
    dt = feat_market_breadth.daily_table(frames, events)
    S["frac_by_date"] = dict(zip(dt["date"], dt["frac"]))
    return S


def pct_ranks(vals):
    """Percentile (0..1, ties averaged) among non-None values; None -> 0.5."""
    out = [0.5] * len(vals)
    idx = [i for i, v in enumerate(vals) if v is not None]
    m = len(idx)
    if m <= 1:
        return out
    r = pd.Series([vals[i] for i in idx], dtype="float64").rank(method="average").values
    for j, i in enumerate(idx):
        out[i] = (r[j] - 1.0) / (m - 1.0)
    return out


# --------------------------------------------------------------- order fns
def order_depth(live, pending, S):
    return sorted(live, key=lambda t: (pending[t]["gap_pct"], t))


def _col(S, name, live, pending):
    d = S[name]
    return [d[(t, pending[t]["gap_date"])] for t in live]


def order_B(live, pending, S):
    live = list(live)
    q = [(a + b + c) / 3.0 for a, b, c in zip(
        pct_ranks(_col(S, "vol_norm", live, pending)),
        pct_ranks(_col(S, "vol_spike", live, pending)),
        pct_ranks(_col(S, "quiet", live, pending)))]
    qd = dict(zip(live, q))
    def key(t):
        ab = -pending[t]["gap_pct"]
        return (0 if ab >= 10 else 1 if ab >= 5 else 2, -qd[t], pending[t]["gap_pct"], t)
    return sorted(live, key=key)


def order_C(live, pending, S):
    live = list(live)
    depth = [-pending[t]["gap_pct"] for t in live]
    comp = [(a + b + c + d) / 4.0 for a, b, c, d in zip(
        pct_ranks(depth),
        pct_ranks(_col(S, "vol_norm", live, pending)),
        pct_ranks(_col(S, "vol_spike", live, pending)),
        pct_ranks(_col(S, "quiet", live, pending)))]
    cd = dict(zip(live, comp))
    return sorted(live, key=lambda t: (-cd[t], pending[t]["gap_pct"], t))


def make_order_D(Dscores):
    def order(live, pending, S):
        def key(t):
            s = Dscores.get((t, pending[t]["gap_date"]))
            return (0, -s, pending[t]["gap_pct"], t) if s is not None else (1, 0.0, pending[t]["gap_pct"], t)
        return sorted(live, key=key)
    return order


# -------------------------------------------------------------- D (ridge)
def fit_D_scores(frames, events, S):
    """Walk-forward ridge. Returns dict (ticker, gap_date) -> predicted pnl per slot-day."""
    ot = pd.read_csv(RESULTS / "outcome_table.csv")
    def feats(df_t, df_g, df_gap):
        vn = [S["vol_norm"].get((t, g)) for t, g in zip(df_t, df_g)]
        vs = [S["vol_spike"].get((t, g)) for t, g in zip(df_t, df_g)]
        qq = [S["quiet"].get((t, g)) for t, g in zip(df_t, df_g)]
        X = pd.DataFrame({
            "abs_gap": np.abs(np.asarray(df_gap, dtype=float)),
            "vol_norm": pd.to_numeric(pd.Series(vn), errors="coerce").values,
            "log_spike": np.log(pd.to_numeric(pd.Series(vs), errors="coerce").clip(lower=1e-3)).values,
            "frac": -pd.to_numeric(pd.Series(qq), errors="coerce").values,
        })
        X["b10"] = (X.abs_gap >= 10).astype(float)
        X["b5"] = ((X.abs_gap >= 5) & (X.abs_gap < 10)).astype(float)
        return X
    Xo = feats(ot.ticker, ot.gap_date, ot.gap_pct)
    yo = (ot.pnl_pct / ot.days_held.clip(lower=1)).values
    Xe = feats(events.ticker, events.gap_date, events.gap_pct)
    year_e = events.gap_date.str[:4].astype(int).values
    scores = {}
    for Y in range(2019, 2027):
        tr = (ot.exit_date < f"{Y}-01-01").values & Xo.notna().all(axis=1).values
        if tr.sum() < 500:
            continue
        lo, hi = np.percentile(yo[tr], [1, 99])
        y = np.clip(yo[tr], lo, hi)
        mu, sd = Xo[tr].mean(), Xo[tr].std().replace(0, 1.0)
        Z = ((Xo[tr] - mu) / sd).values
        ym = y.mean()
        beta = np.linalg.solve(Z.T @ Z + RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - ym))
        sel = (year_e == Y) & Xe.notna().all(axis=1).values
        pred = ((Xe[sel] - mu) / sd).values @ beta + ym
        for t, g, p in zip(events.ticker[sel], events.gap_date[sel], pred):
            scores[(t, g)] = float(p)
        print(f"  D fold {Y}: train n={int(tr.sum())}, coefs "
              f"{dict(zip(Xo.columns, np.round(beta, 3)))}", flush=True)
    return scores


# ------------------------------------------------------------- simulation
def simulate(days, S, order_fn, sector_cap_n=None, crowd_cap=None):
    engine.MAX_SLOTS = 20
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    frac_by_date = S["frac_by_date"]
    def resting(live, pending, free):
        if free <= 0:
            return set()
        held = state["open_positions"]
        sc, crowd_open = {}, 0
        for tk, p in held.items():
            s = sector_cap.sector_of(tk)
            sc[s] = sc.get(s, 0) + 1
            if crowd_cap is not None and frac_by_date.get(p["gap_date"], 0.0) >= CROWD_FRAC:
                crowd_open += 1
        chosen = []
        for t in order_fn(live, pending, S):
            if len(chosen) >= free:
                break
            s = sector_cap.sector_of(t)
            if sector_cap_n is not None and s != sector_cap.UNKNOWN and sc.get(s, 0) >= sector_cap_n:
                continue
            is_crowd = crowd_cap is not None and frac_by_date.get(pending[t]["gap_date"], 0.0) >= CROWD_FRAC
            if is_crowd and crowd_open >= crowd_cap:
                continue
            chosen.append(t)
            sc[s] = sc.get(s, 0) + 1
            crowd_open += int(is_crowd)
        return set(chosen)
    engine.resting_orders = resting
    dates, equity, trades = [], [], []
    try:
        for today, bars in days:
            res = engine.step_one_day(state, today, bars, filler)
            trades.extend(res["trades"])
            dates.append(today)
            equity.append(res["equity"])
    finally:
        engine.resting_orders = harness._ORIG_RESTING
    return pd.Series(equity, index=pd.to_datetime(dates)), pd.DataFrame(trades)


def yearly(eq, tr, years):
    out = {}
    for y in years:
        a = max(f"{y}-01-01", TRAIN_START)
        out[y] = slice_metrics(eq, tr, a, f"{y}-12-31")
    return out


def rdd(m):
    if m["maxdd_pct"] == 0:
        return np.inf if m["return_pct"] > 0 else 0.0
    return m["return_pct"] / -m["maxdd_pct"]


def evaluate(name, eq, tr, base):
    train = slice_metrics(eq, tr, TRAIN_START, TRAIN_END)
    oos = slice_metrics(eq, tr, OOS_START, OOS_END)
    yrs = yearly(eq, tr, range(2016, 2023))
    yb = base["years"]
    ge_all = sum(rdd(yrs[y]) >= rdd(yb[y]) for y in range(2016, 2023))
    ge_oos = sum(rdd(yrs[y]) >= rdd(yb[y]) for y in range(2019, 2023))
    return {"cand": name, "train": train, "oos": oos, "years": yrs, "yrs_ge_base_7": ge_all, "yrs_ge_base_oos4": ge_oos}


def qualifies(c, base, kind):
    if kind == "train":
        m, b, ok_years = c["train"], base["train"], c["yrs_ge_base_7"] >= 4
    else:
        m, b, ok_years = c["oos"], base["oos"], c["yrs_ge_base_oos4"] >= 2
    return (m["return_pct"] >= 0.75 * b["return_pct"] and rdd(m) > rdd(b)
            and m["maxdd_pct"] > b["maxdd_pct"] and ok_years)


def main():
    frames = common.load_frames()
    events = common.gap_events(frames)
    days = harness.nse.load(10)
    S = load_feature_scores(frames, events)

    # baseline through the wrapper: must equal the engine's own sort
    beq, btr = simulate(days, S, order_depth)
    full = slice_metrics(beq, btr, TRAIN_START, LOCK_END)
    ok = len(btr) == 821 and abs(full["return_pct"] - 177.81) < 0.05 and abs(full["maxdd_pct"] + 25.96) < 0.05
    print("wrapper baseline reproduces 821 / +177.81% / -25.96%:", "MATCH" if ok else f"MISMATCH {len(btr)} {full}", flush=True)
    if not ok:
        return 1
    base = evaluate("depth_baseline", beq, btr, {"train": None, "oos": None, "years": None}) if False else None
    base = {"train": slice_metrics(beq, btr, TRAIN_START, TRAIN_END),
            "oos": slice_metrics(beq, btr, OOS_START, OOS_END),
            "years": yearly(beq, btr, range(2016, 2023))}

    print("fitting D (walk-forward ridge) ...", flush=True)
    Dscores = fit_D_scores(frames, events, S)

    runs = {}
    specs = {
        "A_crowd_throttle": dict(order_fn=order_depth, crowd_cap=CROWD_CAP),
        "B_bucket_then_quality": dict(order_fn=order_B),
        "C_rank_composite": dict(order_fn=order_C),
        "D_ridge_walkforward": dict(order_fn=make_order_D(Dscores)),
    }
    store = {}
    for name, kw in specs.items():
        eq, tr = simulate(days, S, **kw)
        c = evaluate(name, eq, tr, base)
        runs[name] = c
        store[name] = (eq, tr, kw)
        print(f"  {name}: TRAIN {c['train']} | 2019-22 {c['oos']} | yrs>=base (7yr/oos4) {c['yrs_ge_base_7']}/{c['yrs_ge_base_oos4']}", flush=True)

    kinds = {"A_crowd_throttle": "train", "B_bucket_then_quality": "train", "C_rank_composite": "train", "D_ridge_walkforward": "oos"}
    quals = [n for n in specs if qualifies(runs[n], base, kinds[n])]
    print("\nbaseline TRAIN", base["train"], "| 2019-22", base["oos"])
    print("qualifiers:", quals)

    rows = [{"cand": "depth_baseline", **{f"train_{k}": v for k, v in base["train"].items()}, **{f"oos_{k}": v for k, v in base["oos"].items()}}]
    yrows = [{"cand": "depth_baseline", **{y: round(rdd(m), 2) for y, m in base["years"].items()}}]
    for n, c in runs.items():
        rows.append({"cand": n, "qualifies": n in quals, "yrs_ge_base_7": c["yrs_ge_base_7"], "yrs_ge_base_oos4": c["yrs_ge_base_oos4"],
                     **{f"train_{k}": v for k, v in c["train"].items()}, **{f"oos_{k}": v for k, v in c["oos"].items()}})
        yrows.append({"cand": n, **{y: round(rdd(m), 2) for y, m in c["years"].items()}})

    final_name, final_run = None, None
    if quals:
        X = max(quals, key=lambda n: rdd(runs[n]["oos"]))
        print(f"X = {X} (highest 2019-22 ret/DD among qualifiers)")
        kw = dict(store[X][2]); kw["sector_cap_n"] = SECTOR_CAP
        eq, tr = simulate(days, S, **kw)
        cE = evaluate("E_" + X + "+sector_cap4", eq, tr, base)
        runs["E"] = cE
        qE = qualifies(cE, base, kinds[X])
        better = rdd(cE["oos"]) > rdd(runs[X]["oos"])
        print(f"  E ({X} + sector cap 4): TRAIN {cE['train']} | 2019-22 {cE['oos']} | qualifies={qE} | beats X on 2019-22 ret/DD={better}", flush=True)
        rows.append({"cand": cE["cand"], "qualifies": qE, "yrs_ge_base_7": cE["yrs_ge_base_7"], "yrs_ge_base_oos4": cE["yrs_ge_base_oos4"],
                     **{f"train_{k}": v for k, v in cE["train"].items()}, **{f"oos_{k}": v for k, v in cE["oos"].items()}})
        yrows.append({"cand": cE["cand"], **{y: round(rdd(m), 2) for y, m in cE["years"].items()}})
        use_E = qE and better
        final_name = cE["cand"] if use_E else X
        final_eq, final_tr = (eq, tr) if use_E else store[X][:2]
    pd.DataFrame(rows).to_csv(RESULTS / "candidates_train.csv", index=False)
    pd.DataFrame(yrows).to_csv(RESULTS / "candidates_years.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nper-year ret/DD (2016 partial):")
    print(pd.DataFrame(yrows).to_string(index=False))
    if final_name is None:
        print("\nNO QUALIFIER. Lockbox NOT run. Depth-only stays.")
        return 0
    print(f"\nFINAL candidate for the lockbox: {final_name}")
    lb_b = slice_metrics(beq, btr, LOCK_START, LOCK_END)
    lb_c = slice_metrics(final_eq, final_tr, LOCK_START, LOCK_END)
    passed = (rdd(lb_c) >= 1.25 * rdd(lb_b) and lb_c["return_pct"] >= 0.75 * lb_b["return_pct"] and lb_c["maxdd_pct"] > lb_b["maxdd_pct"])
    out = pd.DataFrame([{"cand": "depth_baseline", **lb_b}, {"cand": final_name, **lb_c}])
    out.to_csv(RESULTS / "candidates_lockbox.csv", index=False)
    print("LOCKBOX 2023-01-03..2026-09-04 (scored once):")
    print(out.to_string(index=False))
    print("PASS BAR (ret/DD >= 1.25x, return >= 75%, smaller maxDD):", "PASS" if passed else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
