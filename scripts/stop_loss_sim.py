"""Same down-gap trading rule (buy Open on a down gap, TP at prior close),
now with a stop-loss added. Walks forward day-by-day within the 63-day
horizon; on any day where both the stop and the TP level are touched, the
STOP is assumed to fire first (conservative -- no intraday sequencing data
exists to know which came first, so this cannot overstate performance).

Tests two stop schemes per bucket:
  A. data-derived: p90 MAE from the no-stop run, rounded -- "wide enough to
     survive 90% of eventual winners' own path, tight enough to bound the
     CVNA/PCG/NCLH-style tail".
  B. a flat sensitivity sweep (5/10/15/20/25/30%) applied to EVERY bucket,
     so the tight-vs-wide tradeoff is visible rather than asserted.

No re-download -- reuses daily_bars/ already on disk.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
HORIZON = 63

BUCKETS = [("1-2%", 1.0, 2.0), ("2-3%", 2.0, 3.0), ("3-5%", 3.0, 5.0),
           ("5-10%", 5.0, 10.0), ("10%+", 10.0, np.inf)]

# Scheme A: data-derived per-bucket stop, from the p90 MAE table computed
# earlier (mae_table.csv), rounded to a clean number.
DATA_STOPS = {"1-2%": 8.0, "2-3%": 11.0, "3-5%": 15.0, "5-10%": 24.0, "10%+": 35.0}
FLAT_STOPS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]


def bucket_of(abs_gap):
    for name, lo, hi in BUCKETS:
        if lo <= abs_gap < hi:
            return name
    return None


def load_ticker(path):
    df = pd.read_parquet(path)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    need = {"Date", "Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < HORIZON + 5:
        return None
    return df


def simulate(df, ticker, stop_for_bucket):
    """stop_for_bucket: dict bucket_name -> stop_pct (positive number, e.g. 8.0)"""
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    out = []
    for t in range(1, n - 1):
        prior_close = c[t - 1]
        if prior_close <= 0:
            continue
        gap_pct = (o[t] - prior_close) / prior_close * 100
        if gap_pct >= -1.0:
            continue
        b = bucket_of(abs(gap_pct))
        stop_pct = stop_for_bucket.get(b)
        if stop_pct is None:
            continue
        entry = o[t]
        stop_price = entry * (1 - stop_pct / 100)
        end = min(t + HORIZON, n - 1)
        outcome, pnl_pct, days = "unresolved", None, None
        for k in range(t, end + 1):
            hit_stop = l[k] <= stop_price
            hit_tp = h[k] >= prior_close
            if hit_stop and hit_tp:
                outcome, pnl_pct, days = "stopped", -stop_pct, k - t
                break
            if hit_stop:
                outcome, pnl_pct, days = "stopped", -stop_pct, k - t
                break
            if hit_tp:
                outcome, pnl_pct, days = "win", (prior_close - entry) / entry * 100, k - t
                break
        if outcome == "unresolved":
            pnl_pct = (c[end] - entry) / entry * 100  # mark-to-market, not realized
            days = end - t
        out.append({"ticker": ticker, "bucket": b, "outcome": outcome,
                    "pnl_pct": pnl_pct, "days": days})
    return out


def summarize(rows, label):
    df = pd.DataFrame(rows)
    print(f"\n--- {label} ---")
    out_rows = []
    for name, lo, hi in BUCKETS:
        sub = df[df["bucket"] == name]
        n = len(sub)
        if n == 0:
            continue
        win = sub[sub["outcome"] == "win"]
        stop = sub[sub["outcome"] == "stopped"]
        unres = sub[sub["outcome"] == "unresolved"]
        expectancy = sub["pnl_pct"].mean()  # unresolved counted at mark-to-market, not realized
        expectancy_realized_only = pd.concat([win, stop])["pnl_pct"].mean() if len(win) + len(stop) else None
        out_rows.append({
            "bucket": name, "n": n,
            "win_%": round(100 * len(win) / n, 1),
            "stopped_%": round(100 * len(stop) / n, 1),
            "unresolved_%": round(100 * len(unres) / n, 1),
            "avg_win_%": round(win["pnl_pct"].mean(), 2) if len(win) else None,
            "avg_days_to_win": round(win["days"].mean(), 1) if len(win) else None,
            "expectancy_%_per_trade_all_incl_mtm": round(expectancy, 3),
            "expectancy_%_realized_only": round(expectancy_realized_only, 3) if expectancy_realized_only is not None else None,
        })
    t = pd.DataFrame(out_rows)
    print(t.to_string(index=False))
    return t


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    dfs = []
    for p in files:
        d = load_ticker(p)
        if d is not None:
            dfs.append((p.stem, d))
    print(f"{len(dfs)} tickers loaded")

    # Scheme A: data-derived per-bucket stop
    rows_a = []
    for tk, d in dfs:
        rows_a.extend(simulate(d, tk, DATA_STOPS))
    table_a = summarize(rows_a, f"SCHEME A: data-derived stop per bucket {DATA_STOPS}")
    table_a.to_csv((HERE.parent / "results" / "stop_sim_data_derived.csv"), index=False)

    # Scheme B: flat stop sweep, one number applied to every bucket
    sweep_rows = []
    for flat in FLAT_STOPS:
        rows_b = []
        stops = {name: flat for name, _, _ in BUCKETS}
        for tk, d in dfs:
            rows_b.extend(simulate(d, tk, stops))
        dfb = pd.DataFrame(rows_b)
        win = dfb[dfb["outcome"] == "win"]
        stop = dfb[dfb["outcome"] == "stopped"]
        n = len(dfb)
        sweep_rows.append({
            "flat_stop_%": flat,
            "n_trades": n,
            "win_%": round(100 * len(win) / n, 1),
            "stopped_%": round(100 * len(stop) / n, 1),
            "expectancy_%_per_trade_all": round(dfb["pnl_pct"].mean(), 3),
        })
    print("\n--- SCHEME B: flat stop sweep, ALL down-gap buckets pooled ---")
    sweep = pd.DataFrame(sweep_rows)
    print(sweep.to_string(index=False))
    sweep.to_csv((HERE.parent / "results" / "stop_sweep_flat.csv"), index=False)

    # baseline, no stop, for direct comparison (recompute quickly)
    rows_none = []
    NO_STOP = {name: 1e9 for name, _, _ in BUCKETS}  # effectively infinite, never trips
    for tk, d in dfs:
        rows_none.extend(simulate(d, tk, NO_STOP))
    summarize(rows_none, "BASELINE: no stop at all (for comparison)")


if __name__ == "__main__":
    main()
