"""Trade-level follow-up to scripts/slot_sweep_honest_rule.py (hub request
2026-09-08): what the per-arm summary table can't show.

Replays every slot count through engine.step_one_day() (honest rule:
resting-limit fill + rest-top-N priority, 2yr held-out window) and keeps,
per arm: every trade, every day's open set, every touch that had no order
resting ("missed"), and the daily equity. Then answers, with data:

  A. trade identity across arms -- is the trade set nested as slots grow,
     and is what the smaller arms drop really an even "weak tail";
  B. where the dropped trades cluster (bucket / sector / month);
  C. concentration the maxDD hides -- sector share of the open book,
     worst single days, same-day entry clusters;
  D. behaviour through the window's stress episodes (2025-04 tariff
     crash and rebound, 2024-12-18, 2025-10-10, 2026-03-12, 2026-06-17);
  E. half-year breakdown per arm (never one blended number);
  F. dependence on a handful of winners;
  G. exit composition and slot utilisation;
  H. the money left on the table by missed touches, valued off the
     50-slot arm's realised outcome for the same (ticker, gap_date);
  I. selection vs sizing: select top-N but size as equity/D, N x D grid,
     to separate "which trades" from "how much per trade".

Outputs: results/slot_sweep_trades/{trades,daily,missed}_<n>.csv and the
grid in results/slot_sweep_select_vs_size.csv. Measurement only.
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from live import engine
from live.config import CAPITAL, HORIZON, VARIANTS
from live.fillers import StockFiller, make_filler
from priority_lookahead_check import load

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "slot_sweep_trades"
SLOTS = [5, 10, 15, 20, 30, 40, 50]
STRESS = {
    "2024-12 Fed day": ("2024-12-16", "2024-12-31"),
    "2025-04 tariff crash+rebound": ("2025-04-01", "2025-04-30"),
    "2025-10-10": ("2025-10-08", "2025-10-24"),
    "2026-03-12": ("2026-03-09", "2026-03-27"),
    "2026-06-17": ("2026-06-15", "2026-06-30"),
}
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def sector_map() -> dict:
    df = pd.read_csv(ROOT / "data" / "sp500_constituents.csv")
    return {s.replace(".", "-"): g for s, g in zip(df["Symbol"], df["GICS Sector"])}


class ScaledSizeFiller(StockFiller):
    """Size every entry as equity / size_div instead of equity / MAX_SLOTS,
    leaving selection (MAX_SLOTS) alone. Used only by the N x D grid."""
    def __init__(self, size_div: int):
        super().__init__(commission_per_trade=0.0)
        self.size_div = size_div

    def price_entry(self, ticker, entry_price, size_dollars, chain_provider=None, tp_price=None):
        return super().price_entry(ticker, entry_price, size_dollars * engine.MAX_SLOTS / self.size_div,
                                   chain_provider, tp_price)


def replay(days, n_slots: int, filler=None, keep_detail: bool = True):
    engine.MAX_SLOTS = n_slots
    filler = filler or make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades, daily, missed = [], [], []
    for today, bars in days:
        # what the engine is about to do with touches, computed identically
        exits_today = set()
        for tk, pos in state["open_positions"].items():
            r = bars.get(tk)
            dh = pos["days_held"] + 1
            if (r is None and dh >= HORIZON) or (r is not None and (
                    r["l"] <= pos["stop_price"] or r["h"] >= pos["tp_price"] or dh >= HORIZON)):
                exits_today.add(tk)
        live = [tk for tk, w in state["pending"].items()
                if w["gap_date"] != today and w["days_waited"] + 1 <= HORIZON and tk not in state["open_positions"]]
        free = n_slots - len(state["open_positions"]) + len(exits_today)
        resting = engine.resting_orders(live, state["pending"], free)
        for tk in live:
            r = bars.get(tk)
            if r is not None and engine.limit_touched(r, state["pending"][tk]["gap_open"]) and tk not in resting:
                w = state["pending"][tk]
                missed.append({"date": today, "ticker": tk, "gap_date": w["gap_date"], "gap_pct": w["gap_pct"]})
        res = engine.step_one_day(state, today, bars, filler)
        for t in res["trades"]:
            trades.append({**t, "slots": n_slots})
        if keep_detail:
            daily.append({"date": today, "equity": res["equity"], "cash": state["cash"],
                          "n_open": len(state["open_positions"]),
                          "open": ",".join(sorted(state["open_positions"])),
                          "n_entries": sum(1 for p in state["open_positions"].values() if p["entry_date"] == today),
                          "n_exits": len(res["trades"]), "n_resting": len(resting), "n_live": len(live)})
        else:
            daily.append({"date": today, "equity": res["equity"]})
    return pd.DataFrame(trades), pd.DataFrame(daily), pd.DataFrame(missed)


def stats(eq: pd.Series) -> dict:
    ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    d = eq.pct_change().dropna()
    return {"return_pct": round(ret, 2), "maxdd_pct": round(dd, 2),
            "ret_over_dd": round(ret / -dd, 2) if dd < 0 else None,
            "worst_day_pct": round(d.min() * 100, 2), "worst_5d_pct": round((eq / eq.shift(5) - 1).min() * 100, 2),
            "daily_vol_pct": round(d.std() * 100, 2), "days_lt_-2pct": int((d < -0.02).sum())}


def main():
    days = load(2)
    sectors = sector_map()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {}
    for n in SLOTS:
        t, d, m = replay(days, n)
        t["key"] = list(zip(t["ticker"], t["gap_date"]))
        t["sector"] = t["ticker"].map(sectors).fillna("?")
        t["entry_month"] = t["entry_date"].str[:7]
        arms[n] = (t, d, m)
        t.drop(columns=["key"]).to_csv(OUT / f"trades_{n}.csv", index=False)
        d.to_csv(OUT / f"daily_{n}.csv", index=False)
        m.to_csv(OUT / f"missed_{n}.csv", index=False)
    t20 = arms[20][0]
    assert len(t20) == 174, f"anchor drift: {len(t20)} trades at 20 slots"
    print("anchor reproduction at 20 slots: MATCH (174 trades)")

    # ---------------- A. trade identity across arms
    print("\n=== A. Trade identity: is the trade set nested as slots grow? ===")
    keys = {n: set(arms[n][0]["key"]) for n in SLOTS}
    rows = []
    for n in SLOTS:
        for m in SLOTS:
            if m <= n:
                continue
            rows.append({"small": n, "large": m, "n_small": len(keys[n]), "n_large": len(keys[m]),
                         "in_both": len(keys[n] & keys[m]), "only_in_small": len(keys[n] - keys[m]),
                         "only_in_large": len(keys[m] - keys[n])})
    A = pd.DataFrame(rows)
    print(A[A.large == 20].to_string(index=False))
    print(A[(A.small == 20)].to_string(index=False))
    print("\n  persistence: in how many of the 7 arms does each (ticker, gap_date) trade appear, and how did it pay (50-slot arm's outcome)?")
    t50 = arms[50][0].set_index("key")
    pers = Counter()
    for n in SLOTS:
        for k in keys[n]:
            pers[k] += 1
    pdf = pd.DataFrame({"key": list(pers), "arms": list(pers.values())})
    pdf["pnl_pct_50"] = pdf["key"].map(lambda k: t50.loc[[k], "pnl_pct"].iloc[0] if k in t50.index else np.nan)
    pdf["gap_pct"] = pdf["key"].map(lambda k: t50.loc[[k], "gap_pct"].iloc[0] if k in t50.index else np.nan)
    g = pdf.groupby("arms").agg(n=("key", "size"), mean_pnl=("pnl_pct_50", "mean"), median_pnl=("pnl_pct_50", "median"),
                                win=("pnl_pct_50", lambda s: (s > 0).mean() * 100), mean_gap=("gap_pct", "mean"))
    print(g.round(2).to_string())
    print("  (arms=7 -> traded at every slot count incl. 5; arms=1 -> only the 50-slot arm took it)")

    # what the 10-arm gives up vs 20, valued at the 20-arm's own outcome
    only20 = t20[~t20["key"].isin(keys[10])]
    both = t20[t20["key"].isin(keys[10])]
    print(f"\n  20-arm trades also taken by the 10-arm: n={len(both)}, mean {both.pnl_pct.mean():+.2f}%, "
          f"median {both.pnl_pct.median():+.2f}%, win {100*(both.pnl_pct>0).mean():.0f}%, sum ${both.pnl_dollar.sum():,.0f}")
    print(f"  20-arm trades the 10-arm did NOT take:    n={len(only20)}, mean {only20.pnl_pct.mean():+.2f}%, "
          f"median {only20.pnl_pct.median():+.2f}%, win {100*(only20.pnl_pct>0).mean():.0f}%, sum ${only20.pnl_dollar.sum():,.0f}")
    print("  dropped trades, P&L distribution (pct):", only20.pnl_pct.describe().round(2).to_dict())
    print("  dropped trades by bucket:"); print(only20.groupby("bucket").pnl_pct.agg(["size", "mean", "median"]).round(2).to_string())
    print("  kept trades by bucket:"); print(both.groupby("bucket").pnl_pct.agg(["size", "mean", "median"]).round(2).to_string())

    # ---------------- B. where the dropped trades cluster
    print("\n=== B. Dropped (20 vs 10) trades: sector / month clustering ===")
    print("  by sector (dropped vs kept):")
    sec = pd.DataFrame({"dropped": only20.groupby("sector").size(), "kept": both.groupby("sector").size(),
                        "dropped_mean_pnl": only20.groupby("sector").pnl_pct.mean(),
                        "kept_mean_pnl": both.groupby("sector").pnl_pct.mean()}).fillna(0).round(2)
    print(sec.sort_values("dropped", ascending=False).to_string())
    print("  by entry month (dropped count / dropped P&L$ / kept count):")
    mon = pd.DataFrame({"dropped": only20.groupby("entry_month").size(), "dropped_pnl$": only20.groupby("entry_month").pnl_dollar.sum().round(0),
                        "kept": both.groupby("entry_month").size()}).fillna(0)
    print(mon.to_string())

    # ---------------- C. concentration
    print("\n=== C. Concentration the maxDD hides ===")
    rows = []
    for n in SLOTS:
        t, d, _ = arms[n]
        sec_share, hhi, one_sector_days = [], [], 0
        for s in d["open"]:
            names = [x for x in s.split(",") if x]
            if not names:
                continue
            c = Counter(sectors.get(x, "?") for x in names)
            top = max(c.values()) / len(names)
            sec_share.append(top)
            hhi.append(sum((v / len(names)) ** 2 for v in c.values()))
            one_sector_days += top >= 0.5
        eq = d.set_index("date")["equity"]
        st = stats(eq)
        rows.append({"slots": n, **st, "avg_top_sector_share": round(np.mean(sec_share) * 100, 1),
                     "max_top_sector_share": round(max(sec_share) * 100, 1),
                     "days_>=50%_one_sector": one_sector_days, "avg_sector_HHI": round(np.mean(hhi), 2),
                     "days_>=3_entries": int((d.n_entries >= 3).sum()), "max_entries_one_day": int(d.n_entries.max()),
                     "days_at_cap_pct": round(100 * (d.n_open >= n).mean(), 1)})
    C = pd.DataFrame(rows)
    print(C.to_string(index=False))
    print("  worst single days per arm (date, portfolio return, open book that day):")
    for n in (5, 10, 15, 20):
        d = arms[n][1].copy(); d["ret"] = d.equity.pct_change() * 100
        w = d.nsmallest(3, "ret")
        for _, r in w.iterrows():
            names = r["open"].split(",")
            c = Counter(sectors.get(x, "?") for x in names if x)
            print(f"    {n:>2} slots  {r.date}  {r.ret:+.2f}%  n_open={r.n_open}  sectors={dict(c.most_common(3))}")

    # ---------------- D. stress episodes
    print("\n=== D. Stress episodes ===")
    rows = []
    for name, (a, b) in STRESS.items():
        for n in SLOTS:
            t, d, m = arms[n]
            w = d[(d.date >= a) & (d.date <= b)].set_index("date")["equity"]
            pre = d[d.date < a]["equity"].iloc[-1] if (d.date < a).any() else CAPITAL
            ent = t[(t.entry_date >= a) & (t.entry_date <= b)]
            stops = t[(t.exit_date >= a) & (t.exit_date <= b) & (t.exit_reason == "stop")]
            miss = m[(m.date >= a) & (m.date <= b)]
            rows.append({"episode": name, "slots": n, "window_ret_pct": round((w.iloc[-1] / pre - 1) * 100, 2),
                         "dd_in_window_pct": round(((w - w.cummax()) / w.cummax()).min() * 100, 2),
                         "entries": len(ent), "entries_eventual_pnl$": round(ent.pnl_dollar.sum(), 0),
                         "entries_eventual_mean_pct": round(ent.pnl_pct.mean(), 2) if len(ent) else None,
                         "stop_outs": len(stops), "missed_touches": len(miss)})
    D = pd.DataFrame(rows)
    for name in STRESS:
        print(f"  -- {name}"); print(D[D.episode == name].drop(columns="episode").to_string(index=False))
    # the tariff rebound specifically: what happened to the watches registered Apr 3-4 2025
    print("\n  April 2025 in detail (20-slot arm): watches registered on the crash days and what became of them")
    t, d, m = arms[20]
    reg = t[(t.gap_date >= "2025-04-03") & (t.gap_date <= "2025-04-10")]
    print(f"    trades whose gap_date fell 2025-04-03..10: {len(reg)}, mean {reg.pnl_pct.mean():+.2f}% (n) -> entered on: {sorted(reg.entry_date.unique())}")
    mm = m[(m.gap_date >= "2025-04-03") & (m.gap_date <= "2025-04-10")]
    print(f"    missed touches (no order resting) on watches from those days: {len(mm)}")
    # how many crash-day gaps were registered, and how many ever traded through gap_open (limit-fillable) within HORIZON?
    reg_days = [dd for dd in days if "2025-04-03" <= dd[0] <= "2025-04-10"]
    gaps = {}
    for today, bars in reg_days:
        for tk, r in bars.items():
            pr = r.get("prior_close")
            if pr and pr > 0 and (r["o"] - pr) / pr * 100 < -2.0 and tk not in gaps:
                gaps[tk] = (today, r["o"])
    fillable = 0
    for tk, (gd, go) in gaps.items():
        later = [bars for today, bars in days if today > gd][:HORIZON]
        if any(tk in b and engine.limit_touched(b[tk], go) for b in later):
            fillable += 1
    print(f"    gaps >2% registered 2025-04-03..10: {len(gaps)}; of those, ever limit-fillable (traded through gap_open) within {HORIZON}d: {fillable}")

    # ---------------- E. half-year breakdown
    print("\n=== E. Half-year returns per arm (compounded within each half) ===")
    rows = []
    for n in SLOTS:
        d = arms[n][1].copy(); d["half"] = d.date.str[:4] + "H" + ((d.date.str[5:7].astype(int) > 6) + 1).astype(str)
        r = {"slots": n}
        prev = CAPITAL
        for h, g in d.groupby("half"):
            r[h] = round((g.equity.iloc[-1] / prev - 1) * 100, 2)
            prev = g.equity.iloc[-1]
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- F. dependence on few winners
    print("\n=== F. Dependence on a handful of winners ===")
    rows = []
    for n in SLOTS:
        t = arms[n][0].sort_values("pnl_dollar", ascending=False)
        tot = t.pnl_dollar.sum()
        rows.append({"slots": n, "total_pnl$": round(tot), "top1$": round(t.pnl_dollar.iloc[0]), "top5_share_pct": round(100 * t.pnl_dollar.head(5).sum() / tot, 1) if tot else None,
                     "pnl_without_top3$": round(tot - t.pnl_dollar.head(3).sum()), "pnl_without_top5$": round(tot - t.pnl_dollar.head(5).sum()),
                     "worst_trade$": round(t.pnl_dollar.iloc[-1]), "bottom5$": round(t.pnl_dollar.tail(5).sum()),
                     "top3": ", ".join(f"{r.ticker}({r.pnl_pct:+.0f}%)" for r in t.head(3).itertuples())})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- G. exits and utilisation
    print("\n=== G. Exit composition ===")
    rows = []
    for n in SLOTS:
        t = arms[n][0]
        c = t.exit_reason.value_counts()
        rows.append({"slots": n, **{k: int(c.get(k, 0)) for k in ("tp_gap_filled", "stop", "time_exit", "time_exit_no_data")},
                     "tp_mean_pct": round(t[t.exit_reason == "tp_gap_filled"].pnl_pct.mean(), 2),
                     "stop_mean_pct": round(t[t.exit_reason == "stop"].pnl_pct.mean(), 2),
                     "time_mean_pct": round(t[t.exit_reason.str.startswith("time")].pnl_pct.mean(), 2),
                     "mean_days_held": round(t.days_held.mean(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- H. missed touches, valued
    print("\n=== H. Touches with no order resting, valued at the 50-arm's realised outcome for the same (ticker, gap_date) ===")
    rows = []
    for n in SLOTS:
        m = arms[n][2].copy()
        if m.empty:
            rows.append({"slots": n, "missed": 0}); continue
        m["key"] = list(zip(m.ticker, m.gap_date))
        m["pnl_pct_50"] = m["key"].map(lambda k: t50.loc[[k], "pnl_pct"].iloc[0] if k in t50.index else np.nan)
        known = m.dropna(subset=["pnl_pct_50"])
        rows.append({"slots": n, "missed": len(m), "valued": len(known), "mean_pnl_pct": round(known.pnl_pct_50.mean(), 2),
                     "median_pnl_pct": round(known.pnl_pct_50.median(), 2), "win_pct": round(100 * (known.pnl_pct_50 > 0).mean(), 1),
                     "mean_gap_pct": round(m.gap_pct.mean(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- I. selection vs sizing grid
    print("\n=== I. Selection (top-N) vs sizing (equity/D) ===")
    rows = []
    for n in (10, 15, 20):
        for dv in (10, 15, 20, 30):
            t, d, _ = replay(days, n, filler=ScaledSizeFiller(dv), keep_detail=False)
            st = stats(d.set_index("date")["equity"])
            rows.append({"select_top_N": n, "size_equity_over": dv, "n_trades": len(t), "win_pct": round(100 * (t.pnl_pct > 0).mean(), 1), **st})
    G = pd.DataFrame(rows)
    G.to_csv(ROOT / "results" / "slot_sweep_select_vs_size.csv", index=False)
    print(G.to_string(index=False))
    print("  (N == D rows are the plain slot-count arms; D > N = same trades, less capital each; a D < N cell is over-invested by design)")


if __name__ == "__main__":
    main()
