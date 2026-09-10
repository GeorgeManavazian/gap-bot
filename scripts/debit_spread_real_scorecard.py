"""Aggregate debit_spread_real_backtest.py's raw per-trade results into a
scorecard, isolated-per-bucket portfolio equity curve (matching
portfolio_sim_isolated_buckets.py's CAPITAL=$100k / MAX_SLOTS=20 shape), and
coverage, liquidity and BS-fallback reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from debit_spread_real_backtest import ANCHOR, CAPITAL, MAX_SLOTS, GAP

RAW = GAP / "results/_debit_spread_real_raw.pkl"


def build_equity_curve(trades: pd.DataFrame) -> dict:
    """trades: status=='ok' rows for ONE (bucket, dte, width) cell, with
    entry_date, entry_cost, effective_exit, hold_dates, daily_marks."""
    if not len(trades):
        return {"max_drawdown_pct": None, "total_return_pct": None, "final_equity": None}
    trades = trades.sort_values("entry_date").reset_index(drop=True)
    mark_by_id = {}
    for i, r in trades.iterrows():
        mark_by_id[i] = dict(zip([pd.Timestamp(d) for d in r.hold_dates], r.daily_marks))

    all_days = sorted(set(d for r in trades.itertuples() for d in r.hold_dates))
    cash = CAPITAL
    open_pos = {}  # idx -> {contracts, last_mark, effective_exit}
    zero_contract_skips = 0
    entries_by_day = trades.groupby(trades.entry_date.dt.normalize()).apply(
        lambda g: list(g.index)).to_dict()
    equity_curve = []

    for day in all_days:
        # close positions whose effective_exit is today (mark then release)
        for idx in list(open_pos.keys()):
            if pd.Timestamp(open_pos[idx]["effective_exit"]).normalize() == pd.Timestamp(day).normalize():
                r = trades.loc[idx]
                exit_val = r.exit_value
                cash += open_pos[idx]["contracts"] * exit_val
                del open_pos[idx]

        # open new positions entering today, capacity permitting
        for idx in entries_by_day.get(pd.Timestamp(day).normalize(), []):
            if len(open_pos) >= MAX_SLOTS:
                continue
            r = trades.loc[idx]
            mtm = sum(open_pos[j]["contracts"] * open_pos[j]["last_mark"] for j in open_pos)
            equity_now = cash + mtm
            budget = equity_now / MAX_SLOTS
            contracts = int(budget // r.entry_cost) if r.entry_cost > 0 else 0
            if contracts <= 0:
                zero_contract_skips += 1
                continue
            cost = contracts * r.entry_cost
            if cost > cash:
                contracts = int(cash // r.entry_cost)
                if contracts <= 0:
                    zero_contract_skips += 1
                    continue
                cost = contracts * r.entry_cost
            cash -= cost
            open_pos[idx] = {"contracts": contracts, "last_mark": r.entry_cost,
                              "effective_exit": r.effective_exit}

        # mark to market
        for idx in open_pos:
            m = mark_by_id[idx].get(pd.Timestamp(day))
            if m is not None:
                open_pos[idx]["last_mark"] = m
        mtm = sum(open_pos[j]["contracts"] * open_pos[j]["last_mark"] for j in open_pos)
        equity_curve.append(cash + mtm)

    eq = pd.Series(equity_curve, index=all_days)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return {
        "max_drawdown_pct": round(dd.min() * 100, 2),
        "total_return_pct": round((eq.iloc[-1] / CAPITAL - 1) * 100, 2),
        "final_equity": round(eq.iloc[-1], 2),
        "zero_contract_skips": zero_contract_skips,
    }


def main():
    raw = pd.read_pickle(RAW)
    anchor = pd.read_csv(ANCHOR, parse_dates=["gap_date", "entry_date", "exit_date"])
    n_anchor_tickers = anchor.ticker.nunique()

    print("=== status counts (all trade x dte x width evaluations) ===")
    print(raw.status.value_counts())
    print()

    no_chain = raw[raw.status == "no_chain_data"]
    tickers_no_chain = sorted(no_chain.ticker.unique())
    print(f"tickers with NO chain data at all: {len(tickers_no_chain)} / {n_anchor_tickers}")
    print(tickers_no_chain)
    print()

    ok = raw[raw.status == "ok"].copy()
    ok["pnl_pct_of_notional"] = ok.pnl / ok.stock_entry_price * 100
    # stock_entry_price here is the CSV's (split-adjusted) entry_price, used
    # ONLY as the stock-comparison notional denominator -- never for strike
    # math, which used the real underlying_price (see backtest's real_S).

    print(f"ok trades: {len(ok)} of {len(raw)} evaluations "
          f"({len(ok)/len(raw)*100:.1f}%)")
    print(f"bs_fallback rate (ok trades): {ok.bs_fallback_any.mean()*100:.1f}%")
    print(f"itm_short_any rate (ok trades): {ok.itm_short_any.mean()*100:.1f}%")
    print(f"expiry_forced_exit rate (ok trades): {ok.expiry_forced_exit.mean()*100:.1f}%")
    print(f"oi_known_both_legs rate (ok trades): {ok.oi_known_both_legs.mean()*100:.1f}%")
    print(f"zero_volume_either_leg rate (ok trades): {ok.zero_volume_either_leg.mean()*100:.1f}%")
    print(f"width_floored rate (ok trades): {ok.width_floored.mean()*100:.1f}%")
    print(f"median long_spread_pct / short_spread_pct: "
          f"{ok.long_spread_pct.median():.1f}% / {ok.short_spread_pct.median():.1f}%")
    print()

    illiquid = raw[raw.status == "illiquid_low_volume"]
    print(f"excluded for volume<{4}: {len(illiquid)} evaluations "
          f"({len(illiquid)/len(raw)*100:.1f}%)")
    no_expiry = raw[raw.status == "no_expiry"]
    print(f"no listed expiry in band: {len(no_expiry)} evaluations "
          f"({len(no_expiry)/len(raw)*100:.1f}%)")
    print()

    rows = []
    for (bucket, dte, wf), g in ok.groupby(["bucket", "target_dte", "width_frac"]):
        all_cell = raw[(raw.bucket == bucket) & (raw.target_dte == dte) & (raw.width_frac == wf)]
        n_total_cell = len(all_cell)
        n_ok = len(g)
        eqc = build_equity_curve(g)
        rows.append({
            "bucket": bucket, "dte": dte, "width_frac": wf,
            "n_trades": n_ok,
            "n_total_bucket_trades": n_total_cell,
            "coverage_pct": round(n_ok / n_total_cell * 100, 1) if n_total_cell else None,
            "win_rate_pct": round((g.pnl > 0).mean() * 100, 1),
            "avg_pnl_pct_of_notional": round(g.pnl_pct_of_notional.mean(), 3),
            "avg_roc_on_premium_pct": round(g.roc_pct.mean(), 1),
            "median_roc_on_premium_pct": round(g.roc_pct.median(), 1),
            **eqc,
            "bs_fallback_pct": round(g.bs_fallback_any.mean() * 100, 1),
            "itm_short_leg_pct": round(g.itm_short_any.mean() * 100, 1),
            "expiry_forced_exit_pct": round(g.expiry_forced_exit.mean() * 100, 1),
            "oi_known_both_legs_pct": round(g.oi_known_both_legs.mean() * 100, 1),
        })
    scorecard = pd.DataFrame(rows).sort_values(["bucket", "dte", "width_frac"])
    out = GAP / "results/debit_spread_real_scorecard.csv"
    scorecard.to_csv(out, index=False)
    print(f"scorecard written: {out} ({len(scorecard)} rows)")

    trades_out = GAP / "results/debit_spread_real_trades.csv"
    keep = ["ticker", "bucket", "target_dte", "width_frac", "gap_pct", "entry_date",
            "exit_date", "exit_reason", "expiration", "actual_dte", "long_strike",
            "short_strike", "width_dollars", "entry_cost", "exit_value", "pnl",
            "roc_pct", "pnl_pct_of_notional", "max_value_reached", "bs_fallback_any",
            "itm_short_any", "expiry_forced_exit", "oi_known_both_legs",
            "zero_volume_either_leg", "width_floored", "long_volume", "short_volume",
            "long_oi", "short_oi", "long_spread_pct", "short_spread_pct"]
    ok[keep].to_csv(trades_out, index=False)
    print(f"per-trade detail written: {trades_out} ({len(ok)} rows, all configs)")

    print()
    print("=== headline: 45 DTE, width=1.0 (comparable to the first-order pass) ===")
    head = scorecard[(scorecard.dte == 45) & (scorecard.width_frac == 1.0)]
    print(head[["bucket", "n_trades", "n_total_bucket_trades", "coverage_pct",
                "win_rate_pct", "avg_pnl_pct_of_notional", "avg_roc_on_premium_pct",
                "max_drawdown_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
