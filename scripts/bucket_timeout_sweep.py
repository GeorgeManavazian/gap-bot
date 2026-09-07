"""Per-bucket exit timeout vs the flat 63-day HORIZON, on the anchor config
(wick entry, gap_desc priority, 20 slots, 1-2% bucket excluded, $0 commission,
10bps slippage, 2yr window). HORIZON governs two clocks in run_sim: the
pre-entry wait-for-a-wick clock and the post-entry hold-until-time_exit clock.
This variant makes both clocks per-bucket instead of one flat value for every
bucket, using the size of the gap (not the ticker) to pick the horizon.

Baseline (flat 63, this exact anchor, 2yr, friction on):
  n=501, return=65.42%, maxDD=-17.69%, ret/maxDD=3.697
  results/confirm2yr_final_exclude_1-2pct_friction.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

from slot_and_priority_sweep import (
    BARS_DIR, CAPITAL, MAX_SLOTS, load_ticker, TickerView,
    bucket_matches, bucket_name_for, stop_for_abs_gap,
)

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
WINDOW_YEARS = 2
FLAT_HORIZON = 63


def run_sim_bucket_horizon(tickers_data, calendar, entry_style, scope_lo_hi,
                            horizon_by_bucket, max_slots=MAX_SLOTS,
                            priority="alpha", commission_per_trade=0.0,
                            slippage_bps=0.0, stop_fn=stop_for_abs_gap):
    """Same as slot_and_priority_sweep.run_sim, except HORIZON is looked up
    per-bucket (by the gap's own bucket name) instead of one flat constant,
    for both the pre-entry wick-wait clock and the post-entry time_exit
    clock. entry_style must be 'wick' or 'close_confirm' (the pending state
    machine) -- 'naive' has no pre-entry clock so it's not supported here."""
    assert entry_style in ("wick", "close_confirm")
    slip = slippage_bps / 10_000.0
    cash = CAPITAL
    open_positions = {}
    pending = {}
    equity_curve = []
    trades = []

    def horizon_for(bucket):
        return horizon_by_bucket.get(bucket, FLAT_HORIZON)

    for day_idx, day in enumerate(calendar):
        # 1) exits on open positions
        for tk in list(open_positions):
            pos = open_positions[tk]
            row = tickers_data[tk].get(day)
            if row is None:
                continue
            o, h, l, c = row
            days_held = day_idx - pos["entry_day_idx"]
            exit_price, reason = None, None
            if l <= pos["stop_price"]:
                exit_price, reason = pos["stop_price"], "stop"
            elif h >= pos["tp_price"]:
                exit_price, reason = pos["tp_price"], "tp_gap_filled"
            elif days_held >= horizon_for(pos["bucket"]):
                exit_price, reason = c, "time_exit"
            if exit_price is not None:
                exit_price *= (1 - slip)
                pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                pnl_dollar = pos["shares"] * (exit_price - pos["entry"]) - 2 * commission_per_trade
                cash += pos["shares"] * exit_price - commission_per_trade
                trades.append({
                    "ticker": tk, "gap_date": pos["gap_date"],
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "bucket": pos["bucket"], "gap_pct": pos["gap_pct"],
                    "entry_price": round(pos["entry"], 4),
                    "exit_price": round(exit_price, 4),
                    "exit_reason": reason,
                    "contracts_shares": round(pos["shares"], 4),
                    "position_dollars": round(pos["position_dollars"], 2),
                    "pnl_pct": round(pnl_pct, 3),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "days_held": days_held,
                })
                del open_positions[tk]

        # 2) update pending watches / find new gaps -> today's fill candidates
        fill_candidates = []

        for tk in sorted(tickers_data):
            if tk in open_positions or tk in pending:
                continue
            df = tickers_data[tk]
            tr = df.get(day)
            pr = df.get_prior_close(day)
            if tr is None or pr is None or pr <= 0:
                continue
            o, h, l, c = tr
            gap_pct = (o - pr) / pr * 100
            abs_gap = -gap_pct
            if gap_pct >= -1.0 or not bucket_matches(abs_gap, scope_lo_hi):
                continue
            pending[tk] = {"gap_open": o, "prior_close": pr,
                          "start_day_idx": day_idx, "gap_pct": gap_pct,
                          "gap_date": day, "confirmed_day_idx": None,
                          "bucket": bucket_name_for(abs_gap)}

        for tk in list(pending):
            w = pending[tk]
            if day_idx <= w["start_day_idx"]:
                continue
            if day_idx - w["start_day_idx"] > horizon_for(w["bucket"]):
                del pending[tk]
                continue
            if tk in open_positions:
                del pending[tk]
                continue
            row = tickers_data[tk].get(day)
            if row is None:
                continue
            o, h, l, c = row

            if entry_style == "wick":
                if h >= w["gap_open"]:
                    abs_gap = -w["gap_pct"]
                    fill_candidates.append((tk, w["gap_open"], w["prior_close"],
                                            bucket_name_for(abs_gap),
                                            stop_fn(abs_gap),
                                            w["gap_pct"], w["gap_date"]))
                    del pending[tk]

            elif entry_style == "close_confirm":
                if w["confirmed_day_idx"] is None:
                    if c >= w["gap_open"]:
                        w["confirmed_day_idx"] = day_idx
                elif day_idx == w["confirmed_day_idx"] + 1:
                    abs_gap = -w["gap_pct"]
                    fill_candidates.append((tk, o, w["prior_close"],
                                            bucket_name_for(abs_gap),
                                            stop_fn(abs_gap),
                                            w["gap_pct"], w["gap_date"]))
                    del pending[tk]

        # 3) fill candidates into open slots
        if priority == "gap_desc":
            fill_candidates.sort(key=lambda x: (-abs(x[5]), x[0]))
        else:
            fill_candidates.sort(key=lambda x: x[0])
        free = max_slots - len(open_positions)
        for tk, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date in fill_candidates[:max(free, 0)]:
            mtm = sum(
                (tickers_data[p].get(day) or (0, 0, 0, pp["entry"]))[3] * pp["shares"]
                for p, pp in open_positions.items()
            )
            equity_now = cash + mtm
            size_dollars = equity_now / max_slots
            if size_dollars > cash:
                continue
            entry_price *= (1 + slip)
            shares = size_dollars / entry_price
            cash -= size_dollars + commission_per_trade
            open_positions[tk] = {
                "entry": entry_price, "stop_price": entry_price * (1 - stop_w / 100),
                "tp_price": tp_price, "entry_day_idx": day_idx,
                "shares": shares, "bucket": bucket, "gap_pct": round(gap_pct, 3),
                "gap_date": gap_date, "entry_date": day,
                "position_dollars": size_dollars,
            }

        # 4) mark equity
        mtm = sum(
            (tickers_data[tk].get(day) or (0, 0, 0, pos["entry"]))[3] * pos["shares"]
            for tk, pos in open_positions.items()
        )
        equity_curve.append(cash + mtm)

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    tdf = pd.DataFrame(trades)
    return {
        "n_trades": len(tdf),
        "win_rate_pct": 100 * (tdf["pnl_pct"] > 0).mean() if len(tdf) else None,
        "avg_win_pct": tdf["pnl_pct"].mean() if len(tdf) else None,
        "total_pnl_dollar": tdf["pnl_dollar"].sum() if len(tdf) else 0.0,
        "max_drawdown_pct": dd.min() * 100,
        "total_return_pct": (eq.iloc[-1] / CAPITAL - 1) * 100,
        "final_equity": eq.iloc[-1],
        "trades_df": tdf,
        "equity_curve": eq,
    }


def load_universe():
    files = sorted(BARS_DIR.glob("*.parquet"))
    tickers_data = {}
    longest_dates = None
    for p in files:
        d = load_ticker(p)
        if d is None:
            continue
        tickers_data[p.stem] = TickerView(d)
        if longest_dates is None or len(d) > len(longest_dates):
            longest_dates = d["Date"]
    calendar = pd.DatetimeIndex(sorted(longest_dates))
    cutoff = calendar[-1] - pd.DateOffset(years=WINDOW_YEARS)
    calendar = calendar[calendar >= cutoff]
    return tickers_data, calendar


def summarize(label, r):
    ret_dd = (r["total_return_pct"] / abs(r["max_drawdown_pct"])
              if r["max_drawdown_pct"] else None)
    exit_reasons = (r["trades_df"]["exit_reason"].value_counts().to_dict()
                    if len(r["trades_df"]) else {})
    return {
        "arm": label,
        "n_trades": r["n_trades"],
        "win_rate_%": round(r["win_rate_pct"], 1) if r["win_rate_pct"] is not None else None,
        "total_return_%": round(r["total_return_pct"], 2),
        "max_drawdown_%": round(r["max_drawdown_pct"], 2),
        "ret_per_maxdd": round(ret_dd, 3) if ret_dd else None,
        "final_equity_$": round(r["final_equity"], 2),
        "n_stop": exit_reasons.get("stop", 0),
        "n_tp": exit_reasons.get("tp_gap_filled", 0),
        "n_time_exit": exit_reasons.get("time_exit", 0),
    }


def main():
    tickers_data, calendar = load_universe()
    print(f"{len(tickers_data)} tickers, {len(calendar)} trading days, "
          f"{calendar[0].date()} -> {calendar[-1].date()} ({WINDOW_YEARS}yr window)")

    scope = (2.0, np.inf)  # exclude 1-2%, per anchor
    common = dict(max_slots=MAX_SLOTS, priority="gap_desc",
                  commission_per_trade=0.0, slippage_bps=10.0)

    arms = [
        ("horizon_A_20_25_40_63", {"2-3%": 20, "3-5%": 25, "5-10%": 40, "10%+": 63}),
        ("horizon_B_30_35_50_63", {"2-3%": 30, "3-5%": 35, "5-10%": 50, "10%+": 63}),
    ]

    rows = []
    for label, hmap in arms:
        print(f"\nrunning {label} horizons={hmap} ...")
        r = run_sim_bucket_horizon(tickers_data, calendar, "wick", scope, hmap, **common)
        r["trades_df"].to_csv(RESULTS / f"bucket_timeout_sweep_{label}.csv", index=False)
        r["equity_curve"].to_csv(RESULTS / f"bucket_timeout_sweep_{label}_equity.csv")
        row = summarize(label, r)
        rows.append(row)
        print(f"  n={row['n_trades']}  win%={row['win_rate_%']}  "
              f"return={row['total_return_%']}%  maxDD={row['max_drawdown_%']}%  "
              f"ret/maxdd={row['ret_per_maxdd']}  "
              f"exits: stop={row['n_stop']} tp={row['n_tp']} time={row['n_time_exit']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "bucket_timeout_sweep_summary.csv", index=False)
    print("\n" + "=" * 110)
    print("PER-BUCKET EXIT TIMEOUT vs FLAT 63 -- anchor config, 2yr, friction on")
    print("baseline (flat 63): n=501 return=65.42% maxDD=-17.69% ret/maxdd=3.697")
    print("=" * 110)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
