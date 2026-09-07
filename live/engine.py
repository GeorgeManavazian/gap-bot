"""One live trading day, ported from scripts/slot_and_priority_sweep.py's
run_sim() day-step -- same rules, driven by one fresh day of bars instead
of a precomputed backtest calendar. Wick-entry only (the only style this
bot trades live); gap_desc priority only (the config's own choice, not a
knob here).

`bars`: {ticker: {"o","h","l","c","prior_close","adv"}} for TODAY only.
`adv` is the caller's trailing 20-day average dollar volume, look-back
only (today's own volume excluded) -- same definition as the backtest's
TickerView.get_adv(). A ticker absent from `bars` (failed pull, holiday
mismatch, delisted) is treated exactly like a missing bar in the
backtest: exits/fills involving it are skipped for the day, not errored.

State mutated in place and returned; the caller (run_daily.py) persists
it. `days_held`/`days_waited` are integer counters incremented once per
call -- exact for a bot that runs once per real trading day, which
run_daily.py enforces via the last_run_date guard."""
from __future__ import annotations

from live.config import (
    MAX_SLOTS, HORIZON, COMMISSION_PER_TRADE, SLIPPAGE_BPS, MAX_PCT_OF_ADV,
    EXCLUDE_BELOW, stop_for_abs_gap, bucket_name_for,
)

SLIP = SLIPPAGE_BPS / 10_000.0


def step_one_day(state: dict, today: str, bars: dict) -> dict:
    cash = state["cash"]
    open_positions = state["open_positions"]
    pending = state["pending"]
    day_trades = []

    # 1) exits on open positions
    for tk in list(open_positions):
        pos = open_positions[tk]
        row = bars.get(tk)
        if row is None:
            continue
        days_held = pos["days_held"] + 1
        exit_price, reason = None, None
        if row["l"] <= pos["stop_price"]:
            exit_price, reason = pos["stop_price"], "stop"
        elif row["h"] >= pos["tp_price"]:
            exit_price, reason = pos["tp_price"], "tp_gap_filled"
        elif days_held >= HORIZON:
            exit_price, reason = row["c"], "time_exit"

        if exit_price is None:
            pos["days_held"] = days_held
            continue

        exit_price *= (1 - SLIP)  # slippage always against you: sell lower
        pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
        pnl_dollar = pos["shares"] * (exit_price - pos["entry"]) - 2 * COMMISSION_PER_TRADE
        cash += pos["shares"] * exit_price - COMMISSION_PER_TRADE
        day_trades.append({
            "ticker": tk, "gap_date": pos["gap_date"], "entry_date": pos["entry_date"],
            "exit_date": today, "bucket": pos["bucket"], "gap_pct": pos["gap_pct"],
            "entry_price": round(pos["entry"], 4), "exit_price": round(exit_price, 4),
            "exit_reason": reason, "shares": round(pos["shares"], 4),
            "position_dollars": round(pos["position_dollars"], 2),
            "pnl_pct": round(pnl_pct, 3), "pnl_dollar": round(pnl_dollar, 2),
            "days_held": days_held,
        })
        del open_positions[tk]

    # 2a) register brand-new gaps into pending FIRST, same order as the
    # backtest -- skips a ticker already pending (bug found 2026-09-07: a
    # ticker mid-watch that gaps AGAIN today must NOT get a fresh entry;
    # only its original wick counts until it touches, times out, or fills.
    # Registering before advancing is what makes that guard effective --
    # advancing first would let a same-day touch-then-reregister slip a
    # newer, bigger gap in ahead of tickers that have been waiting longer).
    for tk, row in bars.items():
        if tk in open_positions or tk in pending:
            continue
        pr = row.get("prior_close")
        if pr is None or pr <= 0:
            continue
        gap_pct = (row["o"] - pr) / pr * 100
        abs_gap = -gap_pct
        if gap_pct >= -1.0 or abs_gap < EXCLUDE_BELOW:
            continue
        pending[tk] = {"gap_open": row["o"], "prior_close": pr, "gap_pct": gap_pct,
                       "gap_date": today, "days_waited": 0}

    # 2b) advance pending watches -- skip anything registered TODAY (2a,
    # above), a touch can't be checked the same day it starts.
    fill_candidates = []
    for tk in list(pending):
        w = pending[tk]
        if w["gap_date"] == today:
            continue
        w["days_waited"] += 1
        if w["days_waited"] > HORIZON:
            del pending[tk]
            continue
        if tk in open_positions:  # defensive parity with the backtest; can't
            del pending[tk]        # actually happen given this ordering
            continue
        row = bars.get(tk)
        if row is None:
            continue
        if row["h"] >= w["gap_open"]:
            abs_gap = -w["gap_pct"]
            fill_candidates.append((tk, w["gap_open"], w["prior_close"],
                                    bucket_name_for(abs_gap), stop_for_abs_gap(abs_gap),
                                    w["gap_pct"], w["gap_date"]))
            del pending[tk]

    # 3) fill candidates into open slots, biggest-gap-first, equal-weight
    fill_candidates.sort(key=lambda x: (-abs(x[5]), x[0]))
    free = MAX_SLOTS - len(open_positions)
    filled = 0
    for tk, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date in fill_candidates:
        if filled >= free:
            break
        mtm = sum((bars.get(p, {}).get("c", pp["entry"])) * pp["shares"]
                  for p, pp in open_positions.items())
        equity_now = cash + mtm
        size_dollars = equity_now / MAX_SLOTS
        if size_dollars > cash:
            continue
        if MAX_PCT_OF_ADV is not None:
            adv = bars.get(tk, {}).get("adv")
            if adv is not None and size_dollars > MAX_PCT_OF_ADV * adv:
                continue
        entry_price *= (1 + SLIP)  # slippage always against you: buy higher
        shares = size_dollars / entry_price
        cash -= size_dollars + COMMISSION_PER_TRADE
        open_positions[tk] = {
            "entry": entry_price, "stop_price": entry_price * (1 - stop_w / 100),
            "tp_price": tp_price, "days_held": 0, "shares": shares, "bucket": bucket,
            "gap_pct": round(gap_pct, 3), "gap_date": gap_date, "entry_date": today,
            "position_dollars": size_dollars,
        }
        filled += 1

    # 4) mark equity
    mtm = sum((bars.get(tk, {}).get("c", pos["entry"])) * pos["shares"]
              for tk, pos in open_positions.items())
    equity = cash + mtm

    state["cash"] = cash
    state["last_run_date"] = today
    return {"state": state, "trades": day_trades, "equity": equity,
            "n_open": len(open_positions), "n_pending": len(pending)}
