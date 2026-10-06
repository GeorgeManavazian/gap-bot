"""One live trading day, ported from scripts/slot_and_priority_sweep.py's
run_sim() day-step, driven by one fresh day of bars instead of a
precomputed backtest calendar -- with the ENTRY mechanics replaced on
2026-09-08 (owner ruling) by the two rules a real order can actually
execute:

  * resting-limit fill: a watch fills at gap_open only on a day the price
    traded through it (low <= gap_open <= high); otherwise it keeps
    resting until HORIZON. The original rule filled on high >= gap_open
    alone and booked 276 of its 508 anchor fills at a price the stock had
    opened above -- see scripts/priority_lookahead_check.py;
  * rest-top-N priority: with `free` open slots, only the `free`
    biggest-gap live watches have an order resting today; a touch on any
    other watch is a missed wick and consumes the watch. Decided from
    what's known at the start of the day, never from the day's full set
    of touches (the original biggest-gap-first sorted the whole day's
    touches, which only the close knows).

Anchor for this rule (2yr held-out, 2026-09-08): 174 trades, +11.45%,
maxDD -14.59% -- scripts/verify_live_engine.py must reproduce it.

Three instrument variants (stock/call/spread, live/fillers.py) share this
exact function: the signal (which tickers wick-fill, at what stock price,
what bucket/stop/tp) and the exit TRIGGER (stop/tp/timeout, always checked
against the stock price) are identical across variants -- only what you buy
against the signal, and what it's worth, is delegated to `filler`. A
variant's filler can refuse a candidate (e.g. no listed option) via
price_entry() returning None; that candidate is simply skipped for THIS
variant's ledger, other variants are unaffected -- which is why each
variant runs its own independent step_one_day() call with its own state,
not a shared position list.

`bars`: {ticker: {"o","h","l","c","prior_close","adv"}} for TODAY only.
`adv` is the caller's trailing 20-day average dollar volume, look-back
only (today's own volume excluded) -- same definition as the backtest's
TickerView.get_adv(). A ticker absent from `bars` (failed pull, holiday
mismatch, delisted) is treated exactly like a missing bar in the
backtest: exits/fills involving it are skipped for the day, not errored.

State mutated in place and returned; the caller (run_daily.py) persists
it. `days_held`/`days_waited` are integer counters incremented once per
call -- exact for a bot that runs once per real trading day, which
run_daily.py enforces via the last_run_date guard.

Intraday polling (live/intraday.py, 2026-09-08) reuses the two helpers
below -- close_position() and fill_slots() -- so the exit economics and
the slot-filling rules have exactly one implementation. It does NOT call
step_one_day(); it only checks ALREADY-PENDING watches and open positions
against the running session high/low between the EOD runs. Everything
that advances the calendar (counters, new-gap registration, time exits)
stays here, once per session."""
from __future__ import annotations

from live.config import (
    MAX_SLOTS, HORIZON, SLIPPAGE_BPS, MAX_PCT_OF_ADV,
    EXCLUDE_BELOW, stop_for_abs_gap, bucket_name_for,
    PROFILE, RAMP_BASE, RAMP_MAX, RAMP_G, SIZE_DIV, ORDER_FLOOR,
)

SLIP = SLIPPAGE_BPS / 10_000.0


def slot_limit(open_positions: dict, pending: dict) -> int:
    """How many slots the book may use right now. PROFILE "anchor": the fixed
    MAX_SLOTS (read at call time, so backtests that patch engine.MAX_SLOTS
    still work). PROFILE "ramp": RAMP_BASE, plus one for each waiting watch
    with gap <= -RAMP_G% that is not already held, capped at RAMP_MAX. Pure
    function of state; step_one_day calls it on the state it was handed
    (before that day's exits and new-gap registration), the intraday poller
    on the state at poll time -- both through this one accessor."""
    if PROFILE != "ramp":
        return MAX_SLOTS
    big = sum(1 for tk, w in pending.items()
              if tk not in open_positions and w["gap_pct"] <= -RAMP_G)
    return int(min(RAMP_MAX, max(RAMP_BASE, len(open_positions) + big)))

# Position fields that get their own named column in a trade record; every
# other key on the position (filler extras like option symbols/strikes,
# entry_time from an intraday fill) is passed through as-is.
_TRADE_NAMED = frozenset(("entry", "stop_price", "tp_price", "days_held", "units", "bucket",
                          "gap_pct", "gap_date", "entry_date", "position_dollars",
                          "cash_cost", "last", "ticker"))


def limit_touched(row: dict, gap_open: float) -> bool:
    """Would a resting limit buy at gap_open have filled on this bar? Only
    if the price actually traded through the level."""
    return row["l"] <= gap_open <= row["h"]


def resting_orders(live: list, pending: dict, free: int) -> set:
    """The watches that have an order resting today: the `free` biggest
    gaps among `live` (ticker as tie-break, same as the backtest's sort).
    Known from state alone -- no look-ahead into which will touch.
    ORDER_FLOOR (ramp profile): watches with |gap| below it get no order, but
    stay registered -- see live/config.py for why that must not move to
    registration."""
    if free <= 0:
        return set()
    if ORDER_FLOOR is not None:
        live = [t for t in live if pending[t]["gap_pct"] <= -ORDER_FLOOR]
    return set(sorted(live, key=lambda t: (pending[t]["gap_pct"], t))[:free])


def close_position(pos: dict, tk: str, exit_price: float, reason: str, days_held: int,
                   today: str, filler, chain_provider=None) -> tuple[float, dict]:
    """Settle one exit: slippage, filler proceeds, P&L, trade record.
    Returns (proceeds, trade). Does NOT touch cash or open_positions --
    the caller does, so the same helper serves both the EOD day-step and
    the intraday poller."""
    if reason != "time_exit_no_data":
        exit_price *= (1 - SLIP)  # slippage always against you: sell lower
        # (no-data close isn't a real fill -- nothing to slip against)
    proceeds = filler.close_value(pos, exit_price, reason, chain_provider)
    pnl_dollar = proceeds - pos["cash_cost"]
    pnl_pct = pnl_dollar / pos["cash_cost"] * 100 if pos["cash_cost"] else 0.0
    trade = {
        "ticker": tk, "gap_date": pos["gap_date"], "entry_date": pos["entry_date"],
        "exit_date": today, "bucket": pos["bucket"], "gap_pct": pos["gap_pct"],
        "entry_price": round(pos["entry"], 4), "exit_price": round(exit_price, 4),
        "exit_reason": reason, "units": round(pos["units"], 4),
        "position_dollars": round(pos["position_dollars"], 2),
        "cash_cost": round(pos["cash_cost"], 2), "proceeds": round(proceeds, 2),
        "pnl_pct": round(pnl_pct, 3), "pnl_dollar": round(pnl_dollar, 2),
        "days_held": days_held, **{k: v for k, v in pos.items() if k not in _TRADE_NAMED},
    }
    return proceeds, trade


def fill_slots(fill_candidates: list, cash: float, open_positions: dict, today: str,
               bars: dict, filler, chain_provider=None, extra_fields: dict | None = None,
               max_slots: int | None = None) -> float:
    """Step 3 of the day: fill candidates into open slots, equal-weight.
    Mutates open_positions, returns the new cash. A candidate is
    (tk, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date). Sorted
    biggest-gap-first for determinism; since 2026-09-08 the caller only
    passes touches on watches that had a resting order (never more than
    `free`), so the sort no longer selects, it only orders.

    A filler can refuse a candidate (price_entry -> None: no listed
    contract, no quote, priced as a credit/free -- data looks bad) --
    that candidate is dropped for THIS variant only, without consuming a
    slot, and does not fall back to the next candidate taking its place
    (matching the backtest's own doctrine: a signal either fills as
    scoped or it doesn't, no substitution). `extra_fields` (e.g. an
    intraday entry_time) is stamped onto every position opened here.
    `max_slots` (None = MAX_SLOTS) is the day's slot limit from slot_limit();
    position size is equity / SIZE_DIV when the profile sets one (ramp: 9),
    else equity / MAX_SLOTS exactly as before."""
    fill_candidates.sort(key=lambda x: (-abs(x[5]), x[0]))
    free = (MAX_SLOTS if max_slots is None else max_slots) - len(open_positions)
    filled = 0
    for tk, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date in fill_candidates:
        if filled >= free:
            break
        mtm = sum(filler.mark(pp, bars.get(p, {}).get("c", pp["entry"]), chain_provider)
                  for p, pp in open_positions.items())
        equity_now = cash + mtm
        size_dollars = equity_now / (SIZE_DIV or MAX_SLOTS)
        if MAX_PCT_OF_ADV is not None:
            adv = bars.get(tk, {}).get("adv")
            if adv is not None and size_dollars > MAX_PCT_OF_ADV * adv:
                continue
        entry_price_slipped = entry_price * (1 + SLIP)  # slippage always against you: buy higher
        quote = filler.price_entry(tk, entry_price_slipped, size_dollars, chain_provider, tp_price=tp_price)
        if quote is None:
            continue  # filler couldn't price this today -- skip, slot stays open
        if quote.cash_cost > cash:
            continue
        cash -= quote.cash_cost
        open_positions[tk] = {
            "ticker": tk, "entry": entry_price_slipped,
            "stop_price": entry_price_slipped * (1 - stop_w / 100), "tp_price": tp_price,
            "days_held": 0, "units": quote.units, "bucket": bucket,
            "gap_pct": round(gap_pct, 3), "gap_date": gap_date, "entry_date": today,
            "position_dollars": size_dollars, "cash_cost": quote.cash_cost,
            **quote.extra, **(extra_fields or {}),
        }
        filled += 1
    return cash


def step_one_day(state: dict, today: str, bars: dict, filler, chain_provider=None) -> dict:
    cash = state["cash"]
    open_positions = state["open_positions"]
    pending = state["pending"]
    day_trades = []
    # the day's slot limit comes from the state as handed in (before exits
    # and new-gap registration) -- the same point the 10yr backtest read it
    slots_today = slot_limit(open_positions, pending)

    # 1) exits on open positions
    for tk in list(open_positions):
        pos = open_positions[tk]
        # A position opened TODAY is never exit-checked today. In the pure
        # daily flow this can't happen (fills are step 3, after this loop),
        # so this guard is a no-op there; it exists for a position the
        # intraday poller already filled earlier in this same session --
        # the backtest never checked a fill day's own low/high against the
        # new stop/tp, and days_held stays 0 until the NEXT session, so
        # neither may the EOD backstop.
        if pos.get("entry_date") == today:
            continue
        # days_held advances unconditionally, BEFORE the bars lookup -- same
        # pattern as pending's days_waited below. Otherwise a ticker that
        # vanishes from bars for good (delisted/halted post-entry) never
        # accumulates days_held again and never reaches HORIZON: it sits
        # open forever, permanently occupying a slot (found + repro'd
        # 2026-09-07).
        days_held = pos["days_held"] + 1
        row = bars.get(tk)
        exit_price, reason = None, None
        if row is None:
            # No price to check stop/tp against, but the clock still ran
            # out -- close it at cost (same fallback already used for
            # equity marking elsewhere in this file: `.get("c",
            # pos["entry"])`) and tag it distinctly so the trade log shows
            # this wasn't a normal, price-based timeout.
            if days_held >= HORIZON:
                exit_price, reason = pos["entry"], "time_exit_no_data"
            else:
                pos["days_held"] = days_held
                continue
        else:
            if row["l"] <= pos["stop_price"]:
                exit_price, reason = pos["stop_price"], "stop"
            elif row["h"] >= pos["tp_price"]:
                exit_price, reason = pos["tp_price"], "tp_gap_filled"
            elif days_held >= HORIZON:
                exit_price, reason = row["c"], "time_exit"

            if exit_price is None:
                pos["days_held"] = days_held
                continue

        proceeds, trade = close_position(pos, tk, exit_price, reason, days_held, today,
                                         filler, chain_provider)
        cash += proceeds
        day_trades.append(trade)
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
    live = []
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
        live.append(tk)

    # 2c) which of them has an order resting today (rest-top-N by gap,
    # `free` counted after this morning's exits), and which of those the
    # price actually traded through (resting-limit fill). A touch on a
    # watch WITHOUT an order is a missed wick: the watch is consumed,
    # nothing is bought -- the price got there and we weren't in line.
    free = slots_today - len(open_positions)
    resting = resting_orders(live, pending, free)
    fill_candidates = []
    for tk in live:
        w = pending[tk]
        if w.get("consumed_on"):
            # The intraday poller already consumed this watch today (touch
            # without an order resting, or a refused fill). It stayed in
            # `live` so the resting set above matches the daily view; it
            # is never a candidate, and goes the way a consumed watch
            # goes here. Never set by the backtest or the catch-up replay.
            del pending[tk]
            continue
        row = bars.get(tk)
        if row is None:
            continue
        if not limit_touched(row, w["gap_open"]):
            continue
        if tk in resting:
            abs_gap = -w["gap_pct"]
            fill_candidates.append((tk, w["gap_open"], w["prior_close"],
                                    bucket_name_for(abs_gap), stop_for_abs_gap(abs_gap),
                                    w["gap_pct"], w["gap_date"]))
        del pending[tk]

    # 3) fill the touched resting orders into the open slots, equal-weight
    # (see fill_slots for the refusal doctrine).
    # (max_slots only passed under the ramp profile, so the anchor path -- and
    # backtests that swap in their own fill_slots -- keep the old call shape)
    fill_kw = {"max_slots": slots_today} if PROFILE == "ramp" else {}
    cash = fill_slots(fill_candidates, cash, open_positions, today, bars, filler, chain_provider, **fill_kw)

    # 4) mark equity -- also persists each position's last-close price so a
    # dashboard reading state.json (never re-fetches quotes itself) can show
    # a mark and per-position P&L instead of only the entry price.
    mtm = 0.0
    for tk, pos in open_positions.items():
        last = bars.get(tk, {}).get("c", pos["entry"])
        pos["last"] = last
        mtm += filler.mark(pos, last, chain_provider)
    equity = cash + mtm

    state["cash"] = cash
    state["last_run_date"] = today
    return {"state": state, "trades": day_trades, "equity": equity,
            "n_open": len(open_positions), "n_pending": len(pending)}
