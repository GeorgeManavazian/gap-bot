"""Intraday polling for the paper ledger (2026-09-08): between the EOD
runs, check the ALREADY-PENDING wick-watches and the open positions
against the session's running high/low, so a touch, stop or take-profit
gets written to the ledger within one poll interval of happening instead
of ~7 hours later at the after-close tick.

What this changes and what it does not
--------------------------------------
Changes: only WHEN an event is noticed and recorded. The price a fill is
recorded at (gap_open for an entry, stop_price / tp_price for an exit,
slippage applied the same way) is exactly what engine.step_one_day()
would have recorded from the settled daily candle -- engine.py already
checks the day's real high/low, never just the close.

Does not change (and the EOD run still does, once per session):
  * new-gap registration (needs the settled daily open vs prior close);
  * the same-day rule -- engine.py step 2b: a watch registered TODAY can
    never fill TODAY. Enforced here by construction: this module never
    registers a watch, and it skips any watch whose gap_date == today
    (only the EOD tick can create one with today's date, and it runs
    after this module has stopped for the day). Preserved on purpose;
    it matches the backtested rule exactly;
  * days_held / days_waited counters, HORIZON expiry, time exits at the
    close, equity snapshots.

The EOD run is unchanged and remains the backstop: if this poller never
runs (outage, token failure), the after-close tick still records every
event from the settled candle, exactly as before this module existed.
A position this poller filled earlier in the session is protected from
being exit-checked by that same evening's EOD run by engine.py's
entry_date == today guard (the backtest never checked a fill day's own
low/high against the fresh stop/tp).

Entry rule (owner ruling 2026-09-08, engine.py): resting-limit fill --
a watch fills only when the session's running low <= gap_open <= running
high -- and rest-top-N priority -- at each poll, only the `free`-slots
biggest-gap live watches have an order resting; a touch on any other
watch is a missed wick and consumes the watch. Same helpers as the EOD
engine (limit_touched, resting_orders), so the two can't drift.

Divergences from the daily-bar backtest that intraday timing CAN'T avoid
-- flagged, not hidden:
  1. Stop vs take-profit on the SAME day: the daily bar can't order them,
     so the backtest assumes stop-first (conservative). Intraday, whichever
     the poller observes first wins. Inside a single poll interval both
     may show as hit at once; that case keeps the backtest's stop-first
     order. Real orders would behave like the intraday version.
  2. The resting set is recomputed each poll: an exit at 10am frees a slot
     and the next-biggest watch gets an order from the next poll on. The
     daily engine counts the whole day's exits before choosing the day's
     resting set, so a watch the poller passed over at 09:35 could have
     been resting in the daily view. Real orders behave like the poller.
  3. Position sizing uses equity marked at the poll's last price rather
     than the close.

Data source: one batched Schwab get_quotes() call on the watched set
(open positions + pending watches, typically a few dozen names, never
the whole universe). Verified live 2026-09-08 on 50 names during a
session: quote.openPrice/highPrice/lowPrice are REGULAR-SESSION values
(21 of the 50 had a pre-market high above the regular high and the
quote field ignored it), and they matched the max/min of 1-minute
regular-session bars exactly on all 50. lastPrice is real-time. ADV for
the liquidity floor isn't in a quote, so it's pulled per touched ticker
from the daily endpoint only when a touch actually happens (rare).
"""
from __future__ import annotations

import pandas as pd

from live.config import HORIZON, MAX_SLOTS, stop_for_abs_gap, bucket_name_for
from live.engine import close_position, fill_slots, limit_touched, resting_orders
from live.schwab_data import throttle, fetch_universe_bars, bars_for_today

ET = "America/New_York"
QUOTE_CHUNK = 250  # 500 fit in one call live (2026-09-08); half that leaves margin


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET)


def watched_tickers(state: dict) -> list[str]:
    return sorted(set(state["open_positions"]) | set(state["pending"]))


def fetch_quotes(client, tickers: list[str]) -> dict:
    """{ticker: {"o","h","l","c","prior_close","trade_time","status"}} of the
    regular session so far, from get_quotes. "c" is the LAST price (the
    engine's mark), not a close. A ticker missing from the response, or
    with no regular-session print yet (highPrice 0/None before its first
    trade), is left out -- engine semantics for an absent bar: skip it
    this poll, never error. One failed chunk is logged and skipped."""
    out = {}
    for i in range(0, len(tickers), QUOTE_CHUNK):
        chunk = tickers[i:i + QUOTE_CHUNK]
        try:
            r = throttle(client.get_quotes, chunk)
            if r.status_code != 200:
                print(f"fetch_quotes: HTTP {r.status_code} on {len(chunk)} tickers, skipped this poll")
                continue
            body = r.json()
        except Exception as e:
            print(f"fetch_quotes: {e} on {len(chunk)} tickers, skipped this poll")
            continue
        bad = (body.get("errors") or {}).get("invalidSymbols")
        if bad:
            print(f"fetch_quotes: Schwab rejected symbols {bad}")
        for tk in chunk:
            q = (body.get(tk) or {}).get("quote") or {}
            o, h, l, c = q.get("openPrice"), q.get("highPrice"), q.get("lowPrice"), q.get("lastPrice")
            if not all(isinstance(v, (int, float)) and v > 0 for v in (o, h, l, c)):
                continue
            tt = (body[tk].get("regular") or {}).get("regularMarketTradeTime") or q.get("tradeTime")
            trade_time = (pd.to_datetime(tt, unit="ms", utc=True).tz_convert(ET)
                          if tt else None)
            out[tk] = {"o": float(o), "h": float(h), "l": float(l), "c": float(c),
                       "prior_close": q.get("closePrice"), "trade_time": trade_time,
                       "status": q.get("securityStatus")}
    return out


def session_date_from_quotes(quotes: dict, reference: str = "SPY"):
    """Today's session date, off the DATA (the reference ticker's last
    regular-session trade time in ET), not the wall clock -- same doctrine
    as schwab_data.latest_session_date. On a holiday the reference's last
    trade is a prior session, so the caller's `today == wall-clock ET
    date` check fails and the poll no-ops. None if the reference is
    missing."""
    ref = quotes.get(reference)
    if ref is None or ref["trade_time"] is None:
        return None
    return ref["trade_time"].date()


def drop_stale_quotes(quotes: dict, session) -> tuple[dict, list[str]]:
    """Keep only quotes whose last regular trade is dated `session` (ET).
    A name with no regular print yet today (illiquid open, halt, bad feed)
    can carry the PRIOR session's open/high/low in its quote; checking that
    range against today's limits and stops would book phantom touches. A
    quote with no trade time can't be dated either, so it is dropped too.
    Dropped names are skipped this poll, exactly like an absent bar --
    the evening run still records anything real from the settled candle.
    Returns (fresh quotes, sorted dropped tickers)."""
    fresh, dropped = {}, []
    for tk, q in quotes.items():
        tt = q.get("trade_time")
        if tt is not None and tt.date() == session:
            fresh[tk] = q
        else:
            dropped.append(tk)
    return fresh, sorted(dropped)


# Longest normal gap between EOD runs is a holiday long weekend (4 calendar
# days); anything past this means the evening run missed sessions (outage,
# or the bot was paused), so counters and watch ages are stale.
MAX_STATE_AGE_DAYS = 5


def state_is_stale(last_run_date: str | None, today: str, max_age_days: int = MAX_STATE_AGE_DAYS) -> bool:
    """True when the ledger's last EOD run is more than `max_age_days` before
    `today`. The EOD engine steps ONE day per run and never backfills, so
    after a long gap the poller would fill limits and check stops against
    watches/positions whose days_waited / days_held and ranges are weeks
    old. The poller refuses; the evening run is unchanged. A ledger that
    has never run (None) is not stale -- it has nothing to be stale about."""
    if last_run_date is None:
        return False
    return (pd.Timestamp(today) - pd.Timestamp(last_run_date)).days > max_age_days


def adv_for(client, tk: str):
    """Trailing 20-day average dollar volume for one ticker as of today,
    today's own (partial) volume excluded -- identical definition to the
    EOD path (schwab_data.bars_for_today), computed from the same daily
    endpoint. None on any failure (unfloored, like a newly-listed name)."""
    try:
        frames = fetch_universe_bars(client, [tk])
        return bars_for_today(frames).get(tk, {}).get("adv")
    except Exception as e:
        print(f"adv_for({tk}): {e} -- treating as unfloored")
        return None


def check_watches(state: dict, today: str, snap: dict, filler, chain_provider=None,
                  adv_lookup=None, stamp: str | None = None) -> dict:
    """One intraday check of `state` against `snap` (running session bars,
    same row schema as engine bars: o/h/l/c(=last)/prior_close, adv
    filled in lazily via `adv_lookup(tk)` for a touched ticker only).
    Mutates state in place, same as step_one_day. Returns {"trades",
    "filled", "equity", "n_open", "n_pending"}.

    Invariants, in order:
      * a position with entry_date == today is never checked (backtest:
        entry day's own range isn't checked against the new stop/tp);
      * a watch with gap_date == today is never checked (the same-day
        rule, engine.py step 2b -- preserved exactly);
      * a watch that the EOD run would expire today (days_waited + 1 >
        HORIZON) is never checked -- the EOD run expires it, unchecked,
        just as the backtest does;
      * counters are NOT advanced here; `days_held` on an intraday exit
        is reported as pos["days_held"] + 1, the value the EOD run would
        have used for today.
    """
    cash = state["cash"]
    open_positions = state["open_positions"]
    pending = state["pending"]
    trades = []
    extra = {"entry_time": stamp} if stamp else {}

    # 1) exits on open positions -- stop first, then tp, same order as the
    # engine when both show inside one poll interval. No time exits here.
    for tk in list(open_positions):
        pos = open_positions[tk]
        if pos.get("entry_date") == today:
            continue
        row = snap.get(tk)
        if row is None:
            continue
        if row["l"] <= pos["stop_price"]:
            exit_price, reason = pos["stop_price"], "stop"
        elif row["h"] >= pos["tp_price"]:
            exit_price, reason = pos["tp_price"], "tp_gap_filled"
        else:
            continue
        proceeds, trade = close_position(pos, tk, exit_price, reason, pos["days_held"] + 1,
                                         today, filler, chain_provider)
        if stamp:
            trade["exit_time"] = stamp
        cash += proceeds
        trades.append(trade)
        del open_positions[tk]

    # 2) touches on pending watches from PRIOR sessions only, engine rule:
    # only the `free` biggest-gap live watches have an order resting right
    # now; a resting-limit touch (running l <= gap_open <= running h) on
    # one of them is a candidate. A touch on any live watch, resting or
    # not, consumes it -- as does a fill refused later (ADV cap, cash,
    # filler) -- identical to engine step 2b/2c/3.
    live = []
    for tk in list(pending):
        w = pending[tk]
        if w["gap_date"] == today:
            continue  # same-day rule: registered today, can't fill today
        if w["days_waited"] + 1 > HORIZON:
            continue  # EOD run will expire it unchecked
        if tk in open_positions:
            continue
        live.append(tk)
    resting = resting_orders(live, pending, MAX_SLOTS - len(open_positions))
    candidates = []
    for tk in live:
        row = snap.get(tk)
        if row is None:
            continue
        w = pending[tk]
        if not limit_touched(row, w["gap_open"]):
            continue
        if tk in resting:
            abs_gap = -w["gap_pct"]
            candidates.append((tk, w["gap_open"], w["prior_close"], bucket_name_for(abs_gap),
                               stop_for_abs_gap(abs_gap), w["gap_pct"], w["gap_date"]))
        del pending[tk]

    # 3) fill -- the engine's own helper, with today's ADV looked up only
    # for the (rare) touched names so the liquidity floor is applied
    # exactly as at EOD.
    if candidates and adv_lookup is not None:
        for c in candidates:
            snap[c[0]] = {**snap[c[0]], "adv": adv_lookup(c[0])}
    before = set(open_positions)
    cash = fill_slots(candidates, cash, open_positions, today, snap, filler, chain_provider,
                      extra_fields=extra)
    filled = [tk for tk in open_positions if tk not in before]

    # 4) mark at the poll's last price -- same field the dashboard reads.
    mtm = 0.0
    for tk, pos in open_positions.items():
        last = snap.get(tk, {}).get("c", pos.get("last", pos["entry"]))
        pos["last"] = last
        mtm += filler.mark(pos, last, chain_provider)

    state["cash"] = cash
    return {"trades": trades, "filled": filled, "equity": cash + mtm,
            "n_open": len(open_positions), "n_pending": len(pending)}
