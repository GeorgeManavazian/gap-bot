"""Checks for the intraday poller (live/intraday.py), in two parts.

Part 1 -- synthetic scenarios, no network. Each one asserts an invariant
the poller must hold against engine.step_one_day():
  a. same-day rule: a watch registered today is never checked intraday;
  b. a watch from a prior session that touches intraday is filled at the
     same price/size the EOD engine would have used, and the EOD run that
     evening then leaves the new position untouched (entry-day guard);
  c. an open position's stop / tp is recorded intraday at the engine's
     price, and the EOD run afterwards does not double-count it;
  d. a watch the EOD run would expire today is not checked intraday;
  e. a touch that the ADV floor refuses still consumes the watch (as at
     EOD), and stop-before-tp ordering holds when both show in one poll;
  f. the EOD backstop: an intraday poll that never happened changes
     nothing -- day-step results with and without a prior no-event poll
     are byte-identical;
  g. an intraday poll that already recorded events, followed by the EOD
     step on the same session, reaches the same end state (positions,
     pending, cash) as the EOD step alone would have from the settled
     candle -- for the case where the events don't depend on order; and a
     watch consumed intraday is not re-registered by the EOD step when
     the same ticker gaps again that day (tagged, not deleted);
  h. the 2026-09-08 entry rule: a day whose whole range sits above
     gap_open is NOT a fill (the watch keeps resting); with one free slot
     and two watches touching, only the bigger gap had an order resting
     and fills, the smaller is consumed unfilled; a resting-less watch
     that does not touch keeps resting;
  i. quote freshness + ledger age: a quote not dated the session (or with
     no trade time) is dropped, a ledger whose last EOD run is not the
     previous NYSE session is refused (never backfilled -- one missed
     evening run, or a pause);
  j. fetch_quotes against a fake Schwab client: parsing, bad rows left
     out, a failed chunk skipped, rejected symbols tolerated;
  k. run_intraday end to end on a temp state dir with a fake client:
     fills and persists (ADV resolved from the daily endpoint; a failed
     ADV lookup leaves the touch for the EOD run), --dry-run persists
     nothing, stale quote / stale ledger / holiday are no-ops, missing
     reference quote or a raising variant exit non-zero, EOD-closed
     variant untouched, heartbeat written; the module never reaches an
     order endpoint;
  l. scripts/gap_bot_tick.sh window gating (fake DOW/HM, stub python):
     09:31-16:00 ET intraday every minute, 17:00-23:30 EOD on minutes
     that are a multiple of 5, weekend/off-hours/GAPBOT_INTRADAY=0
     nothing, a failing run logs FAILED and exits 1, a no-event poll
     after the session's first logs nothing;
  m. an ADV lookup failure intraday leaves the touched watch for the EOD
     run (not filled, not consumed) instead of filling it unfloored.
  The 2026-10-06 blockers also have pytest coverage:
  scripts/test_intraday_blockers.py (staleness refusal in run_daily, tick
  lock, --variants stock, idempotent appends).

Part 3 (inside Part 2's run) -- the replay must actually trade: zero
touch sessions means the bars did not load (see the Date-dtype note
below) and is reported as a FAILURE, not a pass.

Part 2 -- replays the 2yr anchor window through the engine and counts the
sessions where more live watches touched (resting-limit sense) than
there were free slots -- i.e. how often the rest-top-N choice actually
bit. Reported as a number, not hidden.

Run: .venv-live/bin/python scripts/verify_intraday.py
     GAPBOT_PROFILE=ramp .venv-live/bin/python scripts/verify_intraday.py

PROFILE NOTE (2026-10-06): every expectation below is derived from
live/config.py / engine.slot_limit at run time (slot count, size divisor,
stop width), never a literal 20 or a bucket stop, so the same script is
the intraday-vs-EOD parity check under both GAPBOT_PROFILE values. The
banner printed first says which one ran.

ENV NOTE (found 2026-10-05): the cached parquets store Date as
datetime64[ms]; where np.datetime64(Timestamp) yields microseconds the
TickerView lookups silently return no bars. Part 2 then sees zero touches
and this script FAILS loudly instead of passing on nothing. Workaround
without touching shared loader semantics: cast Date to datetime64[us] in a
wrapper around slot_and_priority_sweep.load_ticker before running.
"""
from __future__ import annotations
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from live.config import CAPITAL, HORIZON, MAX_SLOTS, PROFILE, PROFILE_BANNER, SIZE_DIV, VARIANTS, stop_for_abs_gap
from live.engine import step_one_day, slot_limit, SLIP
from live.fillers import make_filler
from live.intraday import (
    check_watches, drop_stale_quotes, state_is_stale, previous_session, fetch_quotes, ET,
)

FILLER = make_filler("stock", VARIANTS["stock"])
D0, D1, D2 = "2026-09-01", "2026-09-02", "2026-09-03"
ADV_BIG = 1e12


def bar(o, h, l, c, prior_close=None, adv=ADV_BIG):
    return {"o": o, "h": h, "l": l, "c": c, "prior_close": prior_close, "adv": adv}


def fresh():
    return {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}


def register(state, tk, gap_open, prior_close, gap_date, days_waited=0):
    state["pending"][tk] = {"gap_open": gap_open, "prior_close": prior_close,
                            "gap_pct": (gap_open - prior_close) / prior_close * 100,
                            "gap_date": gap_date, "days_waited": days_waited}


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    return bool(cond)


def scenario_a():
    print("a. same-day rule")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D1)  # registered TODAY (D1)
    r = check_watches(s, D1, {"AAA": bar(95, 101, 94, 100)}, FILLER)
    return check("AAA" in s["pending"] and not r["filled"], "watch registered today untouched despite h >= gap_open")


def scenario_b():
    print("b. prior-session touch fills intraday, EOD leaves it alone")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0, days_waited=0)
    snap = {"AAA": bar(94, 95.5, 93, 95.2)}
    r = check_watches(s, D1, snap, FILLER, adv_lookup=lambda tk: ADV_BIG, stamp="T10:00")
    pos = s["open_positions"].get("AAA")
    ok = check(r["filled"] == ["AAA"] and pos is not None, "filled AAA")
    ok &= check(pos and abs(pos["entry"] - 95.0 * (1 + SLIP)) < 1e-9, "entry = gap_open * (1+slip)")
    ok &= check(pos and abs(pos["position_dollars"] - CAPITAL / (SIZE_DIV or MAX_SLOTS)) < 1e-6,
                f"size = equity/{SIZE_DIV or MAX_SLOTS} (config SIZE_DIV or MAX_SLOTS)")
    ok &= check(pos and abs(pos["stop_price"] - pos["entry"] * (1 - stop_for_abs_gap(5.0) / 100)) < 1e-9,
                f"stop width = config.stop_for_abs_gap(5.0) = {stop_for_abs_gap(5.0)}% (profile-aware)")
    ok &= check(pos and pos["entry_time"] == "T10:00" and pos["days_held"] == 0, "entry_time stamped, days_held 0")
    # EOD for the same session: the settled candle shows the day's low went
    # through the stop later in the day -- must NOT be checked (entry day).
    eod = step_one_day(s, D1, {"AAA": bar(94, 95.5, 50, 60, prior_close=100)}, FILLER)
    ok &= check("AAA" in s["open_positions"] and not eod["trades"], "EOD did not stop out an entry-day position")
    ok &= check(s["open_positions"]["AAA"]["days_held"] == 0, "days_held still 0 after entry-day EOD")
    # Next session it IS checked, days_held -> 1 (the low is derived from the
    # position's own stop, whatever width the profile gave it).
    lo = pos["stop_price"] - 1
    eod2 = step_one_day(s, D2, {"AAA": bar(lo + 10, lo + 11, lo, lo + 5, prior_close=lo + 10)}, FILLER)
    ok &= check(len(eod2["trades"]) == 1 and eod2["trades"][0]["exit_reason"] == "stop"
                and eod2["trades"][0]["days_held"] == 1, "stopped the next session at days_held 1")
    return ok


def scenario_c():
    print("c. intraday stop/tp at engine prices, no EOD double count")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0)
    register(s, "BBB", 190.0, 200.0, D0)
    step_one_day(s, D1, {"AAA": bar(94, 96, 93, 95, prior_close=100),
                         "BBB": bar(189, 191, 188, 190, prior_close=200)}, FILLER)
    assert set(s["open_positions"]) == {"AAA", "BBB"}
    a, b = s["open_positions"]["AAA"], s["open_positions"]["BBB"]
    # D2 intraday: AAA hits tp, BBB hits stop.
    snap = {"AAA": bar(96, a["tp_price"] + 0.01, 95, 99), "BBB": bar(189, 190, b["stop_price"] - 0.01, 170)}
    cash_before = s["cash"]
    r = check_watches(s, D2, snap, FILLER, stamp="T11:15")
    by = {t["ticker"]: t for t in r["trades"]}
    ok = check(set(by) == {"AAA", "BBB"}, "both exits recorded intraday")
    ok &= check(by["AAA"]["exit_reason"] == "tp_gap_filled" and abs(by["AAA"]["exit_price"] - round(a["tp_price"] * (1 - SLIP), 4)) < 1e-6, "tp at tp_price*(1-slip)")
    ok &= check(by["BBB"]["exit_reason"] == "stop" and abs(by["BBB"]["exit_price"] - round(b["stop_price"] * (1 - SLIP), 4)) < 1e-6, "stop at stop_price*(1-slip)")
    ok &= check(by["AAA"]["days_held"] == 1 and by["AAA"]["exit_time"] == "T11:15", "days_held = 1, exit_time stamped")
    ok &= check(abs(s["cash"] - (cash_before + by["AAA"]["proceeds"] + by["BBB"]["proceeds"])) < 0.01, "cash credited once")
    # EOD same session: nothing left to exit, no re-entry (watches consumed).
    eod = step_one_day(s, D2, {"AAA": bar(96, 105, 95, 99, prior_close=95),
                               "BBB": bar(189, 190, 150, 170, prior_close=190)}, FILLER)
    ok &= check(not eod["trades"] and not s["open_positions"] and not s["pending"], "EOD found nothing to do")
    return ok


def scenario_d():
    print("d. expiring watch not checked intraday")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0, days_waited=HORIZON)  # EOD would make it HORIZON+1 -> expire
    r = check_watches(s, D1, {"AAA": bar(94, 99, 93, 98)}, FILLER, adv_lookup=lambda tk: ADV_BIG)
    ok = check(not r["filled"] and "AAA" in s["pending"], "touch ignored on a watch EOD will expire")
    step_one_day(s, D1, {"AAA": bar(94, 99, 93, 98, prior_close=100)}, FILLER)
    ok &= check("AAA" not in s["pending"] and "AAA" not in s["open_positions"], "EOD expired it, unfilled")
    return ok


def scenario_e():
    print("e. ADV refusal consumes the watch; stop-before-tp inside one poll")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0)
    r = check_watches(s, D1, {"AAA": bar(94, 96, 93, 95)}, FILLER, adv_lookup=lambda tk: 1.0)  # tiny ADV
    ok = check(not r["filled"] and "AAA" not in s["open_positions"] and s["pending"]["AAA"].get("consumed_on") == D1,
               "ADV-refused touch: no position, watch consumed (tagged for the EOD step, not deleted)")
    r = check_watches(s, D1, {"AAA": bar(94, 96, 93, 95)}, FILLER, adv_lookup=lambda tk: ADV_BIG)
    ok &= check(not r["filled"], "a later poll never re-tries a consumed watch")
    step_one_day(s, D1, {"AAA": bar(94, 96, 93, 95, prior_close=100)}, FILLER)
    ok &= check("AAA" not in s["pending"] and not s["open_positions"], "EOD drops the consumed watch unfilled")
    s = fresh()
    register(s, "BBB", 95.0, 100.0, D0)
    step_one_day(s, D1, {"BBB": bar(94, 96, 93, 95, prior_close=100)}, FILLER)
    p = s["open_positions"]["BBB"]
    r = check_watches(s, D2, {"BBB": bar(95, p["tp_price"] + 1, p["stop_price"] - 1, 95)}, FILLER)
    ok &= check(r["trades"] and r["trades"][0]["exit_reason"] == "stop", "both hit in one poll -> stop wins (engine order)")
    return ok


def scenario_f():
    print("f. no-event poll is a pure no-op for the EOD step")
    s1 = fresh()
    register(s1, "AAA", 95.0, 100.0, D0)
    register(s1, "BBB", 190.0, 200.0, D0)
    step_one_day(s1, D1, {"AAA": bar(94, 96, 93, 95, prior_close=100)}, FILLER)
    s2 = copy.deepcopy(s1)
    # poll with nothing touching / hitting
    r = check_watches(s2, D2, {"AAA": bar(95, 96, 94, 95.5), "BBB": bar(180, 185, 179, 182)}, FILLER)
    assert not r["trades"] and not r["filled"]
    day_bars = {"AAA": bar(95, 96, 94, 95.5, prior_close=95), "BBB": bar(180, 185, 179, 182, prior_close=190)}
    e1 = step_one_day(s1, D2, copy.deepcopy(day_bars), FILLER)
    e2 = step_one_day(s2, D2, copy.deepcopy(day_bars), FILLER)
    return check(s1 == s2 and e1["trades"] == e2["trades"] and abs(e1["equity"] - e2["equity"]) < 1e-9,
                 "state after EOD identical with and without the no-event poll")


def scenario_g():
    print("g. intraday events + EOD == EOD alone (order-independent case)")
    # Setup: AAA open (will hit tp on D2), BBB pending (will touch on D2), CCC pending (no touch).
    def setup():
        s = fresh()
        register(s, "AAA", 95.0, 100.0, D0)
        register(s, "BBB", 190.0, 200.0, D0)
        register(s, "CCC", 45.0, 50.0, D0)
        step_one_day(s, D1, {"AAA": bar(94, 96, 93, 95, prior_close=100),
                             "BBB": bar(180, 185, 179, 182, prior_close=200),
                             "CCC": bar(44, 44.5, 43, 44, prior_close=50)}, FILLER)
        return s
    s_eod_only, s_both = setup(), setup()
    tp = s_eod_only["open_positions"]["AAA"]["tp_price"]
    # D2 settled candle
    day_bars = {"AAA": bar(96, tp + 0.5, 95, tp, prior_close=95),
                "BBB": bar(185, 191, 184, 190.5, prior_close=182),
                "CCC": bar(44, 44.8, 43.5, 44.2, prior_close=44)}
    # Intraday snapshot mid-day: AAA already through tp, BBB already touched, marks = same as close
    # so sizing (equity_now) matches the EOD path exactly for this comparison.
    snap = {"AAA": bar(96, tp + 0.5, 95, tp), "BBB": bar(185, 191, 184, 190.5), "CCC": bar(44, 44.8, 43.5, 44.2)}
    ri = check_watches(s_both, D2, snap, FILLER, adv_lookup=lambda tk: ADV_BIG, stamp="T13:00")
    re_ = step_one_day(s_both, D2, copy.deepcopy(day_bars), FILLER)
    r1 = step_one_day(s_eod_only, D2, copy.deepcopy(day_bars), FILLER)
    trades_both = ri["trades"] + re_["trades"]
    strip = lambda t: {k: v for k, v in t.items() if k not in ("exit_time", "entry_time")}
    ok = check([strip(t) for t in trades_both] == [strip(t) for t in r1["trades"]], "same trades (modulo time stamps)")
    pos_strip = lambda ps: {k: {kk: vv for kk, vv in p.items() if kk != "entry_time"} for k, p in ps.items()}
    ok &= check(pos_strip(s_both["open_positions"]) == pos_strip(s_eod_only["open_positions"]), "same open positions")
    ok &= check(s_both["pending"] == s_eod_only["pending"], "same pending")
    ok &= check(abs(s_both["cash"] - s_eod_only["cash"]) < 1e-6, "same cash")
    # 2026-10-06 divergence: a watch consumed intraday (here: ADV-refused)
    # whose ticker ALSO gaps down again on the same settled candle. EOD
    # alone leaves it pending at 2a (no re-registration) and drops it at
    # 2c -> {}. Deleting it intraday let 2a register a fresh watch.
    def consumed_case(intraday):
        s = fresh()
        register(s, "AAA", 95.0, 100.0, D0)
        if intraday:
            check_watches(s, D1, {"AAA": bar(94, 95.5, 93, 95.2)}, FILLER, adv_lookup=lambda tk: 1.0)
        step_one_day(s, D1, {"AAA": bar(94, 95.5, 93, 95.2, prior_close=100, adv=1.0)}, FILLER)
        return s
    e, b = consumed_case(False), consumed_case(True)
    ok &= check(e["pending"] == {} and b["pending"] == e["pending"] and b["open_positions"] == e["open_positions"],
                "consumed watch + same-day re-gap on that ticker: pending == EOD alone (not re-registered)")
    return ok


def scenario_h():
    print("h. resting-limit fill + rest-top-N priority (2026-09-08 rule)")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0)
    r = check_watches(s, D1, {"AAA": bar(96, 99, 95.5, 98)}, FILLER, adv_lookup=lambda tk: ADV_BIG)  # whole day above 95
    ok = check(not r["filled"] and "AAA" in s["pending"], "range entirely above gap_open: no fill, watch keeps resting")
    step_one_day(s, D1, {"AAA": bar(96, 99, 95.5, 98, prior_close=100)}, FILLER)
    ok &= check("AAA" in s["pending"] and not s["open_positions"], "EOD agrees: still resting")
    # one free slot, two watches touch: only the bigger gap had an order resting.
    # Fillers and BIG are -15% gaps so the count is MAX_SLOTS-1 under both
    # profiles (ramp: slot_limit ramps one per >=12% watch up to RAMP_MAX);
    # SMALL is -6%, above the ramp ORDER_FLOOR, so priority is what decides.
    s = fresh()
    n_fill = MAX_SLOTS - 1
    for i in range(n_fill):  # fill all slots but one
        register(s, f"F{i:02d}", 85.0, 100.0, D0)
    step_one_day(s, D1, {f"F{i:02d}": bar(84, 86, 83, 85, prior_close=100) for i in range(n_fill)}, FILLER)
    assert len(s["open_positions"]) == n_fill, (len(s["open_positions"]), n_fill)
    register(s, "BIG", 85.0, 100.0, D1)    # -15%
    register(s, "SMALL", 94.0, 100.0, D1)  # -6%
    register(s, "QUIET", 96.0, 100.0, D1)  # -4%, never touches
    assert slot_limit(s["open_positions"], s["pending"]) - n_fill == 1, "scenario needs exactly one free slot"
    snap = {"BIG": bar(84, 86, 83, 85.5), "SMALL": bar(93, 95, 92, 94.5), "QUIET": bar(94, 95, 93, 94)}
    r = check_watches(s, D2, snap, FILLER, adv_lookup=lambda tk: ADV_BIG)
    ok &= check(r["filled"] == ["BIG"], "only the biggest-gap watch (the one with an order resting) fills")
    ok &= check("SMALL" not in s["open_positions"] and s["pending"]["SMALL"].get("consumed_on") == D2,
                "smaller gap touched without an order: consumed (tagged), unfilled")
    ok &= check("QUIET" in s["pending"] and "consumed_on" not in s["pending"]["QUIET"], "no touch: keeps resting")
    return ok


def _ms(ts: str) -> int:
    import pandas as pd
    return int(pd.Timestamp(ts, tz=ET).value // 1_000_000)


def _qjson(o, h, l, c, when, prior=100.0):
    return {"quote": {"openPrice": o, "highPrice": h, "lowPrice": l, "lastPrice": c,
                      "closePrice": prior, "securityStatus": "Normal"},
            "regular": {"regularMarketTradeTime": _ms(when)}}


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        return self._body


class FakeClient:
    """Stands in for the schwab-py client. Records every call; has NO order
    methods, so any attempt to place one raises AttributeError. `history`
    names the tickers that get a 25-day daily candle series (so adv_for
    resolves); any other ticker's history call fails with a 500."""
    def __init__(self, quotes=None, code=200, errors=None, history=()):
        self.quotes, self.code, self.errors, self.calls = quotes or {}, code, errors, []
        self.history = set(history)

    def get_quotes(self, tickers):
        self.calls.append(("get_quotes", list(tickers)))
        body = {t: self.quotes[t] for t in tickers if t in self.quotes}
        if self.errors:
            body["errors"] = self.errors
        return _Resp(self.code, body)

    def get_price_history_every_day(self, tk, *a, **k):
        self.calls.append(("price_history", (tk,)))
        if tk not in self.history:
            return _Resp(500, {})
        day0 = _ms("2026-07-28 00:00")
        candles = [{"datetime": day0 + i * 86_400_000, "open": 100.0, "high": 101.0, "low": 99.0,
                    "close": 100.0, "volume": 1_000_000} for i in range(25)]
        return _Resp(200, {"candles": candles})


def scenario_i():
    import pandas as pd
    print("i. quote freshness + ledger age guards")
    sess = pd.Timestamp("2026-09-02").date()
    mk = lambda when: {"trade_time": None if when is None else pd.Timestamp(when, tz=ET)}
    fresh, dropped = drop_stale_quotes({"A": mk("2026-09-02 09:50"), "B": mk("2026-09-01 15:59"),
                                        "C": mk(None), "D": mk("2026-09-02 15:59")}, sess)
    ok = check(set(fresh) == {"A", "D"} and dropped == ["B", "C"], "prior-session / undated quotes dropped, today's kept")
    ok &= check(not state_is_stale(None, D1), "never-run ledger is not stale")
    ok &= check(not state_is_stale(D0, D1) and not state_is_stale("2026-08-28", "2026-08-31"),
                "previous session (Tue -> Wed, Fri -> Mon) not stale")
    ok &= check(not state_is_stale("2026-09-04", "2026-09-08") and previous_session("2026-09-08") == "2026-09-04",
                "Fri -> Tue over Labor Day (NYSE holiday) not stale")
    ok &= check(state_is_stale("2026-08-31", "2026-09-02") and state_is_stale("2026-08-28", "2026-09-01"),
                "one missed evening run (Mon -> Wed, Fri -> Tue) is stale")
    ok &= check(state_is_stale("2026-09-04", "2026-10-05"), "post-pause ledger (31d) is stale")
    return ok


def scenario_j():
    print("j. fetch_quotes parsing against a fake client")
    q = {"AAA": _qjson(94, 95.5, 93, 95.2, "2026-09-02 10:00"),
         "ZERO": _qjson(0, 0, 0, 0, "2026-09-02 10:00"),
         "NONE": {"quote": {}},
         "NOTIME": {"quote": {"openPrice": 1, "highPrice": 2, "lowPrice": 1, "lastPrice": 1.5}}}
    out = fetch_quotes(FakeClient(q, errors={"invalidSymbols": ["BRK-B"]}), ["AAA", "ZERO", "NONE", "NOTIME", "GONE"])
    ok = check(set(out) == {"AAA", "NOTIME"}, "no-print / empty / missing rows left out")
    ok &= check(out["AAA"]["h"] == 95.5 and out["AAA"]["c"] == 95.2 and out["AAA"]["trade_time"].hour == 10,
                "o/h/l/last parsed, trade time in ET")
    ok &= check(out["NOTIME"]["trade_time"] is None, "missing trade time kept as None (dropped later by freshness)")
    ok &= check(fetch_quotes(FakeClient(q, code=500), ["AAA"]) == {}, "HTTP error: chunk skipped, no raise")

    class Boom(FakeClient):
        def get_quotes(self, tickers):
            raise RuntimeError("net down")
    ok &= check(fetch_quotes(Boom(), ["AAA"]) == {}, "exception: chunk skipped, no raise")
    return ok


def _run_poller(tmp, client, now, variants="stock", extra_args=()):
    """Drive run_intraday.main() with a fake client, a pinned clock and a
    temp state dir. Returns (exit code, captured stdout)."""
    import pandas as pd
    import live.run_intraday as ri
    import live.schwab_data as sd
    import live.schwab_options as so
    saved = (ri.now_et, sd.get_client, so.make_chain_provider, sys.argv, os.environ.get("GAPBOT_STATE_DIR"))
    ri.now_et = lambda: pd.Timestamp(now, tz=ET)
    sd.get_client = lambda: client
    so.make_chain_provider = lambda c: None
    sys.argv = ["run_intraday.py", "--variants", variants, *extra_args]
    os.environ["GAPBOT_STATE_DIR"] = tmp
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = ri.main()
    finally:
        ri.now_et, sd.get_client, so.make_chain_provider, sys.argv = saved[:4]
        if saved[4] is None:
            os.environ.pop("GAPBOT_STATE_DIR", None)
        else:
            os.environ["GAPBOT_STATE_DIR"] = saved[4]
    return rc, buf.getvalue()


def _seed(tmp, last_run=D0, tk="AAA"):
    from live import state as state_mod
    old = os.environ.get("GAPBOT_STATE_DIR")
    os.environ["GAPBOT_STATE_DIR"] = tmp
    try:
        s = fresh()
        s["last_run_date"] = last_run
        register(s, tk, 95.0, 100.0, D0)
        state_mod.save_state(s, "stock")
    finally:
        if old is None:
            os.environ.pop("GAPBOT_STATE_DIR", None)
        else:
            os.environ["GAPBOT_STATE_DIR"] = old


def _load(tmp):
    with open(os.path.join(tmp, "account", "state.json")) as f:
        return json.load(f)


def scenario_k():
    print("k. run_intraday end to end (fake client, temp state dir)")
    T = "2026-09-02 10:00"
    spy = _qjson(500, 501, 499, 500, "2026-09-02 09:59")
    touch = _qjson(94, 95.5, 93, 95.2, "2026-09-02 09:58")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        _seed(tmp)
        rc, out = _run_poller(tmp, FakeClient({"SPY": spy, "AAA": touch}), T)  # no history -> no ADV
        st = _load(tmp)
        ok &= check(rc == 0 and "ADV unavailable" in out and "AAA" in st["pending"] and not st["open_positions"],
                    "ADV lookup fails (history 500): touch left for the EOD run, no fill")
    with tempfile.TemporaryDirectory() as tmp:
        _seed(tmp)
        c = FakeClient({"SPY": spy, "AAA": touch}, history=["AAA"])
        rc, out = _run_poller(tmp, c, T, extra_args=("--dry-run",))
        st = _load(tmp)
        ok &= check(rc == 0 and "DRY-RUN would record entry AAA" in out and "AAA" in st["pending"]
                    and not st["open_positions"] and not os.path.exists(os.path.join(tmp, "account", "intraday.json")),
                    "--dry-run reports the fill, persists nothing")
        rc, out = _run_poller(tmp, c, T)
        st = _load(tmp)
        ok &= check(rc == 0 and "AAA" in st["open_positions"] and "AAA" not in st["pending"]
                    and st["open_positions"]["AAA"]["entry_date"] == D1
                    and st["open_positions"]["AAA"]["entry_time"].startswith(D1)
                    and st["last_run_date"] == D0, "touch persisted; entry-stamped; last_run_date untouched")
        hb = json.load(open(os.path.join(tmp, "account", "intraday.json")))
        ok &= check(hb["session"] == D1 and hb["entries"] == ["AAA"] and "AAA" in hb["watched"]
                    and hb["profile"] == PROFILE, "heartbeat written, carries the profile")
        ok &= check(out.startswith(PROFILE_BANNER), "profile banner is the poll's first line")
        ok &= check(all(name == "get_quotes" or name == "price_history" for name, _ in c.calls)
                    and not any("order" in n for n in dir(c)), "only data-read endpoints exist/were called")
        # EOD already ran this session -> no-op
        s = _load(tmp)
        s["last_run_date"] = D1
        json.dump(s, open(os.path.join(tmp, "account", "state.json"), "w"))
        rc, out = _run_poller(tmp, c, T)
        ok &= check(rc == 0 and "already closed out" in out, "variant already closed by EOD: no-op")
    with tempfile.TemporaryDirectory() as tmp:
        _seed(tmp)
        rc, out = _run_poller(tmp, FakeClient({"SPY": spy, "AAA": _qjson(94, 95.5, 93, 95.2, "2026-09-01 15:59")}), T)
        st = _load(tmp)
        ok &= check(rc == 0 and "AAA" in st["pending"] and not st["open_positions"] and "not dated" in out,
                    "stale (prior-session) quote: no phantom touch")
    with tempfile.TemporaryDirectory() as tmp:
        _seed(tmp, last_run="2026-08-14")
        rc, out = _run_poller(tmp, FakeClient({"SPY": spy, "AAA": touch}), T)
        st = _load(tmp)
        ok &= check(rc == 0 and "ledger is stale" in out and "AAA" in st["pending"] and not st["open_positions"],
                    "stale ledger (paused / missed sessions): refused, untouched")
    with tempfile.TemporaryDirectory() as tmp:
        _seed(tmp)
        rc, out = _run_poller(tmp, FakeClient({"SPY": _qjson(500, 501, 499, 500, "2026-09-01 15:59"), "AAA": touch}), T)
        ok &= check(rc == 0 and "no session today" in out and "AAA" in _load(tmp)["pending"], "holiday / pre-open: clean no-op")
        rc, out = _run_poller(tmp, FakeClient({"AAA": touch}), T)
        ok &= check(rc == 1 and "reference quote missing" in out, "no reference quote: exits 1")
        import live.run_intraday as ri
        real = ri.check_watches
        ri.check_watches = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            rc, out = _run_poller(tmp, FakeClient({"SPY": spy, "AAA": touch}), T)
        finally:
            ri.check_watches = real
        ok &= check(rc == 1 and "FAILED" in out and "AAA" in _load(tmp)["pending"], "variant raised: exits 1, state intact")
    return ok


def _stub_recording_py(d: str) -> str:
    """A stand-in python that prints what run_intraday prints on a fill."""
    p = Path(d, "py_recording")
    p.write_text("#!/bin/bash\necho 'run_intraday[stock]: recorded entry AAA @ 95.0950'\n")
    p.chmod(0o755)
    return str(p)


def scenario_l():
    print("l. gap_bot_tick.sh window gating")
    repo_root = Path(__file__).parent.parent
    script = repo_root / "scripts" / "gap_bot_tick.sh"
    ok = True

    def tick(dow, hm, py="/bin/echo", env=None, repo=None):
        """One tick in a fresh temp repo (or `repo`, to chain polls in one
        session). Returns (rc, tick.log text)."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = repo or tmp
            e = {**os.environ, "GAPBOT_REPO": repo, "GAPBOT_VENV_PY": py,
                 "GAPBOT_FAKE_DOW": str(dow), "GAPBOT_FAKE_HM": str(hm), **(env or {})}
            r = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True)
            log = Path(repo, "data/live/logs/tick.log")
            return r.returncode, (log.read_text() if log.exists() else "")

    # a fresh repo per case, so each intraday tick is the session's first
    # poll (the one a no-event poll still logs)
    cases = [(2, 930, None), (2, 931, "intraday"), (2, 1000, "intraday"), (2, 1600, "intraday"),
             (2, 1601, None), (2, 1659, None), (2, 1700, "eod"), (2, 1701, None), (2, 1705, "eod"),
             (2, 2330, "eod"), (2, 2331, None), (6, 1000, None), (7, 1800, None)]
    bad = []
    for dow, hm, want in cases:
        rc, log = tick(dow, hm)
        got = "intraday" if "tick ok (intraday)" in log else "eod" if "tick ok (eod)" in log else None
        if rc != 0 or got != want:
            bad.append((dow, hm, want, got, rc))
    ok &= check(not bad, f"window gating correct for {len(cases)} (dow, HHMM) cases" + (f" -- BAD: {bad}" if bad else ""))
    # (a dev box without flock(1) logs "running unlocked" on every tick; the VPS has it)
    quiet = lambda log: "\n".join(ln for ln in log.splitlines() if "flock not on PATH" not in ln)
    with tempfile.TemporaryDirectory() as repo:
        rc1, log1 = tick(2, 1000, repo=repo)
        rc2, log2 = tick(2, 1001, repo=repo)
        ok &= check(rc1 == 0 and rc2 == 0 and "first poll of the session" in log1 and quiet(log2) == quiet(log1),
                    "no-event poll after the session's first writes nothing to tick.log")
        rc3, log3 = tick(2, 1002, repo=repo, py=_stub_recording_py(repo))
        ok &= check(rc3 == 0 and log3.count("tick ok (intraday)") == 2 and "recorded entry" in log3,
                    "a poll that recorded an event is logged with its output")
    rc, log = tick(2, 1000, env={"GAPBOT_INTRADAY": "0"})
    ok &= check(rc == 0 and "tick" not in log, "GAPBOT_INTRADAY=0 leaves the market-hours window a no-op")
    rc2, log2 = tick(2, 1800, env={"GAPBOT_INTRADAY": "0"})
    ok &= check("tick ok (eod)" in log2, "GAPBOT_INTRADAY=0 does not disable the EOD run")
    rc, log = tick(2, 1000, py="/usr/bin/false")
    ok &= check(rc == 1 and "tick FAILED (intraday), exit 1" in log, "failing poll: logged FAILED, exit 1")
    return ok


def scenario_m():
    print("m. ADV lookup failure intraday fails closed (watch left for the EOD run)")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0)
    snap = {"AAA": bar(94, 95.5, 93, 95.2, adv=None)}  # a quote carries no adv; the lookup fails
    r = check_watches(s, D1, dict(snap), FILLER, adv_lookup=lambda tk: None)
    ok = check(not r["filled"] and "AAA" in s["pending"] and "consumed_on" not in s["pending"]["AAA"],
               "no ADV: not filled, not consumed")
    r = check_watches(s, D1, dict(snap), FILLER, adv_lookup=lambda tk: ADV_BIG)
    ok &= check(r["filled"] == ["AAA"], "next poll with ADV fills it")
    s = fresh()
    register(s, "AAA", 95.0, 100.0, D0)
    step_one_day(s, D1, {"AAA": bar(94, 95.5, 93, 95.2, prior_close=100, adv=None)}, FILLER)
    ok &= check("AAA" in s["open_positions"], "EOD unchanged: adv None there is a newly-listed name, unfloored")
    return ok


def part2_contention() -> bool:
    print(f"\nPart 2: slot contention in the 2yr replay, profile={PROFILE} (resting-limit touches vs free slots)")
    import pandas as pd
    from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView
    tickers_data, longest = {}, None
    for p in sorted(BARS_DIR.glob("*.parquet")):
        d = load_ticker(p)
        if d is None:
            continue
        tickers_data[p.stem] = TickerView(d)
        if longest is None or len(d) > len(longest):
            longest = d["Date"]
    calendar = pd.DatetimeIndex(sorted(longest))
    calendar = calendar[calendar >= calendar[-1] - pd.DateOffset(years=2)]
    state = fresh()
    n_days = n_touch_days = n_contended = 0
    excess = []
    for day in calendar:
        today = day.isoformat()[:10]
        bars = {}
        for tk, tv in tickers_data.items():
            row = tv.get(day)
            if row is None:
                continue
            o, h, l, c = row
            pr = tv.get_prior_close(day)
            bars[tk] = {"o": float(o), "h": float(h), "l": float(l), "c": float(c),
                        "prior_close": None if pr is None else float(pr), "adv": tv.get_adv(day)}
        # Before stepping: how many PRIOR-session live watches touch today, vs free slots?
        # (exits today free slots before fills in the engine, so count those too.)
        touches = 0
        for tk, w in state["pending"].items():
            if w["gap_date"] == today or w["days_waited"] + 1 > HORIZON:
                continue
            r = bars.get(tk)
            if r is not None and r["l"] <= w["gap_open"] <= r["h"]:
                touches += 1
        exits_today = 0
        for tk, pos in state["open_positions"].items():
            r = bars.get(tk)
            dh = pos["days_held"] + 1
            if r is None:
                exits_today += dh >= HORIZON
            else:
                exits_today += (r["l"] <= pos["stop_price"] or r["h"] >= pos["tp_price"] or dh >= HORIZON)
        # the day's slot limit the engine itself will read (anchor: MAX_SLOTS; ramp: 6->9)
        free = slot_limit(state["open_positions"], state["pending"]) - len(state["open_positions"]) + exits_today
        n_days += 1
        if touches:
            n_touch_days += 1
        if touches > free:
            n_contended += 1
            excess.append(touches - free)
        step_one_day(state, today, bars, FILLER)
    print(f"  sessions: {n_days}, sessions with >=1 touch: {n_touch_days}, "
          f"sessions where touches > free slots: {n_contended}"
          + (f" (excess touches per such day: min {min(excess)}, max {max(excess)})" if excess else ""))
    print("  -> on those sessions the rest-top-N choice decided which touch got a fill;"
          " on every other session every touch had an order resting.")
    return check(n_touch_days > 0,
                 "replay saw touches (bars loaded) -- zero here means the Date-dtype env bug, see module docstring")


def main():
    print(PROFILE_BANNER)
    print("Part 1: synthetic invariants")
    results = [f() for f in (scenario_a, scenario_b, scenario_c, scenario_d, scenario_e, scenario_f, scenario_g,
                             scenario_h, scenario_i, scenario_j, scenario_k, scenario_l, scenario_m)]
    results.append(part2_contention())
    print("\nALL PASS" if all(results) else "\nFAILURES above")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
