"""Gap-bot's intraday entry point -- fired every minute during the
regular session by scripts/gap_bot_tick.sh (09:31-16:00 ET), one batched
quote pull on the watched set, one check per variant, persist. See
live/intraday.py for what this does and does not change; the short
version: it only makes the ledger notice an already-pending watch's
touch or an open position's stop/tp sooner. Entry/exit RULES, new-gap
registration, the same-day rule and every counter stay with
run_daily.py's after-close run, which is unchanged and still the
backstop if this never fires.

PAPER ONLY. No order placement, nothing here talks to a broker's order
endpoint -- data reads on the same read-only Schwab client as run_daily.

Guards, in order, each a logged no-op rather than a guess:
  * nothing watched (no positions, no pending) -> no quote call at all;
  * reference ticker's last regular trade isn't dated today (ET) ->
    holiday / pre-open / stale feed -> no-op;
  * a variant whose last_run_date == today has already been closed out by
    the EOD run for this session -> skipped (the poller must never act
    after the day-step has advanced the counters for the same date);
  * a variant whose last_run_date is not the previous NYSE session (an
    evening run was missed, or the bot was paused) -> refused: its counters
    and watch ages are stale and the EOD engine never backfills (same rule
    as run_daily's own refusal, live/intraday.py state_is_stale);
  * a watched ticker whose quote isn't dated today (no regular print yet,
    halted, stale feed) -> left out of the poll like an absent bar.

Exit code: 0 for a clean poll or a legitimate no-op (nothing watched,
holiday, already closed out, stale-state refusal, nothing to do); 1 when a
poll COULD NOT run -- reference quote missing, or a variant raised -- so
gap_bot_tick.sh logs FAILED instead of "ok".

Live:     .venv-live/bin/python live/run_intraday.py
Dry run:  .venv-live/bin/python live/run_intraday.py --dry-run
          (real quotes, real check, prints what WOULD be written, persists
          nothing -- for a first smoke on the VPS before enabling)

Run from code/gap-bot/ (paths.py resolves data/live/ relative to cwd).
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live import state as state_mod
from live.config import PROFILE, PROFILE_BANNER, VARIANTS
from live.fillers import make_filler
from live.intraday import (
    watched_tickers, fetch_quotes, session_date_from_quotes, adv_for, check_watches, now_et,
    drop_stale_quotes, state_is_stale, previous_session,
)
from live.paths import account_paths

REFERENCE = "SPY"


def write_heartbeat(variant: str, payload: dict) -> None:
    """data/live/<account>/intraday.json: last poll time, session, and the
    running h/l/last of every watched name next to its trigger levels --
    so the dashboard (and the owner's daily review) can see what the
    poller is watching without re-fetching quotes. Overwritten each poll;
    synced to the mirror with the rest of the account dir."""
    p = account_paths(variant)
    os.makedirs(p["dir"], exist_ok=True)
    state_mod.write_atomic(os.path.join(p["dir"], "intraday.json"), payload)


def watch_table(state: dict, snap: dict) -> dict:
    out = {}
    for tk, pos in state["open_positions"].items():
        row = snap.get(tk, {})
        out[tk] = {"kind": "open", "stop": round(pos["stop_price"], 4), "tp": round(pos["tp_price"], 4),
                   "h": row.get("h"), "l": row.get("l"), "last": row.get("c")}
    for tk, w in state["pending"].items():
        row = snap.get(tk, {})
        out[tk] = {"kind": "pending", "gap_open": round(w["gap_open"], 4), "gap_date": w["gap_date"],
                   "consumed_on": w.get("consumed_on"),
                   "h": row.get("h"), "l": row.get("l"), "last": row.get("c")}
    return out


def run_variant(variant: str, today: str, snap: dict, client, chain_provider, stamp: str,
                dry_run: bool) -> None:
    state = state_mod.load_state(variant)
    if state["last_run_date"] == today:
        print(f"run_intraday[{variant}]: EOD run already closed out {today} -- no-op.")
        return
    if state_is_stale(state["last_run_date"], today):
        print(f"run_intraday[{variant}]: last EOD run was {state['last_run_date']}, not the previous "
              f"session ({previous_session(today)}) -- ledger is stale (missed evening run or paused), "
              f"refusing to act intraday. The evening run refuses too; catch-up is the owner's call.")
        return
    if not watched_tickers(state):
        print(f"run_intraday[{variant}]: nothing watched -- no-op.")
        return
    filler = make_filler(variant, VARIANTS[variant])
    result = check_watches(state, today, snap, filler, chain_provider,
                           adv_lookup=lambda tk: adv_for(client, tk), stamp=stamp)
    tag = "DRY-RUN would record" if dry_run else "recorded"
    for t in result["trades"]:
        print(f"run_intraday[{variant}]: {tag} exit {t['ticker']} {t['exit_reason']} "
              f"@ {t['exit_price']} ({t['pnl_pct']:+.2f}%) at {stamp}")
    for tk in result["filled"]:
        pos = state["open_positions"][tk]
        print(f"run_intraday[{variant}]: {tag} entry {tk} @ {pos['entry']:.4f} "
              f"(stop {pos['stop_price']:.4f}, tp {pos['tp_price']:.4f}) at {stamp}")
    print(f"run_intraday[{variant}]: {len(result['trades'])} exits, {len(result['filled'])} entries, "
          f"{result['n_open']} open, {result['n_pending']} pending, equity ${result['equity']:,.2f}")
    if dry_run:
        return
    for t in result["trades"]:
        state_mod.append_trade(t, variant)
    state_mod.save_state(state, variant)
    write_heartbeat(variant, {
        "polled_at": stamp, "session": today, "profile": PROFILE, "equity": round(result["equity"], 2),
        "exits": [t["ticker"] for t in result["trades"]], "entries": result["filled"],
        "watched": watch_table(state, snap),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="real quotes and a real check, but persist nothing")
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset of stock,call,spread (default: all)")
    args = ap.parse_args()
    variants = args.variants.split(",") if args.variants else list(VARIANTS)
    # once per run, first line: the profile is process-wide and read from the
    # environment, so the log must show which one this poll actually ran
    print(PROFILE_BANNER)

    states = {v: state_mod.load_state(v) for v in variants}
    watched = sorted(set().union(*(watched_tickers(s) for s in states.values())))
    if not watched:
        print("run_intraday: nothing watched by any variant -- no quote call, no-op.")
        return 0

    from live.schwab_data import get_client
    from live.schwab_options import make_chain_provider
    client = get_client()
    snap = fetch_quotes(client, watched + ([REFERENCE] if REFERENCE not in watched else []))
    now = now_et()
    session = session_date_from_quotes(snap, REFERENCE)
    if session is None:
        print("run_intraday: reference quote missing -- can't date the session, skipping this poll.")
        return 1
    if session != now.date():
        print(f"run_intraday: {REFERENCE}'s last regular trade is {session}, wall clock says "
              f"{now.date()} ET -- no session today (holiday / pre-open / stale feed), no-op.")
        return 0
    today = session.isoformat()
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    if REFERENCE not in watched:
        snap.pop(REFERENCE, None)
    snap, stale = drop_stale_quotes(snap, session)
    if stale:
        print(f"run_intraday: {len(stale)} quote(s) not dated {today}, left out this poll: "
              f"{stale[:10]}{' ...' if len(stale) > 10 else ''}")
    print(f"run_intraday: {today} {now.strftime('%H:%M')} ET -- {len(snap)}/{len(watched)} watched tickers quoted")

    chain_provider = make_chain_provider(client)
    failed = False
    for variant in variants:
        cp = None if variant == "stock" else chain_provider
        try:
            run_variant(variant, today, snap, client, cp, stamp, args.dry_run)
        except Exception as e:
            print(f"run_intraday[{variant}]: FAILED, skipped this poll: {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
