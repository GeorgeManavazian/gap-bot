"""Gap-bot's daily entry point -- one call per real trading day, after the
close. Pulls today's bars ONCE, then steps all three instrument variants
forward one day each (stock/call/spread, live/config.py VARIANTS) --
same signal, same day, three independent $100k paper accounts. Refuses to
double-run a variant on the same day (that variant's own last_run_date
guard in its state.json) -- a variant added after the others (call/spread,
2026-09-08) simply starts fresh from today rather than backfilling. Also
refuses (exit 1, nothing written) to step a ledger whose last_run_date is
not the previous NYSE session (2026-10-06): the engine never backfills, so
a missed evening run or a pause has to be replayed with
scripts/catch_up_ledger.py first. `--allow-stale-from YYYY-MM-DD` steps
anyway, but only while last_run_date equals that exact date -- for the
one case the calendar gets wrong (an unscheduled closure makes a real
one-session gap look like a missed run); it can't be left in a cron line.

"Today" is driven by the DATA, not the wall clock: a cheap one-ticker pull
(schwab_data.latest_session_date) finds the most recent session Schwab
actually has a candle for, and that date -- not pd.Timestamp.now() -- is
what gets compared against each variant's last_run_date and passed to the
engine. Fixed 2026-09-07 after finding + reproducing a real bug: the VPS
clock is UTC, which rolls to the next calendar date while the ET trading
day is still open, so wall-clock "today" could both (a) phantom-fill every
pending watch by tricking the engine into thinking a new day had started
mid-session, and (b) on a market holiday, relabel the last real session's
stale bar as the holiday's date instead of recognizing nothing new
happened. Driving the date off the data fixes both, and as a side
effect makes a same-session retrigger cheap (one ticker, not 500) --
the full universe pull only happens once a genuinely new session exists.

The call/spread variants additionally need live option chain quotes
(live/schwab_options.py) -- NOT YET smoke-tested against a live account,
see that module's docstring. One variant's chain-provider failing to
price a candidate skips that candidate for that variant only (fillers.py
doctrine); it never stops the stock variant or crashes the tick.

Live:     .venv-live/bin/python live/run_daily.py
Dry run:  .venv-live/bin/python live/run_daily.py --dry-run --as-of 2026-08-18
          (replays a historical date from the cached parquets instead of
          calling Schwab, and prices call/spread off a synthetic
          Black-Scholes chain instead of a live one -- for exercising the
          pipeline before it's live, not a performance validation)

Run from code/gap-bot/ (paths.py resolves data/live/ relative to cwd, same
convention as the wheel bot's live/paths.py).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # so `live.*` imports work run from anywhere

from live import state as state_mod
from live.config import PROFILE_BANNER, VARIANTS
from live.engine import step_one_day
from live.fillers import make_filler
from live.intraday import previous_session, state_is_stale
from live.schwab_data import bars_for_today

UNIVERSE_CSV = Path(__file__).parent.parent / "data" / "sp500_constituents.csv"


def load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV)
    return [t.replace(".", "-") for t in df["Symbol"].tolist()]


def run_variant(variant: str, session_date: str, bars: dict, chain_provider,
                allow_stale_from: str | None = None) -> bool:
    """Step one variant. False (nothing written) when the ledger is stale."""
    state = state_mod.load_state(variant)
    if state["last_run_date"] == session_date:
        print(f"run_daily[{variant}]: already ran for {session_date} -- refusing to double-step.")
        return True
    if state_is_stale(state["last_run_date"], session_date):
        # The engine steps ONE session and never backfills: stepping a ledger
        # whose last run is older than the previous session would age every
        # watch/position by one day and check stops against one bar for all
        # the sessions in between. Refuse; the owner replays the gap with
        # scripts/catch_up_ledger.py first. The override names the exact
        # last_run_date it is for, so a leftover flag can never skip a
        # session unnoticed.
        if allow_stale_from != state["last_run_date"]:
            print(f"run_daily[{variant}]: last_run_date {state['last_run_date']} is not the session before "
                  f"{session_date} (expected {previous_session(session_date)}) -- ledger is stale, refusing to "
                  f"step it, nothing written. Replay the gap with scripts/catch_up_ledger.py first; for a "
                  f"one-session gap the calendar got wrong (unscheduled closure), "
                  f"--allow-stale-from {state['last_run_date']}.")
            return False
        print(f"run_daily[{variant}]: --allow-stale-from {allow_stale_from}: stepping the ledger straight "
              f"to {session_date}.")
    filler = make_filler(variant, VARIANTS[variant])
    result = step_one_day(state, session_date, bars, filler, chain_provider)
    for t in result["trades"]:
        state_mod.append_trade(t, variant)
    state_mod.append_snapshot({
        "date": session_date, "cash": round(result["state"]["cash"], 2),
        "equity": round(result["equity"], 2), "n_open": result["n_open"],
        "n_pending": result["n_pending"],
    }, variant)
    state_mod.save_state(result["state"], variant)
    print(f"run_daily[{variant}]: {len(result['trades'])} exits today, "
          f"{result['n_open']} open, {result['n_pending']} pending, "
          f"equity ${result['equity']:,.2f}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="use cached local parquets + a synthetic chain instead of live Schwab data")
    ap.add_argument("--as-of", default=None,
                    help="dry-run only: replay this date (YYYY-MM-DD) as 'today'")
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset of stock,call,spread (default: all)")
    ap.add_argument("--allow-stale-from", default=None, metavar="YYYY-MM-DD",
                    help="step a stale ledger ONLY if its last_run_date equals this date: for a one-session "
                         "gap the NYSE calendar refused wrongly (unscheduled market closure). A real gap is "
                         "replayed with scripts/catch_up_ledger.py, after which this is a no-op anyway.")
    args = ap.parse_args()
    variants = args.variants.split(",") if args.variants else list(VARIANTS)
    # once per run, first line: the profile is process-wide and read from the
    # environment, so the log must show which one this step actually ran
    print(PROFILE_BANNER)

    universe = load_universe()

    if args.dry_run:
        from live import fixture_data
        from live.fixture_options import make_fixture_chain_provider
        as_of = args.as_of or pd.Timestamp.now().normalize().isoformat()[:10]
        session_date = pd.Timestamp(as_of).normalize().date().isoformat()
        universe_bars = fixture_data.fetch_universe_bars(universe, as_of)
        bars = bars_for_today(universe_bars)
        chain_provider = make_fixture_chain_provider(lambda tk: bars.get(tk, {}).get("c"))
        client = None
    else:
        from live.schwab_data import get_client, fetch_universe_bars, latest_session_date
        from live.schwab_options import make_chain_provider
        client = get_client()
        # Cheap one-ticker check BEFORE the full pull: session_date comes
        # from the data itself, not pd.Timestamp.now() (that was the bug --
        # see the module docstring). If every variant already processed
        # this session, skip now and never pay for the other ~500 tickers.
        latest = latest_session_date(client)
        if latest is None:
            print("run_daily: couldn't determine the latest session date (reference pull failed) -- skipping this tick.")
            return 0
        session_date = latest.isoformat()
        if all(state_mod.load_state(v)["last_run_date"] == session_date for v in variants):
            print(f"run_daily: no new session yet (last session {session_date} already processed by every requested variant) -- skipping the full pull.")
            return 0
        universe_bars = fetch_universe_bars(client, universe)
        bars = bars_for_today(universe_bars)
        chain_provider = make_chain_provider(client)

    print(f"run_daily: {session_date} -- {len(bars)}/{len(universe)} tickers with usable bars")

    refused = False
    for variant in variants:
        cp = None if variant == "stock" else chain_provider
        try:
            refused |= not run_variant(variant, session_date, bars, cp, args.allow_stale_from)
        except Exception as e:
            # One variant's failure (e.g. a chain-provider outage) must
            # never take down the others -- same one-bad-thing doctrine as
            # fetch_universe_bars' per-ticker try/except.
            print(f"run_daily[{variant}]: FAILED, skipped this tick: {e}")
    return 1 if refused else 0  # a stale refusal makes the tick log FAILED, not "ok"


if __name__ == "__main__":
    sys.exit(main())
