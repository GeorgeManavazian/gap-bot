"""Gap-bot's daily entry point -- one call per real trading day, after the
close. Pulls today's bars, steps the anchor config forward one day, persists
state, appends the trade/equity logs. Refuses to double-run the same day
(the last_run_date guard in state.json).

"Today" is driven by the DATA, not the wall clock: a cheap one-ticker pull
(schwab_data.latest_session_date) finds the most recent session Schwab
actually has a candle for, and that date -- not pd.Timestamp.now() -- is
what gets compared against last_run_date and passed to the engine. Fixed
2026-09-07 after finding + reproducing a real bug: the VPS clock is UTC,
which rolls to the next calendar date while the ET trading day is still
open, so wall-clock "today" could both (a) phantom-fill every pending
watch by tricking the engine into thinking a new day had started mid-
session, and (b) on a market holiday, relabel the last real session's
stale bar as the holiday's date instead of recognizing nothing new
happened. Driving the date off the data fixes both, and as a side
effect makes a same-session retrigger cheap (one ticker, not 500) --
the full universe pull only happens once a genuinely new session exists.

Live:     .venv-live/bin/python live/run_daily.py
Dry run:  .venv-live/bin/python live/run_daily.py --dry-run --as-of 2026-08-18
          (replays a historical date from the cached parquets instead of
          calling Schwab -- for exercising the pipeline before it's live)

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
from live.engine import step_one_day
from live.schwab_data import bars_for_today

UNIVERSE_CSV = Path(__file__).parent.parent / "data" / "sp500_constituents.csv"


def load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV)
    return [t.replace(".", "-") for t in df["Symbol"].tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="use cached local parquets instead of live Schwab data")
    ap.add_argument("--as-of", default=None,
                    help="dry-run only: replay this date (YYYY-MM-DD) as 'today'")
    args = ap.parse_args()

    universe = load_universe()
    state = state_mod.load_state()

    if args.dry_run:
        # Controlled replay path: as_of drives the session date directly,
        # not subject to the live wall-clock bug below.
        from live import fixture_data
        as_of = args.as_of or pd.Timestamp.now().normalize().isoformat()[:10]
        session_date = pd.Timestamp(as_of).normalize().date().isoformat()
        if state["last_run_date"] == session_date:
            print(f"run_daily: already ran for {session_date} -- refusing to double-step.")
            return
        universe_bars = fixture_data.fetch_universe_bars(universe, as_of)
    else:
        from live.schwab_data import get_client, fetch_universe_bars, latest_session_date
        client = get_client()
        # Cheap one-ticker check BEFORE the full pull: session_date comes
        # from the data itself, not pd.Timestamp.now() (that was the bug --
        # see the module docstring). If it's the same session already
        # processed, skip now and never pay for the other ~500 tickers.
        latest = latest_session_date(client)
        if latest is None:
            print("run_daily: couldn't determine the latest session date (reference pull failed) -- skipping this tick.")
            return
        session_date = latest.isoformat()
        if state["last_run_date"] == session_date:
            print(f"run_daily: no new session yet (last session {session_date} already processed) -- skipping the full pull.")
            return
        universe_bars = fetch_universe_bars(client, universe)

    bars = bars_for_today(universe_bars)
    print(f"run_daily: {session_date} -- {len(bars)}/{len(universe)} tickers with usable bars")

    result = step_one_day(state, session_date, bars)

    for t in result["trades"]:
        state_mod.append_trade(t)
    state_mod.append_snapshot({
        "date": session_date, "cash": round(result["state"]["cash"], 2),
        "equity": round(result["equity"], 2), "n_open": result["n_open"],
        "n_pending": result["n_pending"],
    })
    state_mod.save_state(result["state"])

    print(f"run_daily: {len(result['trades'])} exits today, "
          f"{result['n_open']} open, {result['n_pending']} pending, "
          f"equity ${result['equity']:,.2f}")


if __name__ == "__main__":
    main()
