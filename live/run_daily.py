"""Gap-bot's daily entry point -- one call per real trading day, after the
close. Pulls today's bars, steps the anchor config forward one day, persists
state, appends the trade/equity logs. Refuses to double-run the same day
(the last_run_date guard in state.json).

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
        from live import fixture_data
        as_of = args.as_of or pd.Timestamp.now().normalize().isoformat()[:10]
        universe_bars = fixture_data.fetch_universe_bars(universe, as_of)
        today = pd.Timestamp(as_of).normalize().isoformat()[:10]
    else:
        from live.schwab_data import get_client, fetch_universe_bars
        client = get_client()
        universe_bars = fetch_universe_bars(client, universe)
        today = pd.Timestamp.now().normalize().isoformat()[:10]

    if state["last_run_date"] == today:
        print(f"run_daily: already ran for {today} (last_run_date matches) -- refusing to double-step.")
        return

    bars = bars_for_today(universe_bars)
    print(f"run_daily: {today} -- {len(bars)}/{len(universe)} tickers with usable bars")

    result = step_one_day(state, today, bars)

    for t in result["trades"]:
        state_mod.append_trade(t)
    state_mod.append_snapshot({
        "date": today, "cash": round(result["state"]["cash"], 2),
        "equity": round(result["equity"], 2), "n_open": result["n_open"],
        "n_pending": result["n_pending"],
    })
    state_mod.save_state(result["state"])

    print(f"run_daily: {len(result['trades'])} exits today, "
          f"{result['n_open']} open, {result['n_pending']} pending, "
          f"equity ${result['equity']:,.2f}")


if __name__ == "__main__":
    main()
