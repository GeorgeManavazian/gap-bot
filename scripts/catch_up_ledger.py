"""Catch a paper ledger up over sessions the evening run missed (outage, or
the 2026-09-12 pause) by stepping the SAME engine one session at a time from
cached daily bars -- no new rule, nothing sent anywhere. Why it exists: the
EOD engine steps ONE day per run and never backfills, so un-pausing a ledger
whose last_run_date is weeks old would age every watch/position by one day
and check stops against one bar. live/intraday.py refuses such a ledger.

PAPER ONLY, LOCAL ONLY: reads parquet bars, writes only the state dir named
by GAPBOT_STATE_DIR (default data/live). Default is a DRY RUN that prints
what would change; --apply writes state/trades/snapshots. Back up the state
dir first (the script also refuses --apply without --i-have-a-backup).

Semantics are the daily-bar backtest's (resting-limit fill, rest-top-N,
stop-first, entry-day guard), i.e. what the EOD backstop would have booked
had it run every night. It is a REPLAY, not a record of what the paused bot
did; positions "left as-is" during the pause get the exits they would have
had. Whether to replay or reset the ledger is the owner's call.

Bars must reach --through (default: latest date every ticker's longest
frame reaches). Refresh them with download_bars.py (delete stale parquets
first -- it skips cached tickers). Date handling uses pandas indexing, not
numpy datetime64 keys, so the ms-vs-us dtype bug does not apply here.

Usage (from code/gap-bot/):
  GAPBOT_STATE_DIR=<dir> python scripts/catch_up_ledger.py --variants stock [--through YYYY-MM-DD] [--apply --i-have-a-backup]
Self-check (anchor replay through this same code path must MATCH):
  python scripts/verify_catch_up.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from live import state as state_mod
from live.config import ADV_LOOKBACK, HORIZON, VARIANTS
from live.engine import step_one_day
from live.fillers import make_filler

BARS_DIR = Path(__file__).parent.parent / "data" / "daily_bars"


def load_frames(bars_dir: Path = BARS_DIR) -> dict:
    """{ticker: DataFrame indexed by Date (Timestamp) with Open/High/Low/Close/Volume
    plus precomputed trailing-ADV and prior-close columns}."""
    frames = {}
    for p in sorted(bars_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            continue
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date")
        if len(df) < HORIZON + 5:
            continue  # same eligibility rule as the backtest's load_ticker
        df = df.set_index("Date")
        dv = df["Close"] * df["Volume"]
        # look-back only: today's own volume excluded, same as TickerView.get_adv()
        df["adv"] = dv.shift(1).rolling(ADV_LOOKBACK).mean()
        df["prior_close"] = df["Close"].shift(1)
        frames[p.stem] = df
    return frames


def session_calendar(frames: dict) -> pd.DatetimeIndex:
    longest = max(frames.values(), key=len)
    return pd.DatetimeIndex(longest.index)


def bars_on(frames: dict, day: pd.Timestamp) -> dict:
    out = {}
    for tk, df in frames.items():
        if day not in df.index:
            continue
        r = df.loc[day]
        pc, adv = r["prior_close"], r["adv"]
        out[tk] = {"o": float(r["Open"]), "h": float(r["High"]), "l": float(r["Low"]), "c": float(r["Close"]),
                   "prior_close": None if pd.isna(pc) else float(pc), "adv": None if pd.isna(adv) else float(adv)}
    return out


def replay(state: dict, frames: dict, sessions, filler, chain_provider=None):
    """Step `state` through `sessions` in order. Returns (trades, snapshots)."""
    trades, snaps = [], []
    for day in sessions:
        today = day.isoformat()[:10]
        res = step_one_day(state, today, bars_on(frames, day), filler, chain_provider)
        trades.extend(res["trades"])
        snaps.append({"date": today, "cash": round(state["cash"], 2), "equity": round(res["equity"], 2),
                      "n_open": res["n_open"], "n_pending": res["n_pending"]})
    return trades, snaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="stock", help="comma-separated; stock only unless you want synthetic-chain options (not supported here)")
    ap.add_argument("--through", default=None, help="last session to replay (YYYY-MM-DD); default = last date in the bars")
    ap.add_argument("--bars-dir", default=str(BARS_DIR))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--i-have-a-backup", action="store_true")
    a = ap.parse_args()
    variants = a.variants.split(",")
    if set(variants) - {"stock"}:
        print("catch_up_ledger: only the stock variant is supported (call/spread need live chain quotes "
              "for every replayed day, which cached bars don't have).")
        return 2
    if a.apply and not a.i_have_a_backup:
        print("catch_up_ledger: --apply needs --i-have-a-backup (copy the state dir first).")
        return 2
    frames = load_frames(Path(a.bars_dir))
    cal = session_calendar(frames)
    through = pd.Timestamp(a.through) if a.through else cal[-1]
    if through > cal[-1]:
        print(f"catch_up_ledger: bars end {cal[-1].date()}, before --through {through.date()}. Refresh bars first.")
        return 2
    rc = 0
    for v in variants:
        st = state_mod.load_state(v)
        last = st["last_run_date"]
        sessions = cal[(cal > pd.Timestamp(last)) & (cal <= through)] if last else cal[cal <= through]
        if not len(sessions):
            print(f"catch_up_ledger[{v}]: nothing to replay (last_run_date {last}, through {through.date()}).")
            continue
        n_open0, n_pend0 = len(st["open_positions"]), len(st["pending"])
        trades, snaps = replay(st, frames, sessions, make_filler(v, VARIANTS[v]))
        print(f"catch_up_ledger[{v}]: {last} -> {sessions[-1].date()}: {len(sessions)} sessions, "
              f"{len(trades)} exits ({', '.join(sorted({t['exit_reason'] for t in trades})) or '-'}), "
              f"open {n_open0}->{len(st['open_positions'])}, pending {n_pend0}->{len(st['pending'])}, "
              f"equity ${snaps[-1]['equity']:,.2f}")
        if not a.apply:
            print(f"catch_up_ledger[{v}]: DRY RUN, nothing written.")
            continue
        for t in trades:
            state_mod.append_trade(t, v)
        for s in snaps:
            state_mod.append_snapshot(s, v)
        state_mod.save_state(st, v)
        print(f"catch_up_ledger[{v}]: applied.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
