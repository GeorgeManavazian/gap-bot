"""The paper account's persisted state: cash, open positions, pending
wick-watches, and the last day this ran (idempotency guard -- run_daily
refuses to double-step the same trading day). One JSON file plus two
append-only JSONL logs, same shape as the wheel bot's per-account files.

The appends are idempotent (2026-10-06): a run appends its trades and
snapshot BEFORE save_state, so a crash between the two leaves the ledger
un-advanced and the re-run books the same rows again. A row whose key is
already in the tail of the log is skipped, so the re-run books it once."""
from __future__ import annotations
import glob
import json
import os

from live.config import CAPITAL
from live.paths import account_paths

# A position's identity: one position per ticker, no re-entry while open, so
# (ticker, entry_date) names it however it ends -- a crash after an intraday
# tp row whose evening re-run then books a stop is the same position twice.
TRADE_KEY = ("ticker", "entry_date")
SNAPSHOT_KEY = ("date",)
DEDUPE_TAIL = 200  # rows; a re-run only ever repeats the previous run's few rows


def fresh_state() -> dict:
    return {
        "cash": CAPITAL,
        "open_positions": {},   # ticker -> position dict (see engine.py)
        "pending": {},          # ticker -> wick-watch dict
        "last_run_date": None,  # "YYYY-MM-DD", set after each successful step
    }


def load_state(variant: str = "stock") -> dict:
    p = account_paths(variant)
    if not os.path.exists(p["state"]):
        return fresh_state()
    with open(p["state"]) as f:
        return json.load(f)


def write_atomic(path: str, payload) -> None:
    """JSON to `path` via a per-pid tmp + rename (two writers never share a
    tmp; a crash mid-write never corrupts the file). Orphaned tmp files
    from an earlier killed writer are removed first, so nothing half
    written sits in the account dir (which is mirrored publicly)."""
    for stale in glob.glob(f"{path}.tmp.*"):
        try:
            os.remove(stale)
        except OSError:
            pass
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def save_state(state: dict, variant: str = "stock") -> None:
    p = account_paths(variant)
    os.makedirs(p["dir"], exist_ok=True)
    write_atomic(p["state"], state)


def _tail_keys(path: str, key: tuple, n: int = DEDUPE_TAIL) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        lines = f.readlines()[-n:]
    keys = set()
    for ln in lines:
        try:
            row = json.loads(ln)
        except ValueError:
            continue  # a torn line can't be a duplicate of anything
        keys.add(tuple(row.get(k) for k in key))
    return keys


def _append_once(path: str, row: dict, key: tuple) -> bool:
    if tuple(row.get(k) for k in key) in _tail_keys(path, key):
        return False
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return True


def append_trade(trade: dict, variant: str = "stock") -> bool:
    """False (not written) when a row for the same position (ticker,
    entry_date) is already in the tail of trades.jsonl."""
    p = account_paths(variant)
    os.makedirs(p["dir"], exist_ok=True)
    return _append_once(p["trades"], trade, TRADE_KEY)


def append_snapshot(snapshot: dict, variant: str = "stock") -> bool:
    """False (not written) when the date is already in the tail of snapshots.jsonl."""
    p = account_paths(variant)
    os.makedirs(p["dir"], exist_ok=True)
    return _append_once(p["snapshots"], snapshot, SNAPSHOT_KEY)
