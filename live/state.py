"""The paper account's persisted state: cash, open positions, pending
wick-watches, and the last day this ran (idempotency guard -- run_daily
refuses to double-step the same trading day). One JSON file plus two
append-only JSONL logs, same shape as the wheel bot's per-account files."""
from __future__ import annotations
import json
import os

from live.config import CAPITAL
from live.paths import account_paths


def fresh_state() -> dict:
    return {
        "cash": CAPITAL,
        "open_positions": {},   # ticker -> position dict (see engine.py)
        "pending": {},          # ticker -> wick-watch dict
        "last_run_date": None,  # "YYYY-MM-DD", set after each successful step
    }


def load_state() -> dict:
    p = account_paths()
    if not os.path.exists(p["state"]):
        return fresh_state()
    with open(p["state"]) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    p = account_paths()
    os.makedirs(p["dir"], exist_ok=True)
    tmp = p["state"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p["state"])  # atomic: a crash mid-write never corrupts state


def append_trade(trade: dict) -> None:
    p = account_paths()
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["trades"], "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def append_snapshot(snapshot: dict) -> None:
    p = account_paths()
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["snapshots"], "a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")
