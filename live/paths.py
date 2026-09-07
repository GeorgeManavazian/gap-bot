"""Single source of truth for where the live paper-account store lives.
Same env-override pattern as the wheel bot's live/paths.py, for a dashboard
to point at a synced clone instead of the canonical VPS copy."""
from __future__ import annotations
import os

DEFAULT_STATE_ROOT = "data/live"


def state_root() -> str:
    return os.environ.get("GAPBOT_STATE_DIR") or DEFAULT_STATE_ROOT


def in_state(*parts: str) -> str:
    return os.path.join(state_root(), *parts)


def account_paths() -> dict:
    """One paper account for now (the anchor config, $100k). Same
    state.json/trades.jsonl/snapshots.jsonl shape as the wheel bot's
    per-account files, so a future dashboard can share tooling."""
    d = in_state("account")
    return {"dir": d,
            "state": os.path.join(d, "state.json"),
            "trades": os.path.join(d, "trades.jsonl"),
            "snapshots": os.path.join(d, "snapshots.jsonl")}
