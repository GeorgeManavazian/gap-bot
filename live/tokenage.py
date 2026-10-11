"""Age of the Schwab REFRESH token (copied from etf-bot/live/tokenage.py so
gap-bot stays self-contained; same token file, one Schwab login serves both).

Do not use token.json's mtime for this. schwab-py rewrites that file every
time it refreshes the short-lived ACCESS token -- on essentially every run --
so the mtime is never more than minutes old. The real issue time is the
top-level `creation_timestamp` (unix seconds), which schwab-py does not touch
on an access-token refresh. That is the field that tracks the 7-day clock."""
from __future__ import annotations
import json
import os
import time

REFRESH_TOKEN_LIFETIME_DAYS = 7


def token_age_days(path: str):
    """Age of the refresh token in days, or None if it cannot be determined
    (missing file, unreadable JSON, no creation_timestamp)."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None
    ts = d.get("creation_timestamp") if isinstance(d, dict) else None
    try:
        return (time.time() - float(ts)) / 86400.0
    except (TypeError, ValueError):
        return None


def days_until_expiry(path: str):
    """Days left before the refresh token lapses, or None if unknown."""
    age = token_age_days(path)
    return None if age is None else REFRESH_TOKEN_LIFETIME_DAYS - age


def default_token_path() -> str:
    """The token path the bot's Schwab client uses (from ~/.schwab/config.json),
    falling back to schwab-py's conventional location when there is no config
    (dev box)."""
    try:
        from live.schwab_data import load_config
        return load_config()["token_path"]
    except (SystemExit, KeyError, ValueError, OSError):
        return os.path.expanduser("~/.schwab/token.json")
