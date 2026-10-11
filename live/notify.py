"""Phone pushes for the gap bot via ntfy.sh (2026-10-11).

The tick calls these subcommands; each one exits 0 ONLY when a push was
delivered (so the shell can write a once-per-day marker on 0 and retry on
1), prints what it did, and never raises -- same rule as live/sync.py and
live/livefeed.py: nothing in the alerting path may take down the tick.

  python -m live.notify push --title T --message M [--priority P] [--tags a,b]
  python -m live.notify events <tick-output-file>     # recorded entry/exit lines
  python -m live.notify eod <tick-output-file>        # the EOD "booked ..." line
  python -m live.notify tick-failed --mode M --rc N <tick-output-file>
  python -m live.notify token-age [--token PATH]      # warn <= 3 days of runway
  python -m live.notify test

Config: ~/.gapbot/ntfy.json  {"server": "https://ntfy.sh", "topic": "<private>"}
(env GAPBOT_NTFY_CFG overrides the path). The topic is the only secret: anyone
who knows it can read and post to it, so it lives on the VPS, not in the repo.
No config -> every subcommand says so once and exits 1.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from live.tokenage import days_until_expiry, default_token_path, REFRESH_TOKEN_LIFETIME_DAYS

WARN_LEFT_DAYS = 3.0
TIMEOUT_S = 10
SSH_HINT = ("ssh -t -i ~/.ssh/wheelbot.key ubuntu@129.80.185.142 "
            "'cd /home/ubuntu/etf-bot && .venv-live/bin/python scripts/schwab/schwab_login.py'")

_ENTRY = re.compile(r"recorded entry (?P<tk>\S+) @ (?P<px>[\d.]+)")
_EXIT = re.compile(r"recorded exit (?P<tk>\S+) (?P<reason>\S+) @ (?P<px>[\d.]+) \((?P<pnl>[-+][\d.]+%)\)")
_EOD = re.compile(r"run_daily\[(?P<variant>\w+)\]: booked (?P<line>.+)")
_EOD_DATE = re.compile(r"run_daily: (?P<date>\d{4}-\d{2}-\d{2}) --")


def cfg_path() -> str:
    return os.environ.get("GAPBOT_NTFY_CFG") or os.path.expanduser("~/.gapbot/ntfy.json")


def load_cfg(path: str | None = None):
    path = path or cfg_path()
    try:
        with open(path) as f:
            c = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(c, dict) or not c.get("topic"):
        return None
    c.setdefault("server", "https://ntfy.sh")
    return c


def _post(url: str, data: bytes, headers: dict) -> int:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.status


def send(title: str, message: str, *, priority: str = "default", tags=(), cfg=None) -> bool:
    """True only when ntfy accepted the message. Never raises."""
    cfg = cfg or load_cfg()
    if cfg is None:
        print(f"notify: no config at {cfg_path()} -- not sending")
        return False
    headers = {"Title": title.encode("ascii", "replace").decode(), "Priority": priority}
    if tags:
        headers["Tags"] = ",".join(tags)
    url = f"{cfg['server'].rstrip('/')}/{cfg['topic']}"
    try:
        status = _post(url, message.encode("utf-8"), headers)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"notify: send failed: {e}")
        return False
    if 200 <= status < 300:
        print(f"notify: sent '{title}'")
        return True
    print(f"notify: send failed: HTTP {status}")
    return False


# ---------------------------------------------------------------- parsers

def parse_events(text: str) -> tuple[list, list]:
    """(entries, exits) from a tick's run_intraday output."""
    entries = [m.groupdict() for m in _ENTRY.finditer(text)]
    exits = [m.groupdict() for m in _EXIT.finditer(text)]
    return entries, exits


def events_message(entries: list, exits: list) -> tuple[str, str]:
    parts = []
    if entries:
        parts.append(f"{len(entries)} {'entry' if len(entries) == 1 else 'entries'}")
    if exits:
        parts.append(f"{len(exits)} {'exit' if len(exits) == 1 else 'exits'}")
    title = "Gap bot: " + ", ".join(parts)
    lines = [f"BUY {e['tk']} @ {float(e['px']):.2f}" for e in entries]
    lines += [f"SELL {x['tk']} @ {float(x['px']):.2f} {x['pnl']} ({x['reason']})" for x in exits]
    return title, "\n".join(lines)


def parse_eod(text: str):
    m = _EOD.search(text)
    if not m:
        return None
    d = _EOD_DATE.search(text)
    return {"date": d.group("date") if d else None, "line": "booked " + m.group("line")}


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _tail(text: str, n: int = 6) -> str:
    return "\n".join([ln for ln in text.splitlines() if ln.strip()][-n:])


# ------------------------------------------------------------ subcommands

def cmd_events(path: str) -> int:
    entries, exits = parse_events(_read(path))
    if not entries and not exits:
        print("notify: no recorded entry/exit in the tick output -- nothing to send")
        return 1
    title, msg = events_message(entries, exits)
    return 0 if send(title, msg, tags=("chart_with_upwards_trend",)) else 1


def cmd_eod(path: str) -> int:
    info = parse_eod(_read(path))
    if info is None:
        print("notify: no EOD 'booked' line in the tick output -- nothing to send")
        return 1
    title = "Gap bot EOD" + (f" {info['date']}" if info["date"] else "")
    return 0 if send(title, info["line"], tags=("moon",)) else 1


def cmd_tick_failed(mode: str, rc: str, path: str) -> int:
    body = _tail(_read(path)) or "(no output captured)"
    ok = send(f"Gap bot tick FAILED ({mode}, exit {rc})", body, priority="high", tags=("rotating_light",))
    return 0 if ok else 1


def cmd_token_age(token_path: str | None) -> int:
    """Exit 0 = warning DELIVERED, 1 = nothing to say (or delivery failed)."""
    path = token_path or default_token_path()
    left = days_until_expiry(path)
    if left is None:
        if not os.path.exists(path):
            print(f"notify: token-age: no token file at {path} -- nothing to say")
            return 1
        ok = send("Schwab token file unreadable",
                  f"{path} exists but creation_timestamp cannot be read; the 7-day clock is "
                  f"invisible. Re-login:\n{SSH_HINT}", priority="high", tags=("warning",))
        return 0 if ok else 1
    print(f"notify: token-age: {left:.2f} days left")
    if left > WARN_LEFT_DAYS:
        return 1
    if left <= 0:
        ok = send("Schwab token EXPIRED -- gap bot is blind",
                  f"Refresh token lapsed {-left:.1f} days ago. No quotes, no fills, no EOD run "
                  f"until you log in again:\n{SSH_HINT}", priority="urgent", tags=("rotating_light",))
    else:
        ok = send(f"Schwab login due in {left:.1f} days",
                  f"The refresh token dies {REFRESH_TOKEN_LIFETIME_DAYS} days after login. "
                  f"Re-login on the VPS before then:\n{SSH_HINT}", priority="high", tags=("key",))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gap bot phone pushes via ntfy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--title", required=True)
    p.add_argument("--message", required=True)
    p.add_argument("--priority", default="default")
    p.add_argument("--tags", default="")
    sub.add_parser("events").add_argument("path")
    sub.add_parser("eod").add_argument("path")
    p = sub.add_parser("tick-failed")
    p.add_argument("--mode", required=True)
    p.add_argument("--rc", required=True)
    p.add_argument("path")
    sub.add_parser("token-age").add_argument("--token", default=None)
    sub.add_parser("test")
    a = ap.parse_args(argv)

    if a.cmd == "push":
        tags = tuple(t for t in a.tags.split(",") if t)
        return 0 if send(a.title, a.message, priority=a.priority, tags=tags) else 1
    if a.cmd == "events":
        return cmd_events(a.path)
    if a.cmd == "eod":
        return cmd_eod(a.path)
    if a.cmd == "tick-failed":
        return cmd_tick_failed(a.mode, a.rc, a.path)
    if a.cmd == "token-age":
        return cmd_token_age(a.token)
    if a.cmd == "test":
        return 0 if send("Gap bot test", "Pushes from the VPS work.", tags=("white_check_mark",)) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
