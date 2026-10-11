"""The live feed for the dashboard (2026-10-11).

Every intraday poll and every EOD run, the tick script runs
`python -m live.livefeed --push`: it joins the stock account's files
(state.json, intraday.json heartbeat, snapshots.jsonl, trades.jsonl) into
ONE document, live.json, and force-pushes it as the single commit of the
`live` branch of the public state mirror. The GitHub Pages dashboard
(docs/index.html) reads that one file through the GitHub contents API.

Why a separate branch and a single amended commit: the file changes ~390
times a session; on `main` that would be 390 commits a day of noise in the
EOD history, and a branch that is always one commit never grows.

Same doctrine as live/sync.py: the credential lives at ~/.gapbot/git.json
on the VPS, never in this repo; every failure prints and returns False,
nothing here may raise into the tick; the token is scrubbed from any error
text; secret-shaped content aborts the push.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date

from live import state as state_mod
from live.config import PROFILE, bucket_name_for
from live.paths import account_paths
from live.sync import _SECRET_CONTENT, _load_cfg, _run

SCHEMA = 1
LIVE_BRANCH = "live"
LIVE_FILE = "live.json"
DEFAULT_CLONE_DIR = os.path.expanduser("~/gap-bot-live-mirror")
CLOSED_TRADES_MAX = 20


def _heartbeat_is_current(heartbeat: dict | None, snapshots: list) -> bool:
    """A heartbeat is current when it belongs to a session no snapshot has
    closed yet -- then its equity and prices are intraday marks. A heartbeat
    from an already-snapshotted session (last night's EOD ran) is history."""
    if not heartbeat or not heartbeat.get("session"):
        return False
    if not snapshots:
        return True
    return heartbeat["session"] > snapshots[-1]["date"]


def _days_between(a: str | None, b: str | None) -> int | None:
    try:
        return (date.fromisoformat(b) - date.fromisoformat(a)).days
    except (TypeError, ValueError):
        return None


def build_live_doc(state: dict, heartbeat: dict | None, snapshots: list, trades: list, *,
                   slot_limit: int, profile: str, generated_at: str,
                   token_days_left: float | None = None) -> dict:
    """Pure: the whole page model from the loaded files. `intraday` says
    whether equity/last are intraday marks (heartbeat newer than the last
    snapshot) or EOD values."""
    snapshots = sorted(snapshots, key=lambda r: r["date"])
    hb = heartbeat or {}
    intraday = _heartbeat_is_current(hb, snapshots)
    watched = (hb.get("watched") or {}) if intraday else {}
    session = hb.get("session") if intraday else state.get("last_run_date")

    if intraday:
        equity = hb.get("equity")
        if equity is None:
            equity = snapshots[-1]["equity"] if snapshots else state["cash"]
        day_start = snapshots[-1]["equity"] if snapshots else state["cash"]
    else:
        equity = snapshots[-1]["equity"] if snapshots else state["cash"]
        before = [s for s in snapshots if session is None or s["date"] < session]
        day_start = before[-1]["equity"] if before else equity

    open_positions = []
    for tk, pos in sorted(state["open_positions"].items()):
        row = watched.get(tk, {})
        last = row.get("last")
        if last is None:
            last = pos.get("last", pos["entry"])
        open_positions.append({
            "ticker": tk, "bucket": pos.get("bucket"), "gap_pct": pos.get("gap_pct"),
            "entry": pos["entry"], "last": last, "stop": pos["stop_price"], "tp": pos["tp_price"],
            "h": row.get("h"), "l": row.get("l"), "days_held": pos.get("days_held", 0),
            "entry_date": pos.get("entry_date"), "units": pos.get("units"),
        })

    pending = []
    for tk, w in sorted(state["pending"].items()):
        row = watched.get(tk, {})
        pending.append({
            "ticker": tk, "bucket": bucket_name_for(abs(w["gap_pct"])), "gap_pct": w["gap_pct"],
            "gap_open": w["gap_open"], "prior_close": w.get("prior_close"), "last": row.get("last"),
            "h": row.get("h"), "l": row.get("l"), "days_waited": w.get("days_waited", 0),
            "gap_date": w.get("gap_date"), "consumed_on": w.get("consumed_on"),
        })

    def _sort_key(t):
        return (t.get("exit_date") or "", t.get("exit_time") or "")
    trades_sorted = sorted(trades, key=_sort_key, reverse=True)

    exits_today = [t for t in trades_sorted if session and t.get("exit_date") == session]
    entries_today = [{"ticker": tk, "entry": pos["entry"], "time": pos.get("entry_time")}
                     for tk, pos in sorted(state["open_positions"].items())
                     if session and pos.get("entry_date") == session]
    entries_today += [{"ticker": t["ticker"], "entry": t.get("entry_price"), "time": t.get("entry_time")}
                      for t in exits_today if t.get("entry_date") == session]

    closed = []
    for t in trades_sorted[:CLOSED_TRADES_MAX]:
        days = t.get("days_held")
        if days is None:
            days = _days_between(t.get("entry_date"), t.get("exit_date"))
        closed.append({
            "ticker": t["ticker"], "entry_date": t.get("entry_date"), "exit_date": t.get("exit_date"),
            "exit_reason": t.get("exit_reason"), "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"), "pnl_pct": t.get("pnl_pct"),
            "pnl_dollar": t.get("pnl_dollar"), "days_held": days,
        })

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "polled_at": hb.get("polled_at"),
        "session": session,
        "last_run_date": state.get("last_run_date"),
        "profile": profile,
        "slot_limit": slot_limit,
        "intraday": intraday,
        "intraday_series": list(hb.get("equity_series") or []) if intraday else [],
        "schwab_token_days_left": None if token_days_left is None else round(token_days_left, 2),
        "equity": equity,
        "cash": state["cash"],
        "day_start_equity": day_start,
        "open_positions": open_positions,
        "pending": pending,
        "today": {
            "entries": entries_today,
            "exits": [{"ticker": t["ticker"], "exit_reason": t.get("exit_reason"),
                       "exit_price": t.get("exit_price"), "pnl_pct": t.get("pnl_pct"),
                       "time": t.get("exit_time")} for t in exits_today],
        },
        "equity_curve": [{"date": s["date"], "equity": s["equity"]} for s in snapshots],
        "closed_trades": closed,
    }


# ---------------------------------------------------------------- push

def _secret_shaped(doc: dict) -> bool:
    body = json.dumps(doc, default=str)
    return any(pat.search(body) for pat in _SECRET_CONTENT)


def push_live(doc: dict, clone_dir: str) -> bool:
    """Write live.json into clone_dir (an orphan `live` branch, created on
    first use), amend the single commit, force-push. Returns True on
    success; False with a printed reason on any failure -- never raises."""
    cfg = _load_cfg()
    if cfg is None:
        print("push_live: no credential at ~/.gapbot/git.json -- skipping")
        return False
    if _secret_shaped(doc):
        print("push_live: ABORTED, secret-shaped content in the live document")
        return False

    os.makedirs(clone_dir, exist_ok=True)
    if not os.path.exists(os.path.join(clone_dir, ".git")):
        for cmd in (["git", "init", "-q"],
                    ["git", "checkout", "-q", "--orphan", LIVE_BRANCH],
                    ["git", "config", "user.email", "gapbot@localhost"],
                    ["git", "config", "user.name", "gap-bot"]):
            r = _run(cmd, clone_dir)
            if r.returncode != 0:
                print(f"push_live: {' '.join(cmd[:2])} failed: {r.stderr}")
                return False

    state_mod.write_atomic(os.path.join(clone_dir, LIVE_FILE), doc)
    _run(["git", "add", "-A"], clone_dir)
    has_head = _run(["git", "rev-parse", "-q", "--verify", "HEAD"], clone_dir).returncode == 0
    commit = ["git", "commit", "-q", "--allow-empty", "-m", "live"] + (["--amend"] if has_head else [])
    r = _run(commit, clone_dir)
    if r.returncode != 0:
        print(f"push_live: commit failed: {r.stderr}")
        return False

    remote = f"https://x-access-token:{cfg['token']}@github.com/{cfg['repo']}.git"
    r = _run(["git", "push", "-q", "-f", remote, f"HEAD:refs/heads/{LIVE_BRANCH}"], clone_dir)
    if r.returncode != 0:
        print(f"push_live: push failed: {(r.stderr or '').replace(cfg['token'], '***')}")
        return False
    return True


# ------------------------------------------------------------- inputs, CLI

def _rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_inputs(variant: str = "stock"):
    p = account_paths(variant)
    state = state_mod.load_state(variant)
    hb_path = os.path.join(p["dir"], "intraday.json")
    heartbeat = None
    if os.path.exists(hb_path):
        with open(hb_path) as f:
            heartbeat = json.load(f)
    return state, heartbeat, _rows(p["snapshots"]), _rows(p["trades"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build (and push) the dashboard's live.json")
    ap.add_argument("--push", action="store_true", help="push to the state mirror's live branch")
    ap.add_argument("--print", action="store_true", help="print the document to stdout")
    ap.add_argument("--clone-dir", default=DEFAULT_CLONE_DIR)
    ap.add_argument("--variant", default="stock")
    args = ap.parse_args(argv)

    from live.engine import slot_limit
    from live.intraday import now_et
    from live.tokenage import days_until_expiry, default_token_path
    state, heartbeat, snapshots, trades = load_inputs(args.variant)
    doc = build_live_doc(state, heartbeat, snapshots, trades,
                         slot_limit=slot_limit(state["open_positions"], state["pending"]),
                         profile=PROFILE, generated_at=now_et().strftime("%Y-%m-%dT%H:%M:%S%z"),
                         token_days_left=days_until_expiry(default_token_path()))
    if args.print:
        print(json.dumps(doc, indent=2, default=str))
    if args.push:
        return 0 if push_live(doc, args.clone_dir) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
