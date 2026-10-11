# Live Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public GitHub Pages dashboard that shows, within ~2 minutes, exactly what the gap bot's intraday poller sees: watched names vs wick levels, positions marked to market, today's fills and exits.

**Architecture:** The VPS tick, after every intraday poll and every EOD run, runs `live/livefeed.py --push`, which joins the account files into one `live.json` and force-pushes it as the single commit of the `live` branch of the public `gap-bot-state` repo. A static page in `docs/` of the `gap-bot` repo (GitHub Pages) polls that file through the GitHub contents API every 75s during market hours with ETag conditionals and rate-limit backoff.

**Tech Stack:** Python 3 (etf-bot venv, pandas 3 on the VPS), bash (systemd tick), git, vanilla HTML/JS/SVG, GitHub Pages, GitHub REST contents API.

**Spec:** `docs/superpowers/specs/2026-10-11-live-dashboard-design.md`

## Global Constraints

- Nothing in the live path may raise into the tick or change its exit code; the trading decision is already persisted before the push runs.
- Credential stays at `~/.gapbot/git.json` on the VPS, loaded by `live.sync._load_cfg`; the token never appears in output (scrub with `.replace(token, "***")`).
- `live.json` is built only from `data/live/account/` files; the existing secret patterns in `live.sync._SECRET_CONTENT` are checked before every push.
- The `live` branch is always exactly one commit (amend + force-push).
- Page: one file `docs/index.html`, no libraries, Google Fonts only, works at 400px width, dark-first with a light theme.
- Poll cadence: 75s weekdays 09:25–16:10 ET, else 600s; back off under 8 remaining requests.
- All tests run with `../etf-bot/.venv/bin/python -m pytest` from the repo root.
- Commit after every task with the attribution line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. Heartbeat from a previous session (morning before the first poll): `last` prices and equity must come from the EOD state/snapshot, never from yesterday's poll. Pinned in Task 1 (`test_build_doc_ignores_stale_heartbeat`).
2. No heartbeat at all (fresh deploy, or the EOD-only weekend): the doc builds with `polled_at: null`, equity from the last snapshot. Pinned in Task 1 (`test_build_doc_without_heartbeat`).
3. A push failure every minute (dead token) must produce one log line per session, not 390, and the next success must re-arm the warning. Pinned in Task 3 (`test_tick_livefeed_failure_logs_once_per_session`).
4. The page losing the API (offline, 403 rate limit) must keep the last data on screen and say so, never blank. Pinned in Task 4 (manual fixture check with `?src=` pointing at a 404 URL).
5. A timestamp like `2026-10-13T10:32:02-0400` (no colon in the offset) must parse in Safari as well as Chrome. Pinned in Task 4 (`parseStamp` normalises the offset; checked in the fixture render).

---

### Task 1: `build_live_doc` — the page model

**Files:**
- Create: `live/livefeed.py`
- Create: `scripts/test_livefeed.py`

**Interfaces:**
- Consumes: `live.config.bucket_name_for(abs_gap) -> str|None`, `live.config.BUCKETS`.
- Produces: `build_live_doc(state: dict, heartbeat: dict|None, snapshots: list[dict], trades: list[dict], *, slot_limit: int, profile: str, generated_at: str) -> dict` with the keys listed in the spec plus `"intraday": bool` (true when the heartbeat is newer than the last snapshot, i.e. `equity` and `last` are intraday marks).

- [ ] **Step 1: Write the failing tests**

```python
# scripts/test_livefeed.py
"""Tests for live/livefeed.py: the per-minute live.json the dashboard reads
(build_live_doc), its push to the `live` branch of the state mirror
(push_live), and the CLI the tick script calls.

Run: ../etf-bot/.venv/bin/python -m pytest scripts/test_livefeed.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from live import livefeed  # noqa: E402

GEN = "2026-10-13T10:32:07-0400"


def _state():
    return {
        "cash": 71234.10,
        "open_positions": {
            "LULU": {"ticker": "LULU", "entry": 98.25, "stop_price": 49.12, "tp_price": 121.8,
                     "days_held": 2, "units": 50.0, "bucket": "10%+", "gap_pct": -19.4,
                     "gap_date": "2026-10-08", "entry_date": "2026-10-09", "last": 101.0},
            "NEW": {"ticker": "NEW", "entry": 50.0, "stop_price": 25.0, "tp_price": 52.0,
                    "days_held": 0, "units": 10.0, "bucket": "3-5%", "gap_pct": -3.9,
                    "gap_date": "2026-10-10", "entry_date": "2026-10-13", "entry_time": "2026-10-13T09:47:00-0400"},
        },
        "pending": {
            "T": {"gap_open": 22.38, "prior_close": 24.87, "gap_pct": -10.01, "gap_date": "2026-10-09",
                  "days_waited": 1},
            "CCL": {"gap_open": 25.61, "prior_close": 26.15, "gap_pct": -2.07, "gap_date": "2026-10-08",
                    "days_waited": 2, "consumed_on": "2026-10-13"},
        },
        "last_run_date": "2026-10-10",
    }


def _heartbeat():
    return {
        "polled_at": "2026-10-13T10:32:02-0400", "session": "2026-10-13", "profile": "ramp",
        "equity": 100412.55, "slot_limit": 7, "exits": ["OLD"], "entries": ["NEW"],
        "watched": {
            "LULU": {"kind": "open", "stop": 49.12, "tp": 121.8, "h": 103.0, "l": 99.5, "last": 102.2},
            "NEW": {"kind": "open", "stop": 25.0, "tp": 52.0, "h": 51.0, "l": 49.0, "last": 50.5},
            "T": {"kind": "pending", "gap_open": 22.38, "gap_date": "2026-10-09", "consumed_on": None,
                  "h": 22.3, "l": 21.9, "last": 22.1},
            "CCL": {"kind": "pending", "gap_open": 25.61, "gap_date": "2026-10-08", "consumed_on": "2026-10-13",
                    "h": 25.9, "l": 25.3, "last": 25.7},
        },
    }


def _snapshots():
    return [
        {"date": "2026-10-08", "cash": 100000.0, "equity": 100000.0, "n_open": 0, "n_pending": 18},
        {"date": "2026-10-09", "cash": 95087.5, "equity": 100050.0, "n_open": 1, "n_pending": 15},
        {"date": "2026-10-10", "cash": 95087.5, "equity": 100120.0, "n_open": 1, "n_pending": 14},
    ]


def _trades():
    return [
        {"ticker": "AAA", "gap_date": "2026-10-01", "entry_date": "2026-10-02", "exit_date": "2026-10-06",
         "entry_price": 10.0, "exit_price": 11.0, "exit_reason": "tp_gap_filled", "pnl_pct": 10.0,
         "pnl_dollar": 100.0, "days_held": 2, "units": 10.0, "bucket": "2-3%", "gap_pct": -2.5},
        {"ticker": "OLD", "gap_date": "2026-10-07", "entry_date": "2026-10-08", "exit_date": "2026-10-13",
         "entry_price": 40.0, "exit_price": 38.0, "exit_reason": "time_63d", "pnl_pct": -5.0,
         "pnl_dollar": -200.0, "days_held": 3, "exit_time": "2026-10-13T10:15:00-0400"},
    ]


def test_build_doc_joins_everything_for_an_intraday_poll():
    doc = livefeed.build_live_doc(_state(), _heartbeat(), _snapshots(), _trades(),
                                  slot_limit=7, profile="ramp", generated_at=GEN)
    assert doc["schema"] == 1 and doc["generated_at"] == GEN
    assert doc["polled_at"] == "2026-10-13T10:32:02-0400" and doc["session"] == "2026-10-13"
    assert doc["last_run_date"] == "2026-10-10" and doc["profile"] == "ramp" and doc["slot_limit"] == 7
    assert doc["intraday"] is True
    assert doc["equity"] == 100412.55 and doc["cash"] == 71234.10
    assert doc["day_start_equity"] == 100120.0          # last snapshot before the session
    pos = {p["ticker"]: p for p in doc["open_positions"]}
    assert pos["LULU"] == {"ticker": "LULU", "bucket": "10%+", "gap_pct": -19.4, "entry": 98.25,
                           "last": 102.2, "stop": 49.12, "tp": 121.8, "h": 103.0, "l": 99.5,
                           "days_held": 2, "entry_date": "2026-10-09", "units": 50.0}
    pend = {w["ticker"]: w for w in doc["pending"]}
    assert pend["T"] == {"ticker": "T", "bucket": "10%+", "gap_pct": -10.01, "gap_open": 22.38,
                         "prior_close": 24.87, "last": 22.1, "h": 22.3, "l": 21.9, "days_waited": 1,
                         "gap_date": "2026-10-09", "consumed_on": None}
    assert pend["CCL"]["bucket"] == "2-3%" and pend["CCL"]["consumed_on"] == "2026-10-13"
    assert doc["today"]["entries"] == [{"ticker": "NEW", "entry": 50.0, "time": "2026-10-13T09:47:00-0400"}]
    assert doc["today"]["exits"] == [{"ticker": "OLD", "exit_reason": "time_63d", "exit_price": 38.0,
                                      "pnl_pct": -5.0, "time": "2026-10-13T10:15:00-0400"}]
    assert doc["equity_curve"] == [{"date": "2026-10-08", "equity": 100000.0},
                                   {"date": "2026-10-09", "equity": 100050.0},
                                   {"date": "2026-10-10", "equity": 100120.0}]
    assert [t["ticker"] for t in doc["closed_trades"]] == ["OLD", "AAA"]   # newest first
    assert doc["closed_trades"][1] == {"ticker": "AAA", "entry_date": "2026-10-02", "exit_date": "2026-10-06",
                                       "exit_reason": "tp_gap_filled", "entry_price": 10.0, "exit_price": 11.0,
                                       "pnl_pct": 10.0, "pnl_dollar": 100.0, "days_held": 2}


def test_build_doc_ignores_stale_heartbeat():
    """Morning before the first poll: yesterday's heartbeat must not supply
    prices or equity. Session is the last EOD date, marks come from state."""
    hb = {**_heartbeat(), "session": "2026-10-10", "polled_at": "2026-10-10T16:00:02-0400", "equity": 99000.0}
    doc = livefeed.build_live_doc(_state(), hb, _snapshots(), _trades(),
                                  slot_limit=6, profile="ramp", generated_at=GEN)
    assert doc["intraday"] is False
    assert doc["session"] == "2026-10-10" and doc["polled_at"] == "2026-10-10T16:00:02-0400"
    assert doc["equity"] == 100120.0                    # last snapshot, not the heartbeat
    assert doc["day_start_equity"] == 100050.0          # the snapshot before that
    pos = {p["ticker"]: p for p in doc["open_positions"]}
    assert pos["LULU"]["last"] == 101.0 and pos["LULU"]["h"] is None and pos["LULU"]["l"] is None
    assert pos["NEW"]["last"] == 50.0                   # no stored last -> entry
    pend = {w["ticker"]: w for w in doc["pending"]}
    assert pend["T"]["last"] is None
    assert doc["today"] == {"entries": [], "exits": []}  # 2026-10-10 had none


def test_build_doc_without_heartbeat():
    doc = livefeed.build_live_doc(_state(), None, _snapshots(), _trades(),
                                  slot_limit=6, profile="anchor", generated_at=GEN)
    assert doc["polled_at"] is None and doc["intraday"] is False
    assert doc["session"] == "2026-10-10" and doc["equity"] == 100120.0


def test_build_doc_with_no_history_falls_back_to_cash():
    s = _state()
    s["open_positions"], s["last_run_date"] = {}, None
    doc = livefeed.build_live_doc(s, None, [], [], slot_limit=6, profile="anchor", generated_at=GEN)
    assert doc["equity"] == doc["day_start_equity"] == s["cash"]
    assert doc["session"] is None and doc["equity_curve"] == [] and doc["closed_trades"] == []


def test_closed_trades_capped_at_twenty_and_days_held_derived():
    trades = [{"ticker": f"T{i:02d}", "entry_date": "2026-09-01", "exit_date": f"2026-09-{i + 2:02d}",
               "exit_reason": "tp_gap_filled", "entry_price": 1.0, "exit_price": 1.1, "pnl_pct": 10.0,
               "pnl_dollar": 1.0} for i in range(25)]
    doc = livefeed.build_live_doc(_state(), None, _snapshots(), trades,
                                  slot_limit=6, profile="anchor", generated_at=GEN)
    assert len(doc["closed_trades"]) == 20
    assert doc["closed_trades"][0]["ticker"] == "T24"
    assert doc["closed_trades"][0]["days_held"] == 25     # calendar days when the row lacks it
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `../etf-bot/.venv/bin/python -m pytest scripts/test_livefeed.py -q`
Expected: ImportError / `cannot import name 'livefeed'`.

- [ ] **Step 3: Write `build_live_doc`**

```python
# live/livefeed.py
"""The live feed for the dashboard (2026-10-11).

Every intraday poll and every EOD run, the tick script runs
`python -m live.livefeed --push`: it joins the stock account's files
(state.json, intraday.json heartbeat, snapshots.jsonl, trades.jsonl) into
ONE document, live.json, and force-pushes it as the single commit of the
`live` branch of the public state mirror. The GitHub Pages dashboard
(docs/index.html) reads that one file through the contents API.

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
                   slot_limit: int, profile: str, generated_at: str) -> dict:
    """Pure: the whole page model from the loaded files. See the spec for
    the field list; `intraday` says whether equity/last are intraday marks
    (heartbeat newer than the last snapshot) or EOD values."""
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../etf-bot/.venv/bin/python -m pytest scripts/test_livefeed.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add live/livefeed.py scripts/test_livefeed.py
git commit -m "livefeed: build the dashboard's live.json from the account files

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `push_live` and the CLI

**Files:**
- Modify: `live/livefeed.py` (append)
- Modify: `scripts/test_livefeed.py` (append)

**Interfaces:**
- Consumes: `live.sync._load_cfg() -> dict|None`, `live.sync._run(cmd, cwd) -> CompletedProcess`, `live.sync._SECRET_CONTENT`, `live.state.write_atomic(path, payload)`, `live.state.load_state(variant)`, `live.paths.account_paths(variant)`, `live.engine.slot_limit(open_positions, pending)`, `live.intraday.now_et()`.
- Produces: `push_live(doc: dict, clone_dir: str) -> bool`; `load_inputs(variant="stock") -> tuple[state, heartbeat, snapshots, trades]`; `main(argv) -> int` for `python -m live.livefeed [--push] [--print] [--clone-dir DIR]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to scripts/test_livefeed.py

class _R:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _git_stub(cmds, *, has_head=True, push_rc=0, init_rc=0):
    def run(cmd, cwd):
        cmds.append(cmd)
        if cmd[:2] == ["git", "init"]:
            return _R(init_rc)
        if cmd[:2] == ["git", "rev-parse"]:
            return _R(0 if has_head else 1, "abc\n")
        if cmd[:2] == ["git", "push"]:
            return _R(push_rc, err="remote: Invalid username or token TOK123 rejected\n")
        return _R(0)
    return run


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(livefeed, "_load_cfg", lambda: {"repo": "x/y", "token": "TOK123", "user": "u"})


def test_push_live_first_use_inits_orphan_branch_and_pushes(tmp_path, cfg, monkeypatch):
    cmds = []
    monkeypatch.setattr(livefeed, "_run", _git_stub(cmds, has_head=False))
    clone = tmp_path / "mirror"
    assert livefeed.push_live({"schema": 1, "equity": 1.0}, str(clone)) is True
    assert json.load(open(clone / "live.json")) == {"schema": 1, "equity": 1.0}
    assert ["git", "init", "-q"] in cmds
    assert ["git", "checkout", "-q", "--orphan", "live"] in cmds
    assert ["git", "add", "-A"] in cmds
    commit = [c for c in cmds if c[:2] == ["git", "commit"]][0]
    assert "--amend" not in commit and "--allow-empty" in commit        # no HEAD yet
    push = [c for c in cmds if c[:2] == ["git", "push"]][0]
    assert push[2:4] == ["-q", "-f"] and push[-1] == "HEAD:refs/heads/live"
    assert "x-access-token:TOK123@github.com/x/y" in push[4]


def test_push_live_existing_clone_amends(tmp_path, cfg, monkeypatch):
    cmds = []
    monkeypatch.setattr(livefeed, "_run", _git_stub(cmds, has_head=True))
    clone = tmp_path / "mirror"
    (clone / ".git").mkdir(parents=True)
    assert livefeed.push_live({"schema": 1}, str(clone)) is True
    assert not any(c[:2] == ["git", "init"] for c in cmds)
    commit = [c for c in cmds if c[:2] == ["git", "commit"]][0]
    assert "--amend" in commit


def test_push_live_without_credential_does_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(livefeed, "_load_cfg", lambda: None)
    cmds = []
    monkeypatch.setattr(livefeed, "_run", _git_stub(cmds))
    assert livefeed.push_live({"schema": 1}, str(tmp_path / "m")) is False
    assert cmds == [] and "no credential" in capsys.readouterr().out


def test_push_live_failed_push_scrubs_token(tmp_path, cfg, monkeypatch, capsys):
    cmds = []
    monkeypatch.setattr(livefeed, "_run", _git_stub(cmds, push_rc=1))
    assert livefeed.push_live({"schema": 1}, str(tmp_path / "m")) is False
    out = capsys.readouterr().out
    assert "push failed" in out and "TOK123" not in out and "***" in out


def test_push_live_refuses_secret_shaped_content(tmp_path, cfg, monkeypatch, capsys):
    cmds = []
    monkeypatch.setattr(livefeed, "_run", _git_stub(cmds))
    doc = {"note": "github_pat_" + "A" * 30}
    assert livefeed.push_live(doc, str(tmp_path / "m")) is False
    assert not any(c[:2] == ["git", "push"] for c in cmds)
    assert "secret-shaped" in capsys.readouterr().out


def test_load_inputs_reads_the_account_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GAPBOT_STATE_DIR", str(tmp_path))
    acct = tmp_path / "account"
    acct.mkdir()
    (acct / "state.json").write_text(json.dumps(_state()))
    (acct / "intraday.json").write_text(json.dumps(_heartbeat()))
    (acct / "snapshots.jsonl").write_text("\n".join(json.dumps(s) for s in _snapshots()) + "\n")
    (acct / "trades.jsonl").write_text("\n".join(json.dumps(t) for t in _trades()) + "\n")
    state, hb, snaps, trades = livefeed.load_inputs("stock")
    assert state["cash"] == 71234.10 and hb["session"] == "2026-10-13"
    assert len(snaps) == 3 and len(trades) == 2


def test_load_inputs_tolerates_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("GAPBOT_STATE_DIR", str(tmp_path))
    state, hb, snaps, trades = livefeed.load_inputs("stock")
    assert state["cash"] > 0 and hb is None and snaps == [] and trades == []


def test_cli_print_emits_the_doc_and_push_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GAPBOT_STATE_DIR", str(tmp_path))
    assert livefeed.main(["--print"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == 1 and doc["profile"] in ("anchor", "ramp")
    monkeypatch.setattr(livefeed, "push_live", lambda doc, clone_dir: True)
    assert livefeed.main(["--push", "--clone-dir", str(tmp_path / "m")]) == 0
    monkeypatch.setattr(livefeed, "push_live", lambda doc, clone_dir: False)
    assert livefeed.main(["--push", "--clone-dir", str(tmp_path / "m")]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `../etf-bot/.venv/bin/python -m pytest scripts/test_livefeed.py -q`
Expected: the 8 new tests fail with `AttributeError: module 'live.livefeed' has no attribute 'push_live'` (and `load_inputs`, `main`).

- [ ] **Step 3: Write `push_live`, `load_inputs`, `main`**

```python
# append to live/livefeed.py

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
    state, heartbeat, snapshots, trades = load_inputs(args.variant)
    doc = build_live_doc(state, heartbeat, snapshots, trades,
                         slot_limit=slot_limit(state["open_positions"], state["pending"]),
                         profile=PROFILE, generated_at=now_et().strftime("%Y-%m-%dT%H:%M:%S%z"))
    if args.print:
        print(json.dumps(doc, indent=2, default=str))
    if args.push:
        return 0 if push_live(doc, args.clone_dir) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../etf-bot/.venv/bin/python -m pytest scripts/test_livefeed.py -q`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add live/livefeed.py scripts/test_livefeed.py
git commit -m "livefeed: push live.json as the single commit of the state mirror's live branch

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Heartbeat `slot_limit` and the tick wiring

**Files:**
- Modify: `live/run_intraday.py:117-121` (heartbeat payload)
- Modify: `scripts/gap_bot_tick.sh:118-150` (tail: logging, livefeed step, main sync)
- Modify: `scripts/test_intraday_blockers.py` (append tick tests)

**Interfaces:**
- Consumes: `python -m live.livefeed --push` exit code 0/1 (Task 2); `engine.slot_limit`.
- Produces: `intraday.json` gains `"slot_limit": int`; tick.log gains at most one `livefeed push FAILED` line per session; marker file `data/live/logs/.livefeed_fail_session`.

- [ ] **Step 1: Write the failing tick tests**

```python
# append to scripts/test_intraday_blockers.py

# ------------------------------------------------ 9. live feed (2026-10-11)

def _argfile_stub(tmp_path, argfile, fail_livefeed=False):
    body = f'echo "$@" >> "{argfile}"\n'
    if fail_livefeed:
        body += 'case "$*" in *live.livefeed*) echo "push_live: push failed: boom"; exit 1;; esac\n'
    return _stub(tmp_path, "py_lf" + ("_fail" if fail_livefeed else ""), body)


def test_tick_pushes_live_feed_on_quiet_polls_and_eod(tmp_path):
    repo, env = tmp_path / "repo", {"PATH": _flock_on_path(tmp_path)}
    argfile = tmp_path / "args"
    py = _argfile_stub(tmp_path, argfile)
    for hm in (1000, 1001, 1800):
        rc, log = _tick(repo, 2, hm, py=py, extra=env)
        assert rc == 0
    pushes = [ln for ln in argfile.read_text().splitlines() if ln.endswith("-m live.livefeed --push")]
    assert len(pushes) == 3                              # first poll, quiet poll, EOD
    assert "livefeed" not in log                         # success is silent


def test_tick_livefeed_failure_logs_once_per_session(tmp_path):
    repo, env = tmp_path / "repo", {"PATH": _flock_on_path(tmp_path)}
    argfile = tmp_path / "args"
    bad = _argfile_stub(tmp_path, argfile, fail_livefeed=True)
    good = _argfile_stub(tmp_path, argfile)
    rc, log1 = _tick(repo, 2, 1000, py=bad, extra=env)
    assert rc == 0                                       # a dead feed never fails the tick
    assert log1.count("livefeed push FAILED") == 1 and "push_live: push failed: boom" in log1
    rc, log2 = _tick(repo, 2, 1001, py=bad, extra=env)
    assert rc == 0 and log2 == log1                      # silent on the second failure
    rc, log3 = _tick(repo, 2, 1002, py=good, extra=env)
    assert rc == 0 and log3 == log1                      # recovery: marker cleared, nothing logged
    rc, log4 = _tick(repo, 2, 1003, py=bad, extra=env)
    assert rc == 0 and log4.count("livefeed push FAILED") == 2   # re-armed after a success


def test_tick_livefeed_runs_under_a_timeout_when_available(tmp_path):
    repo = tmp_path / "repo"
    argfile = tmp_path / "args"
    fake_timeout = _stub(tmp_path, "timeout", f'echo "timeout $1" >> "{argfile}"; shift; exec "$@"\n')
    env = {"PATH": f"{tmp_path}:{_flock_on_path(tmp_path)}"}
    rc, _ = _tick(repo, 2, 1000, py=_argfile_stub(tmp_path, argfile), extra=env)
    assert rc == 0
    lines = argfile.read_text().splitlines()
    assert "timeout 25" in lines                          # GAPBOT_LIVE_TIMEOUT default


def test_heartbeat_carries_slot_limit(state_dir, monkeypatch, capsys):
    import live.run_intraday as ri
    s = vi.fresh()
    s["pending"] = {"TCH": {"gap_open": 95.0, "prior_close": 100.0, "gap_pct": -5.0,
                            "gap_date": "2026-09-01", "days_waited": 1}}
    s["last_run_date"] = "2026-09-01"
    state_mod.save_state(s, "stock")
    snap = {"TCH": {"o": 94, "h": 94.5, "l": 93, "c": 94.2}, "SPY": {"o": 500, "h": 501, "l": 499, "c": 500}}
    ri.run_variant("stock", "2026-09-02", snap, vi.FakeClient(), None, POLL_AT, dry_run=False)
    hb = json.load(open(os.path.join(state_mod.account_paths("stock")["dir"], "intraday.json")))
    from live.engine import slot_limit
    assert hb["slot_limit"] == slot_limit(s["open_positions"], s["pending"])
```

Note for the implementer: `vi.FakeClient` is the fake Schwab client in
`scripts/verify_intraday.py`; if its name differs, use the class that
`verify_intraday` passes as `client` to `run_variant`, and `adv_for` must be
satisfiable — the simplest is to pass a snapshot where nothing touches
(`h` below `gap_open`), so no ADV lookup happens, as the snap above does.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `../etf-bot/.venv/bin/python -m pytest scripts/test_intraday_blockers.py -q -k "livefeed or live_feed or slot_limit"`
Expected: 4 failed (no `live.livefeed` push in the arg file; no `slot_limit` key).

- [ ] **Step 3: Add `slot_limit` to the heartbeat**

In `live/run_intraday.py`, the `write_heartbeat(variant, {...})` call becomes:

```python
    write_heartbeat(variant, {
        "polled_at": stamp, "session": today, "profile": PROFILE, "equity": round(result["equity"], 2),
        "slot_limit": slot_limit(state["open_positions"], state["pending"]),
        "exits": [t["ticker"] for t in result["trades"]], "entries": result["filled"],
        "watched": watch_table(state, snap),
    })
```

and `from live.engine import slot_limit` joins the imports at the top (check
whether `engine` is already imported there; if `engine.slot_limit` is already
referenced at line ~288 of `live/intraday.py`, import it the same way).

- [ ] **Step 4: Rewrite the tick script's tail**

Replace everything in `scripts/gap_bot_tick.sh` from the line
`# run_intraday.py prints "recorded exit"/"recorded entry" for a real event.`
to the end of the file with:

```bash
# run_intraday.py prints "recorded exit"/"recorded entry" for a real event.
EVENT=0
if [ "$MODE" = intraday ] && grep -q 'recorded \(exit\|entry\)' "$OUT"; then
    EVENT=1
fi
# Quiet no-op poll after the session's first: nothing logged. Everything
# else (EOD, an event, the first poll) logs its output and "tick ok".
if [ "$MODE" = eod ] || [ "$EVENT" = 1 ] || [ "$FIRST" = 1 ]; then
    cat "$OUT" >> "$LOGDIR/tick.log"
    log "tick ok ($MODE)"
    [ "$FIRST" = 1 ] && ET +%F > "$SESSION_MARK"
fi
rm -f "$OUT"

# Live feed (2026-10-11): after EVERY successful poll, quiet or not, and
# after the EOD run, republish live.json to the state repo's `live` branch
# (one amended commit, so the repo never grows) for the GitHub Pages
# dashboard. Best-effort and capped: a hung push must not hold the lock
# into the next minute. Failure is logged once per session -- a dead token
# is one line a day, not 390 -- and re-armed by the next success.
LIVE_FAIL_MARK="$LOGDIR/.livefeed_fail_session"
LIVE_OUT="$(mktemp)"
LIVE_CMD=(env PYTHONPATH="$REPO" "$PY" -m live.livefeed --push)
if command -v timeout >/dev/null 2>&1; then
    LIVE_CMD=(timeout "${GAPBOT_LIVE_TIMEOUT:-25}" "${LIVE_CMD[@]}")
fi
if "${LIVE_CMD[@]}" > "$LIVE_OUT" 2>&1; then
    rm -f "$LIVE_FAIL_MARK"
else
    if [ "$(cat "$LIVE_FAIL_MARK" 2>/dev/null)" != "$(ET +%F)" ]; then
        cat "$LIVE_OUT" >> "$LOGDIR/tick.log"
        log "livefeed push FAILED (first failure this session; silent until it recovers)"
        ET +%F > "$LIVE_FAIL_MARK"
    fi
fi
rm -f "$LIVE_OUT"

# An intraday poll that recorded nothing changed nothing worth mirroring to
# `main` (only its heartbeat file, which the live feed above already
# carries) -- skip the history sync so the public state repo doesn't
# collect ~390 no-op commits a day. The EOD run always syncs.
if [ "$MODE" = intraday ] && [ "$EVENT" = 0 ]; then
    exit 0
fi

# Sync is best-effort and must never fail the tick -- the trading decision
# already happened and is already persisted locally; a sync problem is a
# dashboard-freshness issue, not a trading one.
PYTHONPATH="$REPO" "$PY" -c "
from live.sync import sync_state
ok = sync_state('$REPO/data/live', '/home/ubuntu/gap-bot-state-mirror')
print('sync ' + ('ok' if ok else 'skipped/failed'))
" >> "$LOGDIR/tick.log" 2>&1 || log "sync step raised, continuing anyway"
```

Also extend the header comment block of the script (the "Logging" paragraph)
with one sentence: `Every successful poll and EOD run then pushes live.json
(live/livefeed.py); only its first failure of a session is logged.`

- [ ] **Step 5: Run the whole intraday test file**

Run: `../etf-bot/.venv/bin/python -m pytest scripts/test_intraday_blockers.py scripts/test_livefeed.py -q`
Expected: all pass (the earlier tick tests use `/bin/echo` as the python, so the livefeed step "succeeds" silently and their log assertions still hold).

- [ ] **Step 6: Run the gates that touch the poller**

Run: `GAPBOT_PROFILE=ramp ../etf-bot/.venv/bin/python scripts/verify_intraday.py` and the same under `GAPBOT_PROFILE=anchor`.
Expected: `ALL PASS` both.

- [ ] **Step 7: Commit**

```bash
git add live/run_intraday.py scripts/gap_bot_tick.sh scripts/test_intraday_blockers.py
git commit -m "tick: push the live feed after every poll and EOD run; heartbeat carries slot_limit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The page — `docs/index.html`

**Files:**
- Create: `docs/index.html`
- Create: `docs/fixtures/live.json` (a sample document for local checks; committed, marked as example data in its `generated_at`)

**Interfaces:**
- Consumes: `live.json` schema 1 from Task 1 via `https://api.github.com/repos/GeorgeManavazian/gap-bot-state/contents/live.json?ref=live` with `Accept: application/vnd.github.raw+json`.
- Produces: the public page. `?src=<url>` overrides the data URL (local checks only).

- [ ] **Step 1: Generate the fixture from the real builder**

```bash
cd /Users/georgiemanavazian/Documents/Trading/code/gap-bot
mkdir -p docs/fixtures
../etf-bot/.venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import test_livefeed as t
from live import livefeed
doc = livefeed.build_live_doc(t._state(), t._heartbeat(), t._snapshots(), t._trades(),
                              slot_limit=7, profile="ramp", generated_at="2026-10-13T10:32:07-0400 (EXAMPLE DATA)")
json.dump(doc, open("docs/fixtures/live.json", "w"), indent=2)
print("ok", len(doc["pending"]), "pending", len(doc["open_positions"]), "open")
EOF
```

- [ ] **Step 2: Write the page**

Write `docs/index.html` with this content. It is a complete document (GitHub
Pages serves it as-is, so it carries its own doctype and head, unlike a Claude
artifact).

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Gap-Fill Bot</title>
<meta name="description" content="Live paper-trading view of the S&P 500 gap-fill bot: what it watches, what it holds, what it did today.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;500;700;800&family=Roboto+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* Layout: a terminal-style tape. Dark-first (the bot's existing identity), light theme mirrored. */
:root {
  --bg: #0e1012; --surface: #161b1e; --surface-2: #1d2529; --border: rgba(255,255,255,0.08);
  --fg: #e6e9e9; --fg-2: #9aa0a2; --fg-3: #646b6e;
  --accent: #9be32d; --accent-wash: rgba(155,227,45,0.12);
  --neg: #ef5a55; --neg-wash: rgba(239,90,85,0.12);
  --warn: #f2b53a; --warn-wash: rgba(242,181,58,0.14);
  --display: "Roboto Condensed", "Arial Narrow", system-ui, sans-serif;
  --mono: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  color-scheme: dark;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg: #f3f4f1; --surface: #ffffff; --surface-2: #e9ece6; --border: rgba(0,0,0,0.1);
    --fg: #15181a; --fg-2: #4d5457; --fg-3: #7d8487;
    --accent: #4c8a06; --accent-wash: rgba(76,138,6,0.12);
    --neg: #c62f2a; --neg-wash: rgba(198,47,42,0.1);
    --warn: #a86d05; --warn-wash: rgba(168,109,5,0.14);
    color-scheme: light;
  }
}
:root[data-theme="light"] {
  --bg: #f3f4f1; --surface: #ffffff; --surface-2: #e9ece6; --border: rgba(0,0,0,0.1);
  --fg: #15181a; --fg-2: #4d5457; --fg-3: #7d8487;
  --accent: #4c8a06; --accent-wash: rgba(76,138,6,0.12);
  --neg: #c62f2a; --neg-wash: rgba(198,47,42,0.1);
  --warn: #a86d05; --warn-wash: rgba(168,109,5,0.14);
  color-scheme: light;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--fg); }
body { font: 14px/1.4 var(--display); padding-block: 20px 48px; padding-inline: 16px; }
.page { max-width: 1240px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.pos { color: var(--accent); } .neg { color: var(--neg); } .muted { color: var(--fg-3); }

header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 10px 24px; padding-bottom: 14px; border-bottom: 1px solid var(--border); }
.brand { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 14px; }
.wordmark { font-size: 28px; font-weight: 800; letter-spacing: .01em; text-transform: uppercase; }
.wordmark b { color: var(--accent); }
.sub { font-size: 12px; color: var(--fg-2); text-transform: uppercase; letter-spacing: .08em; }
.status { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; font-size: 12.5px; color: var(--fg-2); }
.pill { display: inline-flex; align-items: center; gap: 7px; padding: 4px 10px; border-radius: 999px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; font-size: 12px; background: var(--surface-2); color: var(--fg-2); }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.live { background: var(--accent-wash); color: var(--accent); }
.pill.live::before { box-shadow: 0 0 8px currentColor; animation: pulse 2s ease-in-out infinite; }
.pill.stale { background: var(--warn-wash); color: var(--warn); }
@keyframes pulse { 50% { opacity: .35; } }
@media (prefers-reduced-motion: reduce) { .pill.live::before { animation: none; } }

.tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
@media (max-width: 900px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
.tile { background: var(--surface); padding: 14px 16px; min-width: 0; }
.tile .label { font-size: 11px; color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.tile .value { font-size: 22px; font-weight: 600; }
.tile .sub { font-size: 12px; color: var(--fg-2); text-transform: none; letter-spacing: 0; margin-top: 3px; }

.row { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; align-items: start; }
@media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; min-width: 0; }
.card-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 12px; margin-bottom: 12px; }
.card-title { font-size: 13px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.card-title .count { color: var(--fg-3); font-weight: 500; letter-spacing: 0; text-transform: none; margin-left: 6px; }
.card-note { font-size: 11.5px; color: var(--fg-3); }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: right; padding: 5px 8px; font: 600 10.5px var(--mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border); white-space: nowrap; }
td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
tr:last-child td { border-bottom: none; }
.tk { font-weight: 700; }
.badge { display: inline-block; font: 600 10px var(--mono); padding: 2px 6px; border-radius: 4px; background: var(--accent-wash); color: var(--accent); margin-left: 6px; vertical-align: middle; }
.bar { width: 64px; height: 5px; border-radius: 3px; background: var(--surface-2); display: inline-block; vertical-align: middle; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); }
.empty { padding: 28px 10px; text-align: center; color: var(--fg-2); }
.empty small { display: block; color: var(--fg-3); margin-top: 4px; }
.feed { display: flex; flex-direction: column; gap: 8px; }
.ev { display: grid; grid-template-columns: 62px 1fr auto; gap: 10px; align-items: baseline; font-size: 13px; }
.ev .when { font: 500 12px var(--mono); color: var(--fg-3); }
.chart svg { display: block; width: 100%; height: auto; }
.chart .axis { font: 10.5px var(--mono); fill: var(--fg-3); }
.chart .grid { stroke: var(--border); }
footer { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 20px; font-size: 11.5px; color: var(--fg-3); padding-top: 6px; }
footer a { color: var(--fg-2); }
a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
</head>
<body>
<div class="page">
  <header>
    <div class="brand">
      <div class="wordmark">Gap-Fill Bot<b>.</b></div>
      <div class="sub">live paper trading · <span id="profile">—</span></div>
    </div>
    <div class="status">
      <span class="pill" id="pill">Connecting</span>
      <span>bot saw <span class="mono" id="asof">—</span> <span class="muted" id="asofAgo"></span></span>
      <span>page checked <span class="mono muted" id="checked">—</span></span>
      <span class="muted" id="note"></span>
    </div>
  </header>

  <div class="tiles">
    <div class="tile"><div class="label">Equity</div><div class="value mono" id="tEquity">—</div><div class="sub mono" id="tEquitySub">since $100k start</div></div>
    <div class="tile"><div class="label">Day P&amp;L</div><div class="value mono" id="tDay">—</div><div class="sub mono" id="tDaySub">vs prior close</div></div>
    <div class="tile"><div class="label">Cash</div><div class="value mono" id="tCash">—</div><div class="sub">available</div></div>
    <div class="tile"><div class="label">Open positions</div><div class="value mono" id="tOpen">—</div><div class="sub" id="tOpenSub">of — slots</div></div>
    <div class="tile"><div class="label">Watching</div><div class="value mono" id="tWatch">—</div><div class="sub">gaps waiting for a wick touch</div></div>
  </div>

  <div class="row">
    <div class="card">
      <div class="card-head"><div class="card-title">Gap watch<span class="count" id="cWatch"></span></div><div class="card-note">closest to the wick first · last = the bot's own quote</div></div>
      <div class="scroll" id="watch"></div>
    </div>
    <div class="card">
      <div class="card-head"><div class="card-title">Open positions<span class="count" id="cPos"></span></div><div class="card-note">marked at the bot's last quote</div></div>
      <div class="scroll" id="positions"></div>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <div class="card-head"><div class="card-title">Equity curve</div><div class="card-note" id="curveNote">daily closes</div></div>
      <div class="chart" id="chart"></div>
    </div>
    <div class="card">
      <div class="card-head"><div class="card-title">Today<span class="count" id="cToday"></span></div><div class="card-note">fills and exits booked this session</div></div>
      <div id="today"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-head"><div class="card-title">Closed trades<span class="count" id="cTrades"></span></div><div class="card-note">newest first, last 20</div></div>
    <div class="scroll" id="trades"></div>
  </div>

  <footer>
    <div id="rule">Resting-limit entry at the gap-day open · gap-size priority · 1–2% bucket excluded · 3%-ADV floor · $0 commission + 10 bps</div>
    <div class="mono">quotes: Schwab, one pull a minute by the bot · <a href="https://github.com/GeorgeManavazian/gap-bot-state/tree/live">state repo</a></div>
  </footer>
</div>

<script>
(function () {
  'use strict';
  var API = 'https://api.github.com/repos/GeorgeManavazian/gap-bot-state/contents/live.json?ref=live';
  var SRC = new URLSearchParams(location.search).get('src') || API;
  var POLL_OPEN = 75000, POLL_CLOSED = 600000, STALE_AFTER = 180000, RATE_FLOOR = 8;

  var $ = function (id) { return document.getElementById(id); };
  var model = null, etag = null, lastFetch = 0, fetchError = null, backoffUntil = 0, timer = null;

  // ---- time helpers (everything in America/New_York) ----
  var etFmt = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  function etParts(d) {
    var o = {};
    etFmt.formatToParts(d).forEach(function (p) { o[p.type] = p.value; });
    return { dow: o.weekday, hm: (+o.hour % 24) * 100 + (+o.minute), label: o.hour % 24 + ':' + o.minute + ':' + o.second };
  }
  function marketHours(d) {
    var p = etParts(d || new Date());
    return ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'].indexOf(p.dow) >= 0 && p.hm >= 925 && p.hm <= 1610;
  }
  // "2026-10-13T10:32:02-0400" -> Date (Safari needs the colon in the offset)
  function parseStamp(s) {
    if (!s) return null;
    var d = new Date(s.replace(/([+-]\d\d)(\d\d)$/, '$1:$2'));
    return isNaN(d.getTime()) ? null : d;
  }
  function timeET(s) {
    var d = parseStamp(s);
    return d ? new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit', second: '2-digit' }).format(d) : '—';
  }
  function ago(ms) {
    if (ms == null) return '';
    var s = Math.max(0, Math.round(ms / 1000));
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's ago';
    return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm ago';
  }

  // ---- formatting ----
  function money(v) { return v == null ? '—' : (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 }); }
  function money2(v) { return v == null ? '—' : (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
  function pct(v, d) { return v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(d == null ? 2 : d) + '%'; }
  function px(v) { return v == null ? '—' : '$' + v.toFixed(2); }
  function sign(v) { return v > 0 ? 'pos' : v < 0 ? 'neg' : ''; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // ---- status line (runs every second) ----
  function setStatus() {
    var now = Date.now();
    var polled = model ? parseStamp(model.polled_at) : null;
    var age = polled ? now - polled.getTime() : null;
    var open = marketHours();
    var pill = $('pill'), cls = '', text = '';
    if (!model && fetchError) { cls = 'stale'; text = 'No data'; }
    else if (!model) { cls = ''; text = 'Connecting'; }
    else if (open && age != null && age < STALE_AFTER && !fetchError) { cls = 'live'; text = 'Live'; }
    else if (open) { cls = 'stale'; text = 'Stale'; }
    else { cls = ''; text = 'Closed'; }
    pill.className = 'pill ' + cls;
    pill.textContent = text;
    $('asof').textContent = model && model.polled_at ? timeET(model.polled_at) + ' ET' : (model ? 'EOD ' + (model.session || '—') : '—');
    $('asofAgo').textContent = age != null ? '· ' + ago(age) : '';
    $('checked').textContent = lastFetch ? ago(now - lastFetch) : '—';
    var note = '';
    if (fetchError) note = 'feed error: ' + fetchError;
    else if (now < backoffUntil) note = 'rate limit, resuming ' + timeET(new Date(backoffUntil).toISOString()) + ' ET';
    else if (!open) note = 'market closed · polling every 10 min';
    $('note').textContent = note;
  }

  // ---- rendering ----
  function render(m) {
    $('profile').textContent = m.profile || '—';
    var nOpen = (m.open_positions || []).length, nPend = (m.pending || []).length;
    $('tEquity').textContent = money(m.equity);
    $('tEquity').className = 'value mono ' + sign(m.equity - 100000);
    var tot = m.equity - 100000;
    $('tEquitySub').textContent = (tot >= 0 ? '+' : '') + money(tot) + ' (' + pct(tot / 1000) + ') since $100k start';
    $('tEquitySub').className = 'sub mono ' + sign(tot);
    var day = m.equity - m.day_start_equity;
    $('tDay').textContent = (day >= 0 ? '+' : '') + money(day);
    $('tDay').className = 'value mono ' + sign(day);
    $('tDaySub').textContent = pct(m.day_start_equity ? day / m.day_start_equity * 100 : 0) + (m.intraday ? ' intraday mark' : ' at the close');
    $('tCash').textContent = money(m.cash);
    $('tOpen').textContent = nOpen;
    $('tOpenSub').textContent = 'of ' + (m.slot_limit != null ? m.slot_limit : '—') + ' slots (' + (m.profile || '') + ')';
    $('tWatch').textContent = nPend;
    renderWatch(m);
    renderPositions(m);
    renderToday(m);
    renderChart(m);
    renderTrades(m);
    $('curveNote').textContent = m.intraday ? 'daily closes + live mark' : 'daily closes';
  }

  function renderWatch(m) {
    var rows = (m.pending || []).map(function (w) {
      var dist = (w.last != null && w.last > 0) ? (w.gap_open - w.last) / w.last * 100 : null;
      return Object.assign({ dist: dist }, w);
    }).sort(function (a, b) {
      if (a.dist == null && b.dist == null) return Math.abs(b.gap_pct) - Math.abs(a.gap_pct);
      if (a.dist == null) return 1;
      if (b.dist == null) return -1;
      return a.dist - b.dist;
    });
    $('cWatch').textContent = rows.length ? '(' + rows.length + ')' : '';
    if (!rows.length) { $('watch').innerHTML = '<div class="empty">Nothing waiting<small>no open gap-down watches right now</small></div>'; return; }
    var h = '<table><thead><tr><th>Ticker</th><th>Gap</th><th>Wick</th><th>Last</th><th>To wick</th><th>Session L–H</th><th>Wait</th></tr></thead><tbody>';
    rows.forEach(function (w) {
      var touched = w.consumed_on && w.consumed_on === m.session;
      var progress = w.dist == null ? 0 : Math.max(0, Math.min(100, 100 - w.dist / Math.abs(w.gap_pct) * 100));
      h += '<tr><td><span class="tk">' + esc(w.ticker) + '</span>' + (touched ? '<span class="badge">touched today</span>' : '') +
        '</td><td class="mono neg">' + pct(w.gap_pct) + ' <span class="muted">' + esc(w.bucket || '') + '</span></td>' +
        '<td class="mono">' + px(w.gap_open) + '</td><td class="mono">' + px(w.last) + '</td>' +
        '<td class="mono ' + (w.dist != null && w.dist <= 0 ? 'pos' : '') + '">' + pct(w.dist) + ' <span class="bar"><i style="width:' + progress.toFixed(0) + '%"></i></span></td>' +
        '<td class="mono muted">' + (w.l != null ? px(w.l) + '–' + px(w.h) : '—') + '</td>' +
        '<td class="mono muted">' + w.days_waited + 'd</td></tr>';
    });
    $('watch').innerHTML = h + '</tbody></table>';
  }

  function renderPositions(m) {
    var rows = m.open_positions || [];
    $('cPos').textContent = rows.length ? '(' + rows.length + ')' : '';
    if (!rows.length) { $('positions').innerHTML = '<div class="empty">No open positions<small>nothing has hit a wick entry yet</small></div>'; return; }
    var h = '<table><thead><tr><th>Ticker</th><th>Entry</th><th>Last</th><th>P&amp;L</th><th>To stop</th><th>To TP</th><th>Held</th></tr></thead><tbody>';
    rows.forEach(function (p) {
      var pnl = (p.last - p.entry) / p.entry * 100;
      var toStop = p.last ? (p.last - p.stop) / p.last * 100 : null;
      var toTp = p.last ? (p.tp - p.last) / p.last * 100 : null;
      h += '<tr><td><span class="tk">' + esc(p.ticker) + '</span> <span class="muted">' + esc(p.bucket || '') + '</span></td>' +
        '<td class="mono">' + px(p.entry) + '</td><td class="mono">' + px(p.last) + '</td>' +
        '<td class="mono ' + sign(pnl) + '">' + pct(pnl) + '</td>' +
        '<td class="mono muted">' + pct(toStop, 1) + '</td><td class="mono muted">' + pct(toTp, 1) + '</td>' +
        '<td class="mono muted">' + p.days_held + 'd</td></tr>';
    });
    $('positions').innerHTML = h + '</tbody></table>';
  }

  function renderToday(m) {
    var t = m.today || { entries: [], exits: [] };
    var evs = [];
    (t.entries || []).forEach(function (e) { evs.push({ time: e.time, html: '<span class="tk">' + esc(e.ticker) + '</span> bought at ' + px(e.entry), right: '<span class="muted">entry</span>' }); });
    (t.exits || []).forEach(function (x) { evs.push({ time: x.time, html: '<span class="tk">' + esc(x.ticker) + '</span> sold at ' + px(x.exit_price) + ' <span class="muted">' + esc(x.exit_reason) + '</span>', right: '<span class="mono ' + sign(x.pnl_pct) + '">' + pct(x.pnl_pct) + '</span>' }); });
    evs.sort(function (a, b) { return (b.time || '').localeCompare(a.time || ''); });
    $('cToday').textContent = evs.length ? '(' + evs.length + ')' : '';
    if (!evs.length) { $('today').innerHTML = '<div class="empty">Nothing booked yet today<small>' + (m.intraday ? 'the poller is watching' : 'session ' + esc(m.session || '—') + ' closed with no fills or exits') + '</small></div>'; return; }
    $('today').innerHTML = '<div class="feed">' + evs.map(function (e) {
      return '<div class="ev"><span class="when">' + (e.time ? timeET(e.time) : '—') + '</span><span>' + e.html + '</span><span>' + e.right + '</span></div>';
    }).join('') + '</div>';
  }

  function renderChart(m) {
    var pts = (m.equity_curve || []).slice();
    if (m.intraday && m.equity != null) pts.push({ date: m.session, equity: m.equity, live: true });
    var box = $('chart');
    if (pts.length < 2) { box.innerHTML = '<div class="empty">Not enough history yet<small>the curve fills in after a few sessions</small></div>'; return; }
    var W = 700, H = 220, L = 56, R = 14, T = 14, B = 26, pw = W - L - R, ph = H - T - B;
    var vals = pts.map(function (p) { return p.equity; }).concat([100000]);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = Math.max((hi - lo) * 0.08, 250); lo -= pad; hi += pad;
    var x = function (i) { return L + i / (pts.length - 1) * pw; };
    var y = function (v) { return T + (1 - (v - lo) / (hi - lo)) * ph; };
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Equity curve">';
    [lo, (lo + hi) / 2, hi].forEach(function (v) {
      s += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(v).toFixed(1) + '" y2="' + y(v).toFixed(1) + '"/>';
      s += '<text class="axis" x="' + (L - 8) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">$' + (v / 1000).toFixed(1) + 'k</text>';
    });
    s += '<line class="grid" stroke-dasharray="3 4" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(100000).toFixed(1) + '" y2="' + y(100000).toFixed(1) + '"/>';
    var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(p.equity).toFixed(1); }).join(' ');
    var area = d + ' L' + x(pts.length - 1).toFixed(1) + ',' + (T + ph) + ' L' + L + ',' + (T + ph) + ' Z';
    s += '<path d="' + area + '" fill="var(--accent)" opacity="0.08"/>';
    s += '<path d="' + d + '" fill="none" stroke="var(--accent)" stroke-width="2"/>';
    var last = pts[pts.length - 1];
    s += '<circle cx="' + x(pts.length - 1).toFixed(1) + '" cy="' + y(last.equity).toFixed(1) + '" r="4" fill="' + (last.live ? 'var(--warn)' : 'var(--accent)') + '"/>';
    s += '<text class="axis" x="' + L + '" y="' + (H - 8) + '">' + esc(pts[0].date) + '</text>';
    s += '<text class="axis" x="' + (W - R) + '" y="' + (H - 8) + '" text-anchor="end">' + esc(last.date) + (last.live ? ' (live)' : '') + '</text>';
    box.innerHTML = s + '</svg>';
  }

  function renderTrades(m) {
    var rows = m.closed_trades || [];
    $('cTrades').textContent = rows.length ? '(' + rows.length + ')' : '';
    if (!rows.length) { $('trades').innerHTML = '<div class="empty">No trades closed yet</div>'; return; }
    var h = '<table><thead><tr><th>Ticker</th><th>Entered</th><th>Exited</th><th>Exit</th><th>In</th><th>Out</th><th>P&amp;L</th><th>$</th><th>Held</th></tr></thead><tbody>';
    rows.forEach(function (t) {
      h += '<tr><td class="tk">' + esc(t.ticker) + '</td><td class="mono muted">' + esc(t.entry_date) + '</td><td class="mono muted">' + esc(t.exit_date) + '</td>' +
        '<td class="mono">' + esc(t.exit_reason) + '</td><td class="mono">' + px(t.entry_price) + '</td><td class="mono">' + px(t.exit_price) + '</td>' +
        '<td class="mono ' + sign(t.pnl_pct) + '">' + pct(t.pnl_pct) + '</td><td class="mono ' + sign(t.pnl_dollar) + '">' + money2(t.pnl_dollar) + '</td>' +
        '<td class="mono muted">' + (t.days_held == null ? '—' : t.days_held + 'd') + '</td></tr>';
    });
    $('trades').innerHTML = h + '</tbody></table>';
  }

  // ---- polling ----
  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(poll, ms != null ? ms : (marketHours() ? POLL_OPEN : POLL_CLOSED));
  }
  function poll() {
    var now = Date.now();
    if (now < backoffUntil) { setStatus(); schedule(backoffUntil - now + 1000); return; }
    var headers = { 'Accept': 'application/vnd.github.raw+json' };
    if (etag) headers['If-None-Match'] = etag;
    fetch(SRC, { headers: headers, cache: 'no-store' }).then(function (r) {
      lastFetch = Date.now();
      var rem = r.headers.get('X-RateLimit-Remaining'), reset = r.headers.get('X-RateLimit-Reset');
      if (rem !== null && +rem < RATE_FLOOR && reset) backoffUntil = +reset * 1000 + 2000;
      if (r.status === 304) { fetchError = null; return null; }
      if (r.status === 403 || r.status === 429) {
        backoffUntil = reset ? +reset * 1000 + 2000 : Date.now() + POLL_CLOSED;
        throw new Error('rate limited (HTTP ' + r.status + ')');
      }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      etag = r.headers.get('ETag');
      return r.json();
    }).then(function (doc) {
      if (doc) { model = doc; fetchError = null; render(model); }
    }).catch(function (e) {
      fetchError = e && e.message ? e.message : String(e);
    }).then(function () { setStatus(); schedule(); });
  }

  document.addEventListener('visibilitychange', function () {
    if (!document.hidden && Date.now() - lastFetch > 60000) poll();
  });
  setInterval(setStatus, 1000);
  setStatus();
  poll();
})();
</script>
</body>
</html>
```

- [ ] **Step 3: Check it locally against the fixture**

```bash
cd /Users/georgiemanavazian/Documents/Trading/code/gap-bot/docs && python3 -m http.server 8765 >/dev/null 2>&1 &
```

Then open `http://localhost:8765/index.html?src=fixtures/live.json` with the
Playwright MCP (`browser_navigate`), take one screenshot at 1280px and one at
400px (`browser_resize` then `browser_take_screenshot`), and check the console
(`browser_console_messages`) is clean. Then open
`http://localhost:8765/index.html?src=fixtures/nope.json` once and confirm the
pill reads "No data" with `feed error: HTTP 404` in the status line (Review
Focus 4). Kill the server.

Expected on the fixture: pill "Closed" or "Stale" (the fixture's `polled_at`
is in the past), equity $100,413, day P&L +$293, 2 positions, 2 watches with
CCL badged "touched today", today feed showing the NEW entry and OLD exit,
curve with an amber live point, 2 closed trades. Fix anything the screenshot
shows, one pass, no loop.

- [ ] **Step 4: Commit**

```bash
git add docs/index.html docs/fixtures/live.json
git commit -m "docs: GitHub Pages live dashboard reading live.json through the contents API

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Cutover

**Files:**
- Modify: `README.md` (dashboard paragraph), `CLAUDE.md` (one update line), vault `11 Gap Bot/_STATUS.md` (2026-10-11 block), memory file `gap-bot-ramp-profile-paper-run.md`.
- VPS: `live/livefeed.py`, `live/run_intraday.py`, `scripts/gap_bot_tick.sh`.

**Interfaces:**
- Consumes: everything above.
- Produces: `https://georgemanavazian.github.io/gap-bot/` live; routine `trig_01WekTjoXaBZcWufztKGZTfW` disabled; artifact `64e72a71-…` republished as a pointer.

- [ ] **Step 1: Enable GitHub Pages on the gap-bot repo**

```bash
gh api -X POST repos/GeorgeManavazian/gap-bot/pages -f 'source[branch]=main' -f 'source[path]=/docs'
gh api repos/GeorgeManavazian/gap-bot/pages --jq '.html_url, .status'
```

Expected: `https://georgemanavazian.github.io/gap-bot/` and a status that
becomes `built` within a couple of minutes (poll with the second command).
Push `main` first so the build has `docs/index.html`.

- [ ] **Step 2: Ship the producer to the VPS**

```bash
cd /Users/georgiemanavazian/Documents/Trading/code/gap-bot
for f in live/livefeed.py live/run_intraday.py scripts/gap_bot_tick.sh; do
  scp -o BatchMode=yes -i ~/.ssh/wheelbot.key "$f" "ubuntu@129.80.185.142:/home/ubuntu/gap-bot/$f"
done
ssh -o BatchMode=yes -i ~/.ssh/wheelbot.key ubuntu@129.80.185.142 '
  cd /home/ubuntu/gap-bot && md5sum live/livefeed.py live/run_intraday.py scripts/gap_bot_tick.sh &&
  GAPBOT_PROFILE=ramp PYTHONPATH=. /home/ubuntu/etf-bot/.venv-live/bin/python -m live.livefeed --print | head -20 &&
  GAPBOT_PROFILE=ramp PYTHONPATH=. /home/ubuntu/etf-bot/.venv-live/bin/python -m live.livefeed --push && echo PUSH_OK'
md5sum live/livefeed.py live/run_intraday.py scripts/gap_bot_tick.sh
curl -s -H "Accept: application/vnd.github.raw+json" "https://api.github.com/repos/GeorgeManavazian/gap-bot-state/contents/live.json?ref=live" | head -c 400
```

Expected: md5s match local; `--print` shows `"profile": "ramp"` and the
current pending count; `PUSH_OK`; the API returns the same document. Also
run the tick tests on the VPS: `cd /home/ubuntu/gap-bot && /home/ubuntu/etf-bot/.venv-live/bin/python -m pytest scripts/test_livefeed.py scripts/test_intraday_blockers.py -q` → all pass.

- [ ] **Step 3: Verify the page end to end**

Open `https://georgemanavazian.github.io/gap-bot/` in the Playwright MCP, one
screenshot: equity $100,000, pending count equal to the VPS state, pill
"Closed" (Sunday). On the next weekday, the first intraday tick at 09:31 ET
makes the pill go "Live"; the owner checks that, or the next session does.

- [ ] **Step 4: Retire the old pipeline**

- Republish the Claude artifact (`url` `https://claude.ai/code/artifact/64e72a71-c1d1-4428-b5f7-0fadc61d6490`, read it first) as a single card: title stays "Gap-Fill Bot", body "This dashboard moved. The live version updates every minute during market hours:" with a link to `https://georgemanavazian.github.io/gap-bot/`; `capabilities: {}` to drop the db grant.
- `RemoteTrigger` `update` on `trig_01WekTjoXaBZcWufztKGZTfW` with `{"enabled": false}`.

- [ ] **Step 5: Update the written record**

- `README.md`: replace the dashboard paragraph with the Pages URL, the
  `live` branch mechanism, and the 75s/1.5–2.5 min lag statement.
- `CLAUDE.md`: append to the 2026-10-11 update: live dashboard URL, `live/livefeed.py`, the tick change, "artifact + routine retired".
- Vault `_STATUS.md` 2026-10-11 block: same facts, plus the raw-CDN and
  artifact-CSP findings so nobody retries those paths.
- Memory `gap-bot-ramp-profile-paper-run.md`: add the dashboard URL and the
  `live` branch note.

- [ ] **Step 6: Commit and push**

```bash
git add README.md CLAUDE.md
git commit -m "Docs: live dashboard on GitHub Pages; artifact and nightly routine retired

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```
