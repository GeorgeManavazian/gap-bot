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


# ------------------------------------------------------------ build_live_doc

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


# ------------------------------------------------------------------ push_live

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


# ------------------------------------------------------------ inputs + CLI

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


def test_module_runs_as_a_script(tmp_path):
    env = {**os.environ, "GAPBOT_STATE_DIR": str(tmp_path), "PYTHONPATH": str(Path(__file__).parent.parent)}
    r = subprocess.run([sys.executable, "-m", "live.livefeed", "--print"], env=env,
                       capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["schema"] == 1


# ----------------------------------------- token days-left, intraday series

def test_build_doc_carries_token_days_left_and_intraday_series():
    hb = {**_heartbeat(), "equity_series": [{"t": "2026-10-13T09:31:02-0400", "equity": 100120.0},
                                            {"t": "2026-10-13T10:32:02-0400", "equity": 100412.55}]}
    doc = livefeed.build_live_doc(_state(), hb, _snapshots(), _trades(),
                                  slot_limit=7, profile="ramp", generated_at=GEN, token_days_left=2.456)
    assert doc["schwab_token_days_left"] == 2.46
    assert doc["intraday_series"] == hb["equity_series"]
    stale = {**hb, "session": "2026-10-10"}
    doc = livefeed.build_live_doc(_state(), stale, _snapshots(), _trades(),
                                  slot_limit=7, profile="ramp", generated_at=GEN)
    assert doc["intraday_series"] == [] and doc["schwab_token_days_left"] is None
