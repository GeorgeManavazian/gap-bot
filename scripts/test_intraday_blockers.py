"""Regression tests for the un-pause blockers found in the 2026-10-06 review
of the intraday paper-fill path (stock variant only; call/spread stay paused).

  1. staleness: run_daily refuses a ledger whose last_run_date is not the
     previous NYSE session (nothing written, exit 1; --allow-stale-from <date>
     overrides for that exact ledger date only);
     the intraday poller applies the same rule, so ONE missed evening run is
     refused instead of polled with counters off by one;
  2. a watch the poller consumed is tagged, not deleted, so the EOD run can
     not re-register the same ticker the same day (engine.py step 2a);
  3. gap_bot_tick.sh runs both modes with --variants stock;
  4. append_trade / append_snapshot are idempotent, so a crash between the
     append and save_state books a trade once on the re-run;
  5. one tick at a time (flock in the tick script) and a per-pid tmp file in
     save_state;
  6. an ADV lookup failure intraday leaves the watch for the EOD run instead
     of filling unfloored.

Run: .venv/bin/python -m pytest scripts/test_intraday_blockers.py
Reuses verify_intraday.py's fake Schwab client and poller driver.
"""
from __future__ import annotations
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import verify_intraday as vi  # noqa: E402
from live import state as state_mod  # noqa: E402
from live.intraday import check_watches, previous_session, state_is_stale  # noqa: E402
from live.engine import step_one_day  # noqa: E402

PY = sys.executable
REPO = Path(__file__).parent.parent
TICK = REPO / "scripts" / "gap_bot_tick.sh"
SPY = vi._qjson(500, 501, 499, 500, "2026-09-02 09:59")
TOUCH = vi._qjson(94, 95.5, 93, 95.2, "2026-09-02 09:58")
POLL_AT = "2026-09-02 10:00"


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GAPBOT_STATE_DIR", str(tmp_path))
    return str(tmp_path)


def _rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path) if ln.strip()]


# ---------------------------------------------------------------- 1. staleness

def test_previous_session_skips_weekends_and_nyse_holidays():
    assert previous_session("2026-09-08") == "2026-09-04"   # Tue after Labor Day -> Fri
    assert previous_session("2026-09-01") == "2026-08-31"   # Tue -> Mon
    assert previous_session("2026-08-31") == "2026-08-28"   # Mon -> Fri
    assert previous_session("2026-07-06") == "2026-07-02"   # Jul 3 observed holiday
    assert previous_session("2026-04-06") == "2026-04-02"   # Good Friday 2026-04-03
    assert previous_session("2027-06-21") == "2027-06-17"   # Juneteenth observed Fri 2027-06-18
    assert previous_session("2027-12-27") == "2027-12-23"   # Christmas observed Fri 2027-12-24
    assert previous_session("2026-11-27") == "2026-11-25"   # Thanksgiving


def test_state_is_stale_is_the_previous_session_rule():
    assert not state_is_stale(None, "2026-09-02")               # never run: nothing to be stale about
    assert not state_is_stale("2026-09-01", "2026-09-02")       # previous session
    assert not state_is_stale("2026-08-28", "2026-08-31")       # Fri -> Mon
    assert not state_is_stale("2026-09-04", "2026-09-08")       # Fri -> Tue over Labor Day
    assert state_is_stale("2026-08-28", "2026-09-01")           # missed Mon 08-31 (old 5-day rule let this through)
    assert state_is_stale("2026-08-31", "2026-09-02")           # one missed evening run
    assert state_is_stale("2026-09-04", "2026-10-05")           # post-pause
    assert state_is_stale("2026-09-02", "2026-09-02")           # same day: callers guard first, still not "previous"


def test_intraday_refuses_one_missed_evening_run(state_dir):
    vi._seed(state_dir, last_run="2026-08-31")  # Mon; poll is Wed 09-02, so Tue's EOD run never happened
    rc, out = vi._run_poller(state_dir, vi.FakeClient({"SPY": SPY, "AAA": TOUCH}), POLL_AT)
    st = vi._load(state_dir)
    assert rc == 0 and "stale" in out
    assert "AAA" in st["pending"] and not st["open_positions"]


def _seed_daily(state_dir, last_run):
    vi._seed(state_dir, last_run=last_run)
    return open(os.path.join(state_dir, "account", "state.json")).read()


def test_run_daily_refuses_stale_ledger_and_writes_nothing(state_dir, capsys):
    import live.run_daily as rd
    before = _seed_daily(state_dir, "2026-08-14")
    bars = {"AAA": vi.bar(94, 95.5, 93, 95.2, prior_close=100)}  # would fill AAA if stepped
    ok = rd.run_variant("stock", "2026-09-02", bars, None)
    out = capsys.readouterr().out
    assert ok is False
    assert "catch_up_ledger" in out and "2026-08-14" in out and "2026-09-01" in out
    assert open(os.path.join(state_dir, "account", "state.json")).read() == before
    assert not os.path.exists(os.path.join(state_dir, "account", "trades.jsonl"))
    assert not os.path.exists(os.path.join(state_dir, "account", "snapshots.jsonl"))
    # the override must name the ledger's exact last_run_date, so it can't be left in a cron line
    assert rd.run_variant("stock", "2026-09-02", bars, None, allow_stale_from="2026-08-13") is False
    assert vi._load(state_dir)["last_run_date"] == "2026-08-14"
    assert rd.run_variant("stock", "2026-09-02", bars, None, allow_stale_from="2026-08-14") is True
    st = vi._load(state_dir)
    assert st["last_run_date"] == "2026-09-02" and "AAA" in st["open_positions"]


def test_run_daily_main_exits_nonzero_on_stale(state_dir, monkeypatch):
    import live.run_daily as rd
    import live.fixture_data as fd
    _seed_daily(state_dir, "2026-08-14")
    monkeypatch.setattr(rd, "load_universe", lambda: ["AAA"])
    monkeypatch.setattr(fd, "fetch_universe_bars", lambda universe, as_of: {})
    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--dry-run", "--as-of", "2026-09-02", "--variants", "stock"])
    assert rd.main() == 1
    assert vi._load(state_dir)["last_run_date"] == "2026-08-14"
    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--dry-run", "--as-of", "2026-09-02", "--variants", "stock",
                                      "--allow-stale-from", "2026-08-14"])
    assert rd.main() == 0
    assert vi._load(state_dir)["last_run_date"] == "2026-09-02"


# ------------------------------------------------- 2. consumed watch is tagged

def _reviewer_case(intraday: bool) -> dict:
    """AAA pending from D0. On D1 it touches intraday but the fill is refused
    (tiny ADV), and D1's settled candle also gaps AAA down again (-6%). EOD
    alone: AAA is still pending at step 2a so it is NOT re-registered, then
    consumed at 2c -> pending {}. Intraday + EOD must end the same way."""
    s = vi.fresh()
    vi.register(s, "AAA", 95.0, 100.0, vi.D0)
    if intraday:
        r = check_watches(s, vi.D1, {"AAA": vi.bar(94, 95.5, 93, 95.2)}, vi.FILLER, adv_lookup=lambda tk: 1.0)
        assert not r["filled"] and not s["open_positions"]
    step_one_day(s, vi.D1, {"AAA": vi.bar(94, 95.5, 93, 95.2, prior_close=100, adv=1.0)}, vi.FILLER)
    return s


def test_consumed_watch_cannot_reregister_same_day():
    eod_only, both = _reviewer_case(False), _reviewer_case(True)
    assert eod_only["pending"] == {}
    assert both["pending"] == eod_only["pending"]
    assert both["open_positions"] == eod_only["open_positions"] == {}


def test_tagged_watch_is_skipped_later_and_keeps_its_resting_slot():
    from live.config import MAX_SLOTS
    s = vi.fresh()
    for i in range(MAX_SLOTS - 1):
        vi.register(s, f"F{i:02d}", 90.0, 100.0, vi.D0)
    step_one_day(s, vi.D1, {f"F{i:02d}": vi.bar(89, 91, 88, 90, prior_close=100) for i in range(MAX_SLOTS - 1)},
                 vi.FILLER)
    assert len(s["open_positions"]) == MAX_SLOTS - 1  # one free slot
    vi.register(s, "BIG", 90.0, 100.0, vi.D1)
    vi.register(s, "SMALL", 97.0, 100.0, vi.D1)
    # D2 poll 1: BIG (the one resting order) touches, ADV-refused -> consumed, tagged, still listed
    r = check_watches(s, vi.D2, {"BIG": vi.bar(89, 91, 88, 90.5)}, vi.FILLER, adv_lookup=lambda tk: 1.0)
    assert not r["filled"] and s["pending"]["BIG"].get("consumed_on") == vi.D2
    # D2 poll 2: BIG touches again (no lookup, no fill); SMALL touches but BIG still holds
    # the day's only resting slot -> SMALL is a missed wick, consumed unfilled, same as EOD alone
    looked_up = []
    snap = {"BIG": vi.bar(89, 91, 88, 90.5), "SMALL": vi.bar(96, 98, 95, 97.5)}
    r = check_watches(s, vi.D2, snap, vi.FILLER, adv_lookup=lambda tk: looked_up.append(tk) or vi.ADV_BIG)
    assert r["filled"] == [] and looked_up == []
    assert s["pending"]["SMALL"].get("consumed_on") == vi.D2
    # EOD drops both without filling
    day = {"BIG": vi.bar(89, 91, 88, 90.5, prior_close=90, adv=1.0), "SMALL": vi.bar(96, 98, 95, 97.5, prior_close=97)}
    step_one_day(s, vi.D2, day, vi.FILLER)
    assert "BIG" not in s["pending"] and "SMALL" not in s["pending"]
    assert set(s["open_positions"]) == {f"F{i:02d}" for i in range(MAX_SLOTS - 1)}


# ----------------------------------------------- 3. tick script: stock only

def _stub(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _tick_env(repo, dow, hm, py, extra=None):
    Path(repo).mkdir(exist_ok=True)
    return {**os.environ, "GAPBOT_REPO": str(repo), "GAPBOT_VENV_PY": py,
            "GAPBOT_FAKE_DOW": str(dow), "GAPBOT_FAKE_HM": str(hm), **(extra or {})}


def _tick(repo, dow, hm, py="/bin/echo", extra=None):
    r = subprocess.run(["bash", str(TICK)], env=_tick_env(repo, dow, hm, py, extra), capture_output=True, text=True)
    log = Path(repo, "data/live/logs/tick.log")
    return r.returncode, (log.read_text() if log.exists() else "")


def test_tick_passes_variants_stock_in_both_windows(tmp_path):
    argfile = tmp_path / "args"
    py = _stub(tmp_path, "py", f'echo "$@" >> "{argfile}"\n')
    for hm in (1000, 1800):
        rc, log = _tick(tmp_path / f"repo{hm}", 2, hm, py=py)
        assert rc == 0 and "tick ok" in log
    runs = [ln for ln in argfile.read_text().splitlines() if "live/run_" in ln]
    assert len(runs) == 2
    assert runs[0].endswith("live/run_intraday.py --variants stock")
    assert runs[1].endswith("live/run_daily.py --variants stock")


# ---------------------------------------------- 4. append once, not twice

def test_append_trade_and_snapshot_are_idempotent(state_dir):
    p = state_mod.account_paths("stock")
    t = {"ticker": "AAA", "entry_date": vi.D1, "exit_date": vi.D2, "exit_reason": "stop", "pnl_pct": -1.0}
    assert state_mod.append_trade(t, "stock") is True
    assert state_mod.append_trade({**t, "exit_time": "T10:05", "pnl_pct": -1.01}, "stock") is False  # same key
    # one position per (ticker, entry_date): a different exit reason/date is the SAME position re-booked
    assert state_mod.append_trade({**t, "exit_reason": "tp_gap_filled", "exit_date": "2026-09-04"}, "stock") is False
    assert state_mod.append_trade({**t, "entry_date": vi.D2}, "stock") is True
    assert [r["entry_date"] for r in _rows(p["trades"])] == [vi.D1, vi.D2]
    assert state_mod.append_snapshot({"date": vi.D1, "equity": 1.0}, "stock") is True
    assert state_mod.append_snapshot({"date": vi.D1, "equity": 2.0}, "stock") is False
    assert state_mod.append_snapshot({"date": vi.D2, "equity": 2.0}, "stock") is True
    assert [r["equity"] for r in _rows(p["snapshots"])] == [1.0, 2.0]


def _seed_open_position(state_dir):
    """AAA filled on D0 (Tue 09-01), ledger closed out for D0."""
    s = vi.fresh()
    vi.register(s, "AAA", 95.0, 100.0, "2026-08-31")
    step_one_day(s, vi.D0, {"AAA": vi.bar(94, 96, 93, 95, prior_close=100)}, vi.FILLER)
    assert "AAA" in s["open_positions"] and s["last_run_date"] == vi.D0
    state_mod.save_state(s, "stock")
    return s["open_positions"]["AAA"]


def _raise_once(monkeypatch, module, name):
    real, calls = getattr(module, name), []

    def boom(*a, **k):
        if not calls:
            calls.append(1)
            raise RuntimeError("simulated crash before save_state")
        return real(*a, **k)
    monkeypatch.setattr(module, name, boom)


def test_crash_between_append_and_save_books_trade_once_intraday(state_dir, monkeypatch):
    pos = _seed_open_position(state_dir)
    p = state_mod.account_paths("stock")
    stop_hit = vi._qjson(94, 95, pos["stop_price"] - 0.5, 80, "2026-09-02 09:58")
    _raise_once(monkeypatch, state_mod, "save_state")
    rc, out = vi._run_poller(state_dir, vi.FakeClient({"SPY": SPY, "AAA": stop_hit}), POLL_AT)
    assert rc == 1 and "FAILED" in out
    assert len(_rows(p["trades"])) == 1 and "AAA" in vi._load(state_dir)["open_positions"]  # row written, state not
    rc, out = vi._run_poller(state_dir, vi.FakeClient({"SPY": SPY, "AAA": stop_hit}), POLL_AT)  # next poll
    assert rc == 0 and "recorded exit AAA stop" in out
    assert len(_rows(p["trades"])) == 1 and "AAA" not in vi._load(state_dir)["open_positions"]


def test_crash_after_intraday_tp_then_eod_stop_books_position_once(state_dir, monkeypatch):
    import live.run_daily as rd
    pos = _seed_open_position(state_dir)
    p = state_mod.account_paths("stock")
    tp_hit = vi._qjson(94, pos["tp_price"] + 0.5, 94, 99, "2026-09-02 09:58")
    _raise_once(monkeypatch, state_mod, "save_state")
    rc, out = vi._run_poller(state_dir, vi.FakeClient({"SPY": SPY, "AAA": tp_hit}), POLL_AT)
    assert rc == 1 and len(_rows(p["trades"])) == 1 and "AAA" in vi._load(state_dir)["open_positions"]
    # the poller never runs again; that evening the settled candle also went through the stop (stop-first)
    bars = {"AAA": vi.bar(94, pos["tp_price"] + 0.5, pos["stop_price"] - 0.5, 80, prior_close=95)}
    assert rd.run_variant("stock", vi.D1, bars, None) is True
    rows = _rows(p["trades"])
    assert len(rows) == 1 and rows[0]["ticker"] == "AAA" and rows[0]["entry_date"] == vi.D0
    assert "AAA" not in vi._load(state_dir)["open_positions"]


def test_crash_between_append_and_save_books_trade_once_eod(state_dir, monkeypatch):
    import live.run_daily as rd
    pos = _seed_open_position(state_dir)
    p = state_mod.account_paths("stock")
    bars = {"AAA": vi.bar(94, 95, pos["stop_price"] - 0.5, 80, prior_close=95)}
    _raise_once(monkeypatch, state_mod, "save_state")
    with pytest.raises(RuntimeError):
        rd.run_variant("stock", vi.D1, bars, None)
    assert len(_rows(p["trades"])) == 1 and len(_rows(p["snapshots"])) == 1
    assert rd.run_variant("stock", vi.D1, bars, None) is True  # the re-run
    assert len(_rows(p["trades"])) == 1 and len(_rows(p["snapshots"])) == 1
    st = vi._load(state_dir)
    assert st["last_run_date"] == vi.D1 and "AAA" not in st["open_positions"]


# ------------------------------------------------------- 5. one writer at a time

def test_save_state_tmp_file_is_per_pid(state_dir, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: seen.append(src) or real(src, dst))
    state_mod.save_state(vi.fresh(), "stock")
    assert seen and seen[0].endswith(f"state.json.tmp.{os.getpid()}")


def test_stale_tmp_files_are_removed_and_never_synced(state_dir, monkeypatch):
    import live.run_intraday as ri
    import live.sync as sync
    p = state_mod.account_paths("stock")
    os.makedirs(p["dir"])
    for name in ("state.json.tmp.4242", "intraday.json.tmp.4242"):
        Path(p["dir"], name).write_text("{")  # torn writes from a killed tick
    state_mod.save_state(vi.fresh(), "stock")
    ri.write_heartbeat("stock", {"polled_at": "x"})
    left = sorted(os.listdir(p["dir"]))
    assert left == ["intraday.json", "state.json"], left
    # belt and braces: the mirror rsync excludes tmp files even if one is mid-write
    cmds = []
    monkeypatch.setattr(sync, "_load_cfg", lambda: {"repo": "x/y", "token": "t", "user": "u"})
    monkeypatch.setattr(sync, "_run", lambda cmd, cwd: cmds.append(cmd) or
                        type("R", (), {"returncode": 0 if cmd[0] == "git" else 1, "stderr": ""})())
    assert sync.sync_state(state_dir, os.path.join(state_dir, "mirror")) is False  # rsync stubbed to fail
    rsync = [c for c in cmds if c[0] == "rsync"]
    assert rsync and "--exclude=*.tmp.*" in rsync[0]


def _flock_on_path(tmp_path):
    """util-linux flock(1) is not on macOS; a stand-in with the same
    `flock -n FD` contract (fcntl on the inherited fd) when it is missing."""
    import shutil
    if shutil.which("flock"):
        return os.environ["PATH"]
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    shim = d / "flock"
    shim.write_text(f"#!{PY}\nimport fcntl, sys\n"
                    "try:\n    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | (fcntl.LOCK_NB if '-n' in sys.argv else 0))\n"
                    "except OSError:\n    sys.exit(1)\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return f"{d}:{os.environ['PATH']}"


def test_tick_skips_when_lock_held(tmp_path):
    repo = tmp_path / "repo"
    slow = _stub(tmp_path, "slow_py", "sleep 2\n")
    env = _tick_env(repo, 2, 1000, slow, {"PATH": _flock_on_path(tmp_path)})
    first = subprocess.Popen(["bash", str(TICK)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log_path = Path(repo, "data/live/logs/tick.log")
    deadline = time.monotonic() + 10
    while not (log_path.exists() and "tick start (intraday)" in log_path.read_text()):
        assert time.monotonic() < deadline, "first tick never started"
        time.sleep(0.05)
    second = subprocess.run(["bash", str(TICK)], env=env, capture_output=True, text=True)
    assert first.wait() == 0 and second.returncode == 0
    log = Path(repo, "data/live/logs/tick.log").read_text()
    assert "tick skipped (lock held)" in log
    assert log.count("tick ok (intraday)") == 1
    assert Path(repo, "data/live/logs/.tick.lock").exists()


def test_tick_fails_loudly_when_lock_file_cannot_be_opened(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can open anything")
    repo = tmp_path / "repo"
    lock = Path(repo, "data/live/logs/.tick.lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    lock.chmod(0)  # e.g. a root-owned lock left by a manual run
    rc, log = _tick(repo, 2, 1000, extra={"PATH": _flock_on_path(tmp_path)})
    lock.chmod(0o644)
    assert rc == 1 and "lock open failed" in log and "skipped" not in log


@pytest.mark.skipif(shutil.which("timeout") is None, reason="coreutils timeout not on PATH (macOS)")
def test_tick_intraday_timeout_logs_failed(tmp_path):
    slow = _stub(tmp_path, "slow_py", "sleep 3\n")
    rc, log = _tick(tmp_path / "repo", 2, 1000, py=slow, extra={"GAPBOT_TIMEOUT": "1"})
    assert rc == 1 and "tick FAILED (intraday), exit 124" in log


# ----------------------------------------- 6. ADV lookup failure fails closed

def test_adv_lookup_failure_leaves_watch_for_eod(capsys):
    s = vi.fresh()
    vi.register(s, "AAA", 95.0, 100.0, vi.D0)
    snap = {"AAA": vi.bar(94, 95.5, 93, 95.2, adv=None)}  # a quote carries no adv
    r = check_watches(s, vi.D1, dict(snap), vi.FILLER, adv_lookup=lambda tk: None)
    assert not r["filled"] and not s["open_positions"]
    assert "AAA" in s["pending"] and "consumed_on" not in s["pending"]["AAA"]
    assert "ADV" in capsys.readouterr().out
    # the next poll with a working lookup fills it as before
    r = check_watches(s, vi.D1, dict(snap), vi.FILLER, adv_lookup=lambda tk: vi.ADV_BIG, stamp="T10:05")
    assert r["filled"] == ["AAA"] and "AAA" not in s["pending"]
    # the EOD engine is unchanged: adv None there is a newly-listed name, unfloored (backtest semantics)
    s = vi.fresh()
    vi.register(s, "AAA", 95.0, 100.0, vi.D0)
    step_one_day(s, vi.D1, {"AAA": vi.bar(94, 95.5, 93, 95.2, prior_close=100, adv=None)}, vi.FILLER)
    assert "AAA" in s["open_positions"]
