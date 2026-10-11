"""Tests for live/notify.py (ntfy pushes) and live/tokenage.py.

Run: ../etf-bot/.venv/bin/python -m pytest scripts/test_notify.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from live import notify, tokenage  # noqa: E402

TICK_OUT = """[gapbot profile=ramp]
run_intraday[stock]: recorded exit LULU tp_gap_filled @ 121.8 (+23.97%) at 2026-10-13T10:15:02-0400
run_intraday[stock]: recorded entry T @ 22.3800 (stop 11.1900, tp 24.8700) at 2026-10-13T10:15:02-0400
run_intraday[stock]: 1 exits, 1 entries, 2 open, 14 pending, equity $100,412.55
"""
EOD_OUT = """[gapbot profile=ramp]
run_daily: 2026-10-13 -- 501/503 tickers with usable bars
run_daily[stock]: booked 1 exits today, 2 open, 14 pending, equity $100,412.55
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "ntfy.json"
    p.write_text(json.dumps({"server": "https://ntfy.example", "topic": "t0p1c"}))
    monkeypatch.setenv("GAPBOT_NTFY_CFG", str(p))
    return p


@pytest.fixture
def posts(monkeypatch):
    sent = []
    def fake_post(url, data, headers):
        sent.append({"url": url, "data": data.decode(), "headers": headers})
        return 200
    monkeypatch.setattr(notify, "_post", fake_post)
    return sent


def test_send_posts_title_priority_tags_to_the_topic(cfg, posts):
    assert notify.send("Hi there", "body", priority="high", tags=("key", "moon")) is True
    assert posts[0]["url"] == "https://ntfy.example/t0p1c"
    assert posts[0]["data"] == "body"
    assert posts[0]["headers"] == {"Title": "Hi there", "Priority": "high", "Tags": "key,moon"}


def test_send_without_config_is_false_and_silent(tmp_path, monkeypatch, posts, capsys):
    monkeypatch.setenv("GAPBOT_NTFY_CFG", str(tmp_path / "missing.json"))
    assert notify.send("x", "y") is False
    assert posts == [] and "no config" in capsys.readouterr().out


def test_send_network_error_is_false_not_raised(cfg, monkeypatch, capsys):
    def boom(url, data, headers):
        raise OSError("connection refused")
    monkeypatch.setattr(notify, "_post", boom)
    assert notify.send("x", "y") is False
    assert "send failed" in capsys.readouterr().out


def test_events_subcommand_parses_entries_and_exits(cfg, posts, tmp_path):
    out = tmp_path / "out"
    out.write_text(TICK_OUT)
    assert notify.main(["events", str(out)]) == 0
    assert posts[0]["headers"]["Title"] == "Gap bot: 1 entry, 1 exit"
    assert posts[0]["data"] == "BUY T @ 22.38\nSELL LULU @ 121.80 +23.97% (tp_gap_filled)"


def test_events_subcommand_sends_nothing_for_a_quiet_poll(cfg, posts, tmp_path):
    out = tmp_path / "out"
    out.write_text("[gapbot profile=ramp]\nrun_intraday[stock]: 0 exits, 0 entries, 1 open, 14 pending, equity $100,000.00\n")
    assert notify.main(["events", str(out)]) == 1 and posts == []


def test_eod_subcommand_sends_the_booked_line(cfg, posts, tmp_path):
    out = tmp_path / "out"
    out.write_text(EOD_OUT)
    assert notify.main(["eod", str(out)]) == 0
    assert posts[0]["headers"]["Title"] == "Gap bot EOD 2026-10-13"
    assert posts[0]["data"] == "booked 1 exits today, 2 open, 14 pending, equity $100,412.55"
    out.write_text("run_daily: no new session yet (last session 2026-10-13 already processed)\n")
    assert notify.main(["eod", str(out)]) == 1 and len(posts) == 1


def test_tick_failed_subcommand_sends_the_tail_at_high_priority(cfg, posts, tmp_path):
    out = tmp_path / "out"
    out.write_text("\n".join(f"line {i}" for i in range(10)) + "\n")
    assert notify.main(["tick-failed", "--mode", "intraday", "--rc", "1", str(out)]) == 0
    assert posts[0]["headers"]["Title"] == "Gap bot tick FAILED (intraday, exit 1)"
    assert posts[0]["headers"]["Priority"] == "high"
    assert posts[0]["data"] == "\n".join(f"line {i}" for i in range(4, 10))


def _token(tmp_path, age_days):
    p = tmp_path / "token.json"
    p.write_text(json.dumps({"creation_timestamp": time.time() - age_days * 86400, "token": {}}))
    return str(p)


def test_tokenage_reads_creation_timestamp_not_mtime(tmp_path):
    p = _token(tmp_path, 5.0)
    assert abs(tokenage.token_age_days(p) - 5.0) < 0.01
    assert abs(tokenage.days_until_expiry(p) - 2.0) < 0.01
    assert tokenage.days_until_expiry(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert tokenage.days_until_expiry(str(bad)) is None


def test_token_age_subcommand_thresholds(cfg, posts, tmp_path):
    assert notify.main(["token-age", "--token", _token(tmp_path, 1.0)]) == 1 and posts == []     # 6d left: quiet
    assert notify.main(["token-age", "--token", _token(tmp_path, 4.5)]) == 0                      # 2.5d left: warn
    assert posts[-1]["headers"]["Title"] == "Schwab login due in 2.5 days" and posts[-1]["headers"]["Priority"] == "high"
    assert notify.main(["token-age", "--token", _token(tmp_path, 7.5)]) == 0                      # lapsed
    assert posts[-1]["headers"]["Title"].startswith("Schwab token EXPIRED") and posts[-1]["headers"]["Priority"] == "urgent"
    assert notify.main(["token-age", "--token", str(tmp_path / "missing.json")]) == 1              # no file: quiet
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert notify.main(["token-age", "--token", str(bad)]) == 0                                    # unreadable: alarm
    assert posts[-1]["headers"]["Title"] == "Schwab token file unreadable"


def test_token_age_delivery_failure_exits_1_so_the_tick_retries(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(notify, "_post", lambda url, data, headers: 500)
    assert notify.main(["token-age", "--token", _token(tmp_path, 5.0)]) == 1
