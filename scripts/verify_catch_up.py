"""Self-check for scripts/catch_up_ledger.py: replaying the 2yr anchor window
from a fresh ledger through ITS bar builder (pandas indexing, own ADV and
prior_close) must reproduce the anchor, 174 trades / +11.45% / maxDD -14.59%,
the same numbers scripts/verify_live_engine.py gets via TickerView. Also
checks the CLI is a dry run by default and refuses --apply without a backup
flag. Run: python scripts/verify_catch_up.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from catch_up_ledger import load_frames, session_calendar, replay
from live.config import CAPITAL, VARIANTS
from live.fillers import make_filler


def main() -> int:
    frames = load_frames()
    cal = session_calendar(frames)
    cal = cal[cal >= cal[-1] - pd.DateOffset(years=2)]
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades, snaps = replay(state, frames, cal, make_filler("stock", VARIANTS["stock"]))
    eq = pd.Series([s["equity"] for s in snaps], index=cal)
    ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    print(f"catch-up replay: n={len(trades)}  return={ret:.2f}%  maxDD={dd:.2f}%")
    print("anchor to match: n=174  return=11.45%  maxDD=-14.59%")
    ok = len(trades) == 174 and abs(ret - 11.45) < 0.05 and abs(dd + 14.59) < 0.05
    print("MATCH" if ok else "MISMATCH")

    script = str(Path(__file__).parent / "catch_up_ledger.py")
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GAPBOT_STATE_DIR": tmp}
        r = subprocess.run([sys.executable, script, "--through", "2024-10-31"], env=env, capture_output=True, text=True,
                           cwd=Path(__file__).parent.parent)
        wrote = any(Path(tmp).rglob("*"))
        good = r.returncode == 0 and "DRY RUN" in r.stdout and not wrote
        print(("  ok   " if good else "  FAIL ") + "default is a dry run, writes nothing")
        ok &= good
        r = subprocess.run([sys.executable, script, "--apply"], env=env, capture_output=True, text=True,
                           cwd=Path(__file__).parent.parent)
        good = r.returncode == 2 and not any(Path(tmp).rglob("*"))
        print(("  ok   " if good else "  FAIL ") + "--apply without --i-have-a-backup refused")
        ok &= good
        r = subprocess.run([sys.executable, script, "--apply", "--i-have-a-backup", "--through", "2024-10-31"], env=env,
                           capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        st = Path(tmp, "account", "state.json")
        good = r.returncode == 0 and st.exists()
        print(("  ok   " if good else "  FAIL ") + "--apply --i-have-a-backup writes state to GAPBOT_STATE_DIR only")
        ok &= good
    print("ALL PASS" if ok else "FAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
