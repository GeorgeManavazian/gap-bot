"""Push data/live/ to a PUBLIC GitHub repo (gap-bot-state) so a dashboard
can read current state without touching the VPS directly. Same one-way
doctrine as the wheel bot's live/sync.py: VPS -> GitHub -> reader, no
merge case, VPS is sole writer. The repo is public by the owner's choice
(cloud routines couldn't read a private one), so everything synced is
world-readable: only data/live/account/ goes up; logs/ is excluded below
because tick.log carries internal paths that nobody outside needs.

Credential lives OUTSIDE this repo, at ~/.gapbot/git.json on the VPS
({"repo": "owner/gap-bot-state", "token": "...", "user": "..."}) --
created by the owner directly on the box, via their own terminal, never
typed into or read back through an assistant session. If that file is
ever missing, sync_state() no-ops loudly (prints and returns) rather
than crash the tick -- a sync failure must never take down the trading
decision it's supposed to be reporting on.

A lightweight secret-scan runs on staged content before every push --
data/live/ should never contain anything credential-shaped, but "should
never" is exactly the assumption a real leak violates, so it's checked,
not assumed."""
from __future__ import annotations
import json
import os
import re
import subprocess

GIT_CFG = os.path.expanduser("~/.gapbot/git.json")

_SECRET_CONTENT = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}"),
)


def _load_cfg():
    if not os.path.exists(GIT_CFG):
        return None
    with open(GIT_CFG) as f:
        return json.load(f)


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def secret_guard(state_dir: str) -> list:
    """Every file under state_dir whose content matches a credential
    shape. Empty == safe to push."""
    offenders = []
    for root, _, files in os.walk(state_dir):
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "rb") as fh:
                    body = fh.read().decode("utf-8", errors="replace")
            except OSError:
                offenders.append(path)  # unreadable -> can't prove clean -> offender
                continue
            if any(pat.search(body) for pat in _SECRET_CONTENT):
                offenders.append(path)
    return offenders


def sync_state(state_dir: str, mirror_clone_dir: str) -> bool:
    """Force-push state_dir's contents into mirror_clone_dir's git repo,
    then push to the configured GitHub mirror. Returns True on success;
    False (with a printed reason) on any failure -- never raises, a sync
    problem must not fail the calling tick."""
    cfg = _load_cfg()
    if cfg is None:
        print(f"sync_state: no credential at {GIT_CFG} -- skipping sync "
              f"(create it yourself on the VPS, see live/README-sync.md)")
        return False

    offenders = secret_guard(state_dir)
    if offenders:
        print(f"sync_state: ABORTED, secret-shaped content in: {offenders}")
        return False

    os.makedirs(mirror_clone_dir, exist_ok=True)
    if not os.path.exists(os.path.join(mirror_clone_dir, ".git")):
        r = _run(["git", "init", "-q"], mirror_clone_dir)
        if r.returncode != 0:
            print(f"sync_state: git init failed: {r.stderr}")
            return False
        _run(["git", "config", "user.email", "gapbot@localhost"], mirror_clone_dir)
        _run(["git", "config", "user.name", "gap-bot"], mirror_clone_dir)

    r = _run(["rsync", "-a", "--delete", "--exclude=logs/",
              state_dir + "/", mirror_clone_dir + "/"], None)
    if r.returncode != 0:
        print(f"sync_state: rsync failed: {r.stderr}")
        return False

    _run(["git", "add", "-A"], mirror_clone_dir)
    diff = _run(["git", "diff", "--cached", "--quiet"], mirror_clone_dir)
    if diff.returncode == 0:
        return True  # nothing changed, nothing to push -- not a failure

    r = _run(["git", "commit", "-q", "-m", "state sync"], mirror_clone_dir)
    if r.returncode != 0:
        print(f"sync_state: commit failed: {r.stderr}")
        return False

    remote = f"https://x-access-token:{cfg['token']}@github.com/{cfg['repo']}.git"
    r = _run(["git", "push", "-q", "-f", remote, "HEAD:main"], mirror_clone_dir)
    if r.returncode != 0:
        # scrub the token out of any error text before it hits a log file
        safe_err = r.stderr.replace(cfg["token"], "***")
        print(f"sync_state: push failed: {safe_err}")
        return False
    return True
