#!/bin/bash
# One tick of the gap bot, fired every minute by systemd (gapbot.timer).
# Market-hours gating happens here in ET so the VPS clock stays UTC, same
# doctrine as the wheel bot's wheelbot_tick.sh. Two windows:
#
#   09:31-16:00 ET  live/run_intraday.py -- every minute, one batched quote
#                   pull on the watched names, records a touch / stop /
#                   take-profit within a minute of it happening (2026-09-08;
#                   1-minute cadence 2026-10-06). Only ever acts on watches
#                   from PRIOR sessions and positions opened on prior
#                   sessions; registers nothing, advances no counter. Paper
#                   only. Set GAPBOT_INTRADAY=0 in the environment to leave
#                   this window a no-op.
#   17:00-23:30 ET  live/run_daily.py -- the one real decision a day: the
#                   EOD gap-fill scan off the settled daily candle. Also the
#                   backstop for anything the intraday poller missed. Fires
#                   only when the minute is a multiple of 5 (the timer now
#                   triggers every minute; run_daily's reference pull is
#                   not free). Its own last_run_date guard makes every
#                   trigger after the first successful one a harmless
#                   no-op, so this gate only needs to be "roughly after
#                   close on a weekday," not DST-precise.
#
# The first intraday poll is 09:31, not 09:30: quotes are cumulative for
# the session (running high/low), so nothing in the first minute is lost,
# and it keeps the poller clear of the opening print settling.
#
# Logging (2026-10-06, 1-minute cadence): an intraday poll that recorded
# nothing writes NOTHING to tick.log -- otherwise ~390 "start/ok" pairs a
# day bury the real events. Logged: the session's first successful poll
# (so the log proves the poller is alive), any poll that recorded an
# event, a FAILED poll, a skipped (lock held) poll. EOD logs as before.
#
# Reuses the wheel bot's Python venv (PY below) -- same schwab-py, pandas,
# numpy already installed there; gap-bot has no venv of its own on the VPS.
# GAPBOT_REPO / GAPBOT_VENV_PY / GAPBOT_FAKE_* are testability hooks, unset
# in production. Both runs are STOCK VARIANT ONLY (2026-10-06); the tick
# takes a lock so two ticks never write the same ledger at once.
set -uo pipefail
REPO="${GAPBOT_REPO:-/home/ubuntu/gap-bot}"
PY="${GAPBOT_VENV_PY:-/home/ubuntu/etf-bot/.venv-live/bin/python}"
cd "$REPO" || exit 1
LOGDIR="$REPO/data/live/logs"
mkdir -p "$LOGDIR"

ET(){ TZ=America/New_York date "$@"; }
log(){ echo "$(ET '+%Y-%m-%d %H:%M:%S ET') $*" >> "$LOGDIR/tick.log"; }

DOW="${GAPBOT_FAKE_DOW:-$(ET +%u)}"       # 1=Mon .. 7=Sun
HM="${GAPBOT_FAKE_HM:-$((10#$(ET +%H%M)))}"
MIN=$((HM % 100))                         # minute of the hour, derived from HM

if [ "$DOW" -gt 5 ]; then
    exit 0  # weekend, no-op, not even worth logging every minute
fi
if [ "$HM" -ge 931 ] && [ "$HM" -le 1600 ]; then
    [ "${GAPBOT_INTRADAY:-1}" = "0" ] && exit 0
    MODE=intraday
    SCRIPT="$REPO/live/run_intraday.py"
elif [ "$HM" -ge 1700 ] && [ "$HM" -le 2330 ]; then
    # EOD window: same 17:00-23:30 ET the wheel bot uses, buffer past the
    # 16:00 close for Schwab's daily candle to settle. Every 5 minutes only.
    [ $((MIN % 5)) -ne 0 ] && exit 0
    MODE=eod
    SCRIPT="$REPO/live/run_daily.py"
else
    exit 0
fi

# One tick at a time: a slow poll must never overlap the next trigger or
# the EOD run (two writers on the same state.json). The kernel drops the
# lock when the holder exits, so a crash can't leave it stale. flock(1) is
# util-linux (always on the VPS); a dev box without it runs unlocked.
if command -v flock >/dev/null 2>&1; then
    # An unopenable lock file (e.g. root-owned from a manual run) must fail
    # loudly, not read as "held" on every tick forever.
    exec 9>"$LOGDIR/.tick.lock" || { log "lock open failed ($LOGDIR/.tick.lock)"; exit 1; }
    flock -n 9 || { log "tick skipped (lock held)"; exit 0; }
else
    log "flock not on PATH, running unlocked"
fi

# Stock variant only (2026-10-06): call/spread stay paused and are not
# stepped in production. Intraday ticks get a timeout under the 1-minute
# cadence so a hung quote pull can't hold the lock into the next minute's
# trigger; the EOD run keeps gapbot.service's TimeoutStartSec as its
# ceiling. coreutils timeout(1) is on the VPS; without it the service
# timeout still applies. GAPBOT_TIMEOUT (seconds) is a test hook, unset in
# production.
RUN=("$PY" "$SCRIPT" --variants stock)
if [ "$MODE" = intraday ] && command -v timeout >/dev/null 2>&1; then
    RUN=(timeout "${GAPBOT_TIMEOUT:-50}" "${RUN[@]}")
fi

# First successful intraday poll of the session: the marker holds the ET
# date of the last session that had one. It is written only after a clean
# poll, so after a FAILED first poll the next clean one still announces
# itself (the log shows the recovery, not just the failure).
SESSION_MARK="$LOGDIR/.intraday_session"
FIRST=0
if [ "$MODE" = intraday ] && [ "$(cat "$SESSION_MARK" 2>/dev/null)" != "$(ET +%F)" ]; then
    FIRST=1
fi

if [ "$MODE" = eod ]; then
    log "tick start (eod)"
elif [ "$FIRST" = 1 ]; then
    log "tick start (intraday) -- first poll of the session"
fi
OUT="$(mktemp)"
"${RUN[@]}" > "$OUT" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    cat "$OUT" >> "$LOGDIR/tick.log"
    rm -f "$OUT"
    log "tick FAILED ($MODE), exit $RC"
    exit 1
fi
# run_intraday.py prints "recorded exit"/"recorded entry" for a real event.
EVENT=0
if [ "$MODE" = intraday ] && grep -q 'recorded \(exit\|entry\)' "$OUT"; then
    EVENT=1
fi
if [ "$MODE" = intraday ] && [ "$EVENT" = 0 ] && [ "$FIRST" = 0 ]; then
    rm -f "$OUT"
    exit 0  # quiet no-op poll: nothing recorded, nothing logged
fi
cat "$OUT" >> "$LOGDIR/tick.log"
rm -f "$OUT"
log "tick ok ($MODE)"
[ "$FIRST" = 1 ] && ET +%F > "$SESSION_MARK"

# An intraday poll that recorded nothing changed nothing worth mirroring
# (only its heartbeat file) -- skip the sync so the public state repo
# doesn't collect ~390 no-op commits a day. The EOD run always syncs.
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
