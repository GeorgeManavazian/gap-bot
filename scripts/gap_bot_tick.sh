#!/bin/bash
# One tick of the gap bot, fired every 5 minutes by systemd (gapbot.timer).
# Market-hours gating happens here in ET so the VPS clock stays UTC, same
# doctrine as the wheel bot's wheelbot_tick.sh. Two windows:
#
#   09:35-16:00 ET  live/run_intraday.py -- one batched quote pull on the
#                   watched names, records a touch / stop / take-profit
#                   within one interval of it happening (2026-09-08). Only
#                   ever acts on watches from PRIOR sessions and positions
#                   opened on prior sessions; registers nothing, advances
#                   no counter. Paper only. Set GAPBOT_INTRADAY=0 in the
#                   environment to leave this window a no-op.
#   17:00-23:30 ET  live/run_daily.py -- the one real decision a day: the
#                   EOD gap-fill scan off the settled daily candle. Also the
#                   backstop for anything the intraday poller missed. Its
#                   own last_run_date guard makes every trigger after the
#                   first successful one a harmless no-op, so this gate only
#                   needs to be "roughly after close on a weekday," not
#                   DST-precise.
#
# The first intraday poll is 09:35, not 09:30: quotes are cumulative for
# the session (running high/low), so nothing in the first five minutes is
# lost, and it keeps the poller clear of the opening print settling.
#
# Reuses the wheel bot's Python venv (PY below) -- same schwab-py, pandas,
# numpy already installed there; gap-bot has no venv of its own on the VPS.
# GAPBOT_REPO / GAPBOT_VENV_PY / GAPBOT_FAKE_* are testability hooks, unset
# in production.
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

if [ "$DOW" -gt 5 ]; then
    exit 0  # weekend, no-op, not even worth logging every 5 minutes
fi
if [ "$HM" -ge 935 ] && [ "$HM" -le 1600 ]; then
    [ "${GAPBOT_INTRADAY:-1}" = "0" ] && exit 0
    MODE=intraday
    SCRIPT="$REPO/live/run_intraday.py"
elif [ "$HM" -ge 1700 ] && [ "$HM" -le 2330 ]; then
    # EOD window: same 17:00-23:30 ET the wheel bot uses, buffer past the
    # 16:00 close for Schwab's daily candle to settle.
    MODE=eod
    SCRIPT="$REPO/live/run_daily.py"
else
    exit 0
fi

log "tick start ($MODE)"
OUT="$(mktemp)"
"$PY" "$SCRIPT" > "$OUT" 2>&1
RC=$?
cat "$OUT" >> "$LOGDIR/tick.log"
if [ "$RC" -ne 0 ]; then
    rm -f "$OUT"
    log "tick FAILED ($MODE), exit $RC"
    exit 1
fi
log "tick ok ($MODE)"

# An intraday poll that recorded nothing changed nothing worth mirroring
# (only its heartbeat file) -- skip the sync so the public state repo
# doesn't collect ~78 no-op commits a day. run_intraday.py prints
# "recorded exit"/"recorded entry" for a real event; the EOD run always
# syncs.
if [ "$MODE" = intraday ] && ! grep -q 'recorded \(exit\|entry\)' "$OUT"; then
    rm -f "$OUT"
    exit 0
fi
rm -f "$OUT"

# Sync is best-effort and must never fail the tick -- the trading decision
# already happened and is already persisted locally; a sync problem is a
# dashboard-freshness issue, not a trading one.
PYTHONPATH="$REPO" "$PY" -c "
from live.sync import sync_state
ok = sync_state('$REPO/data/live', '/home/ubuntu/gap-bot-state-mirror')
print('sync ' + ('ok' if ok else 'skipped/failed'))
" >> "$LOGDIR/tick.log" 2>&1 || log "sync step raised, continuing anyway"
