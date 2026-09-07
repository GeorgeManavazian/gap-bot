#!/bin/bash
# One tick of the gap bot, fired every 5 minutes by systemd (gapbot.timer).
# Market-hours gating happens here in ET so the VPS clock stays UTC, same
# doctrine as the wheel bot's wheelbot_tick.sh -- but far simpler, since
# this bot makes exactly one decision a day (an EOD gap-fill scan), not an
# intraday exit manager. run_daily.py's own last_run_date guard makes every
# trigger inside the window after the first successful one a harmless no-op,
# so this gate only needs to be "roughly after close on a weekday," not
# DST-precise.
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
# EOD window: same 17:00-23:30 ET the wheel bot uses, buffer past the 16:00
# close for Schwab's daily candle to settle.
if [ "$HM" -lt 1700 ] || [ "$HM" -gt 2330 ]; then
    exit 0
fi

log "tick start"
if "$PY" "$REPO/live/run_daily.py" >> "$LOGDIR/tick.log" 2>&1; then
    log "tick ok"
else
    log "tick FAILED, exit $?"
    exit 1
fi

# Sync is best-effort and must never fail the tick -- the trading decision
# already happened and is already persisted locally; a sync problem is a
# dashboard-freshness issue, not a trading one.
PYTHONPATH="$REPO" "$PY" -c "
from live.sync import sync_state
ok = sync_state('$REPO/data/live', '/home/ubuntu/gap-bot-state-mirror')
print('sync ' + ('ok' if ok else 'skipped/failed'))
" >> "$LOGDIR/tick.log" 2>&1 || log "sync step raised, continuing anyway"
