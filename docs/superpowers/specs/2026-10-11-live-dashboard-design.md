# Live dashboard — design (2026-10-11)

## Goal

The owner opens one page during market hours and sees what the bot sees now:
every watched name's price against its wick level, every open position marked
to market with distance to stop and take-profit, and a fill or exit within a
couple of minutes of the bot booking it. Resolution equals the bot's own
intraday poll (one Schwab quote pull per minute); the page never claims more.

## Why the current pipeline cannot do it

- The dashboard is a Claude artifact. Artifact pages run under a CSP that
  blocks `fetch` to every outside host, and no runtime capability or
  connector reaches GitHub. Best case there is an hourly cloud routine.
- `raw.githubusercontent.com` is served from a CDN with `max-age=300` and
  ignores query strings for cache busting (verified: fresh query, `x-cache:
  HIT`, stale body).
- `api.github.com/repos/.../contents/<path>?ref=<branch>` with
  `Accept: application/vnd.github.raw+json` is fresh within 60s, allows
  cross-origin reads, supports `If-None-Match`, and is capped at 60 requests
  per hour per unauthenticated IP (a 304 still counts).

Decisions made with the owner: host the page on GitHub Pages, feed it through
the GitHub contents API, poll every 75s, accept 1.5–2.5 min lag, retire the
artifact and the nightly routine. No Cloudflare, no VPS endpoint, no domain.

## Architecture

```
VPS (every intraday poll + every EOD run)
  run_intraday.py / run_daily.py  -> data/live/account/{state,intraday}.json, trades.jsonl, snapshots.jsonl
  gap_bot_tick.sh                 -> live/livefeed.py push_live()
                                      builds live.json, amend-commits it on orphan branch `live`
                                      of gap-bot-state, force-pushes (one commit, never grows)
GitHub
  gap-bot-state@live/live.json    (public, the live note)
  gap-bot-state@main/...          (unchanged: EOD history sync, 5-min raw cache is fine)
  gap-bot@main/docs/index.html    (GitHub Pages, the dashboard)
Browser
  georgemanavazian.github.io/gap-bot/  polls the contents API every 75s (market hours)
```

## Part 1 — producer: `live/livefeed.py`

### `build_live_doc(state, heartbeat, snapshots, trades, *, slot_limit, profile, now_et) -> dict`

Pure. Inputs are the loaded files (dicts/lists), output is the whole page
model, so the page never has to join files:

```json
{
  "schema": 1,
  "generated_at": "2026-10-13T10:32:07-0400",
  "polled_at": "2026-10-13T10:32:02-0400",   // heartbeat; null before the first poll of a session
  "session": "2026-10-13",                    // heartbeat session, else state.last_run_date
  "last_run_date": "2026-10-10",              // last EOD run
  "profile": "ramp",
  "slot_limit": 7,
  "equity": 100412.55,                        // heartbeat equity when present, else last snapshot equity, else cash
  "cash": 71234.10,
  "day_start_equity": 100000.0,               // last snapshot equity (prior session close)
  "open_positions": [ {"ticker","bucket","gap_pct","entry","last","stop","tp","h","l","days_held","entry_date","units"} ],
  "pending":        [ {"ticker","bucket","gap_pct","gap_open","prior_close","last","h","l","days_waited","gap_date","consumed_on"} ],
  "today": { "entries": [ {"ticker","entry","time"} ], "exits": [ {"ticker","exit_reason","exit_price","pnl_pct","time"} ] },
  "equity_curve": [ {"date","equity"} ],      // every snapshot, oldest first
  "closed_trades": [ {"ticker","entry_date","exit_date","exit_reason","entry_price","exit_price","pnl_pct","pnl_dollar","days_held"} ]  // newest first, max 20
}
```

- `last`/`h`/`l` come from the heartbeat's `watched` table when present,
  else `last` falls back to the position's stored `last` (set by the EOD run)
  or `entry`; `h`/`l` null.
- `bucket` for pending rows is derived from `gap_pct` with the engine's
  bucket edges (engine already has the function; reuse, do not copy).
- `today.exits` are trades whose `exit_date == session`; `today.entries` are
  open positions whose `entry_date == session` plus exits today that also
  entered today. Times come from `exit_time` / `entry_time` when present.
- `days_held` for a closed trade is computed from the two dates if the trade
  row lacks it.
- Nothing credential-shaped can enter the document (it is built from state
  files only), but `push_live` still runs the existing `secret_guard` on the
  clone dir before every push, same as `sync_state`.

### `push_live(doc, clone_dir) -> bool`

- Loads the credential with `sync._load_cfg()`; missing → print once, return
  False (never raise, never fail the tick).
- Ensures `clone_dir` is a git repo on an orphan branch `live` with the
  bot's identity, creating it on first use.
- Writes `live.json` atomically, `git add -A`, then `git commit -q --amend
  --allow-empty -m "live"` (first commit when no HEAD), `git push -q -f
  <remote> HEAD:refs/heads/live`. Token scrubbed from any error text.
- Returns True on success. No marker ref: the branch is always one commit and
  the next minute overwrites it anyway.
- Logging: prints one line on failure; the tick script is responsible for
  throttling (below).

### Heartbeat change

`run_intraday.write_heartbeat` payload gains `"slot_limit": engine.slot_limit(state["open_positions"], state["pending"])`.

### Tick script

After a successful `run_intraday.py` (quiet or not) and after a successful
`run_daily.py`, run:

```
PYTHONPATH="$REPO" "$PY" -m live.livefeed --push
```

which loads the stock account files, builds the doc, pushes. Its output is
appended to `tick.log` only when it fails, and only on the first failure of
the session (marker file `data/live/logs/.livefeed_fail_session` holding the
ET date), so a dead token shows up once a day, not 390 times. A later success
clears the marker so the next failure logs again. The push runs inside the
existing 50s intraday timeout window (the push itself is ~1–2s). The existing
EOD `sync_state` to `main` stays exactly as it is.

## Part 2 — consumer: `docs/index.html`

Single file, vanilla JS, Google Fonts only, no library. Published by GitHub
Pages from `main` `/docs`. Title "Gap-Fill Bot".

### Data

- `GET https://api.github.com/repos/GeorgeManavazian/gap-bot-state/contents/live.json?ref=live`
  with `Accept: application/vnd.github.raw+json` and `If-None-Match` from the
  last ETag. 304 → keep current model, just bump "checked" time. 200 → parse,
  render.
- Cadence: 75s when the ET clock is a weekday between 09:25 and 16:10, else
  every 10 min. Also refetch immediately on tab focus if the last fetch is
  older than 60s.
- Rate limit: read `X-RateLimit-Remaining` and `X-RateLimit-Reset`. Below 8
  remaining, sleep until reset (shown in the header). On 403/429, same.
- Any fetch error keeps the last good model on screen and shows the error in
  the status line; never a blank page.
- First paint before the first fetch: a designed empty state ("Connecting to
  the bot…").

### Status pill (derived in the page)

- **LIVE** (green): market hours and `polled_at` within the last 3 min.
- **CLOSED** (grey): outside market hours; shows last poll time.
- **STALE** (amber): market hours and `polled_at` older than 3 min, or the
  fetch itself failing. This is the "something is wrong on the VPS" signal.

Header also shows: profile, "as of HH:MM:SS ET · Ns ago", "checked Ns ago",
slots used / `slot_limit`.

### Sections, top to bottom

1. Tiles: equity, day P&L (equity − day_start_equity, $ and %), cash, open /
   slot limit, watching.
2. Gap watch table, sorted by distance to wick ascending: ticker, gap %,
   bucket, wick level (gap_open), last, distance to wick % ((last −
   gap_open)/gap_open; negative = below, i.e. would fill on a touch),
   session L–H, waiting days, badge "touched today" when `consumed_on ==
   session`.
3. Open positions: ticker, bucket, entry, last, P&L %, distance to stop %,
   distance to TP %, held days, entry date. P&L colored.
4. Today: entries and exits as a feed, newest first, with times.
5. Equity curve (SVG, to scale, both themes) with today's live equity as the
   last emphasized point when `polled_at` is in the current session.
6. Closed trades, newest first.
7. Footer: profile rule summary read from the doc (not hardcoded), link to
   the state repo, "data: Schwab quotes via the bot's poll, 1-min".

### Theme and layout

Dark-first token system (the existing artifact is dark; keep that identity),
light theme defined too. Phone width works: tiles 2-up, tables scroll inside
their own container. `tabular-nums` on every number column.

## Part 3 — cutover

1. Enable Pages on `GeorgeManavazian/gap-bot`: source `main`, path `/docs`
   (via `gh api -X POST repos/.../pages`).
2. Deploy producer to the VPS (tarball with `COPYFILE_DISABLE=1`, or scp of the
   changed files: `live/livefeed.py`, `live/run_intraday.py`,
   `scripts/gap_bot_tick.sh`). Dry-run the push on the box, confirm
   `gap-bot-state@live/live.json` exists via the API.
3. Verify end to end on the next weekday tick: tick.log stays quiet, the API
   returns a `polled_at` that advances, the page goes LIVE.
4. Republish the Claude artifact as a one-card pointer to the new URL.
   Disable routine `trig_01WekTjoXaBZcWufztKGZTfW` (`enabled: false`); the
   owner deletes it at claude.ai/code/routines when convenient.
5. `_STATUS.md`, `CLAUDE.md`, `README.md` dashboard references updated.

## Testing

- `scripts/test_livefeed.py` (pytest, etf-bot venv):
  - `build_live_doc`: fixture state + heartbeat + snapshots + trades → exact
    expected rows; pending bucket derivation; `last` fallback chain; today
    entries/exits split; equity fallback chain; `closed_trades` cap and order.
  - `push_live` with `_run` stubbed: first-use init on orphan branch; amend +
    force-push command shapes; missing credential → False and no git calls;
    push failure → False, token scrubbed from printed text; secret guard
    offender → False, no push.
  - CLI `-m live.livefeed --push` exit code 0 on success, 1 on failure
    (so the tick can tell).
- Tick harness (existing `_tick` helper): a quiet intraday poll calls the
  livefeed push; a failing push logs once per session and not on the next
  minute; a later success clears the marker; EOD run calls it too; a push
  failure never changes the tick's exit code.
- Page: open `docs/index.html` locally against a fixture `live.json` served by
  `python -m http.server`; the script reads an optional `?src=<url>` query
  parameter to point at that fixture (local use only; the default is the
  GitHub API URL). One screenshot via the Playwright MCP at desktop and phone
  widths. After deploy, one real load of the Pages URL.

## Out of scope

Tick-by-tick quotes, options variants, authentication, any write path from
the page, historical intraday charts.
