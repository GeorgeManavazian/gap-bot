# Real-money execution layer for the gap bot — design spec

Date: 2026-10-11. Status: draft for owner review. Author: Claude session with the owner.

## 0. Summary

The paper bot (`live/`) reads Schwab market data and writes a paper ledger. It
has no order code. This spec adds a separate execution layer that turns the
same engine decisions into real Schwab orders in the owner's existing margin
account, under a fixed dollar allocation, with the ledger driven by Schwab
fills rather than paper assumptions. The paper bot keeps running untouched
alongside it.

Architecture: a **desired-state reconciler**. Every tick the engine's rules
produce a desired order book (resting entries, protective exits, a
market-on-close on day 63). The reconciler pulls Schwab's actual orders and
positions, diffs, and issues the minimum set of place/cancel actions. Fills,
partials, rejects and cancels come back from Schwab and are booked from there.
A crash mid-tick leaves nothing to undo: the next tick re-reads broker truth
and converges.

Decisions made with the owner on 2026-10-11:

| question | decision |
|---|---|
| account | the existing shared Schwab margin account; bot trades a fixed dollar allocation next to the owner's manual positions |
| entry order | stop-limit buy when price is below `gap_open`, limit buy when above (mirrors paper's `low <= gap_open <= high`); GTC, regular session |
| architecture | desired-state reconciler, ledger from Schwab fills only |
| sizing | `GAPBOT_CAPITAL` dollars; bot equity = capital + bot P&L; never the account's balance |

Two findings that changed the original scope:

1. **A plain GTC limit buy at `gap_open` is not the backtested rule.** A limit
   buy fills on any print at or below the limit. Paper fills only when price
   trades *through* `gap_open`. 88 of the anchor's 174 fills came on days the
   stock opened *below* `gap_open` and rose to it; a limit buy would also have
   filled every day a stock opened below and kept falling, trades the backtest
   never took. The mirror of `limit_touched` needs two order types (§3.2).
2. **The Schwab account is shared and on margin.** Read-only API check
   2026-10-11: one account, type MARGIN, liquidation value about $8.9k, six
   owner positions worth about $11.9k, cash balance $0 (about $3k borrowed),
   138 orders in history. The bot must size off its own allocation, must never
   add to the margin debit, and must never touch a position or order it did not
   create.

## 1. Goals and non-goals

Goals:

- Place and manage real orders that implement the live engine's rules
  (resting-limit entry, rest-top-N priority, ramp slot limit, 5% order floor,
  prior-close take-profit, flat 50% stop, 63-session time exit, equity/9
  sizing, 3%-of-ADV cap, one position per ticker).
- Ledger from broker truth: positions, fills, partial fills, rejects, cancels,
  expiries all come from Schwab's order and position endpoints.
- Safety: a hard paper/dry-run/live switch defaulting to off, kill rules
  enforced in code, a dry-run that logs and previews every order it would send,
  idempotent submission.
- Operations: token-age warning, failure alerting, a go-live checklist with
  measurable finish lines.

Non-goals (explicitly out):

- Options, shorting, margin borrowing, more than one account.
- Validating the ramp profile. Paper fills are that evidence; this spec only
  makes real orders possible.
- A real-money dashboard view. The public mirror must not carry real account
  data at all (§7).
- Changing any paper-bot rule or output. Paper gates stay byte-identical.

## 2. Components

All new code lives in `live/` next to the paper modules. Paper modules are
touched only where noted, additively.

| module | responsibility | may call Schwab order endpoints |
|---|---|---|
| `live/broker.py` | the only module that imports `place_order`, `cancel_order`, `preview_order`. Wraps schwab-py 1.5.1. Constructor asserts the execution switch (§6.1) and pins one account hash; refuses any other. Pulls orders, positions, balances. | yes |
| `live/orders.py` | pure builders returning order-spec dicts: stop-limit buy, limit buy, OCO (limit sell + stop sell), market-on-close sell. No I/O. | no |
| `live/reconcile.py` | pure functions: desired order book from engine state + broker snapshot; diff desired vs actual → list of actions; per-order sanity checks; kill-rule evaluation. No I/O. | no |
| `live/real_state.py` | the real ledger under `data/live/real/`: positions registry, intent journal, equity snapshots, high-water mark, halt flag. Atomic writes, idempotent appends (same helpers as `live/state.py`). | no |
| `live/run_execution.py` | entry point. Modes `intraday` and `eod`. Loads state, syncs broker, evaluates kill rules, builds desired book, diffs, executes, persists, writes heartbeat. | via `broker.py` only |
| `live/tokenage.py` | copied from the wheel bot (`etf-bot/live/tokenage.py`): refresh-token age from `creation_timestamp`. | no |
| `live/alerts.py` | copied from the wheel bot: SMTP email with spool and retry; `send_alert` never raises. Config `~/.gapbot/alerts.json`, subject prefix `Gap Bot:`. | no |
| `live/engine.py` (change) | extract step 2a/2b (new-gap registration, watch advance/expiry) into one pure function `register_and_advance(state, today, bars, exclude=())`. `step_one_day` calls it. Anchor gates must still print MATCH 174 / +11.45% / −14.59%. | no |
| `live/run_daily.py` (change) | after `bars_for_today`, write the dict to `data/live/bars/<session>.json` (tiny, additive). The real EOD step reads it so both bots see identical inputs; if missing, the real step pulls bars itself. | no |
| `live/sync.py` (change) | rsync excludes `real/` and `bars/`. | no |
| `scripts/gap_bot_tick.sh` (change) | runs `run_execution.py` after the paper runner in both windows when the switch is not `off`; token-age warning; failure alert (§7). | no |
| `deploy/gapbot.service` (change) | `Environment=GAPBOT_EXECUTION=off` written explicitly, so the unit shows the mode. | no |

Data layout, all under `data/live/real/` (never mirrored):

```
state.json          cash_bot, positions registry, pending watches, hwm, halted, last_run_date, counters
orders.jsonl        intent journal, append-only (§4.2)
fills.jsonl         one row per execution leg booked from Schwab (§4.1)
trades.jsonl        one row per closed bot position, same columns as the paper trades log
snapshots.jsonl     one row per session: equity, cash, n_open, n_pending, hwm
orders-dryrun.jsonl every order the dry-run would have sent, with Schwab's preview response
intraday.json       heartbeat, overwritten each poll
HALT                presence = halted; the owner deletes it to resume
```

## 3. Desired order book

Built from the real ledger's state on every tick. "Session" is the NYSE
session dated from SPY's last regular trade, exactly as the paper poller does;
the wall clock is never the date.

### 3.1 Watches (pending)

Identical to paper. The EOD step calls `engine.register_and_advance` on the
real state with the day's bars. Differences from paper:

- `exclude` = every ticker that has a position in the account and is not in
  the bot's registry (the owner's holdings). Such a ticker never registers. If
  the owner buys a ticker whose watch is already pending, the watch is dropped
  and its entry order canceled at the next tick.
- Consumption: a watch is consumed by (a) any real fill on its entry order,
  (b) a touch (`limit_touched` on the running session range) while the watch
  is **not** in the resting set (the missed wick, as paper), (c) expiry after
  63 sessions. A touch on a resting watch whose order did not fill is **not**
  consumption: the order keeps resting. That is the queue risk paper cannot
  see; the shadow stage measures it (§8).

### 3.2 Entry orders

For each watch in `resting_orders(live, pending, free)` with
`free = slot_limit(bot_positions, pending) − len(bot_positions)` and
`days_waited + 1 <= HORIZON` (the poller's own eligibility rule), one working
buy order:

| last price vs `gap_open` at placement | order | fills when |
|---|---|---|
| below | `STOP_LIMIT`, stop = `gap_open`, limit = `gap_open × 1.001` | price rises to `gap_open` |
| at or above | `LIMIT` at `gap_open` | price dips to `gap_open` |

"Last price at placement" is the session close from the bars file when the
order is placed in the EOD window, and the quote's `lastPrice` when it is
placed intraday (a slot freed mid-session). Duration GTC, session NORMAL
(never extended hours), quantity fixed at placement (§3.4). The side is chosen once, when the order is placed; it does
not flip while working, because a working order on either side fills exactly
when price crosses the level, which is what `limit_touched` books. An order
placed in the EOD window after session G can first fill in session G+1, which
preserves the same-day rule.

Cancel the order when: the watch leaves the resting set (a bigger gap arrived
or a slot filled), the watch is consumed or expired, the ticker becomes
owner-held, the bot halts (§6.2), or the EOD re-size changes the quantity
(cancel and place; never `replace_order`, so the journal stays a chain of
simple intents).

### 3.3 Exit orders

On any fill of an entry order the position is registered and one OCO order
is placed for the broker quantity: `SELL LIMIT` at `prior_close` (the
take-profit) and `SELL STOP` at `stop_price = fill_price × (1 − 0.50)` under
the ramp profile (or the bucket stop under anchor), both GTC, session NORMAL.
Stop price is computed from the **real** average fill price, not `gap_open`.

A partial fill registers the filled quantity and places the OCO on it. The
unfilled remainder is canceled on the next tick (the touch consumed the
watch). A later fill before that cancel lands adds to the position and the
OCO is cancelled and re-placed at the new quantity.

If the broker position quantity is ever below the bot's registered quantity
(the owner sold some), the bot adopts the broker quantity, re-places the OCO,
and alerts. If above, the surplus is the owner's and is ignored.

### 3.4 Sizing

```
bot_cash   = GAPBOT_CAPITAL − Σ cost of open bot positions + Σ realized proceeds
bot_equity = bot_cash + Σ (broker qty × last price) over bot positions
size_usd   = bot_equity / SIZE_DIV            # 9 under ramp, MAX_SLOTS under anchor
qty        = floor(size_usd / limit_price)    # whole shares; 0 → skip and log
qty        = min(qty, max_shares)             # max_shares from the ARMED file, 1 in the shadow stage
```

The 3%-of-ADV cap applies as in paper (`adv` from the bars file); it will not
bite at this size but the code path stays. Quantity is computed when the order
is placed and again at the EOD rebuild; intraday equity drift does not churn
orders.

### 3.5 Time exit (day 63)

Exit session = `entry_date` + 63 NYSE sessions (`np.busday_offset` on the
poller's calendar), which is the session on which the paper engine's
`days_held` reaches 63. On that session, from 15:45 ET:

1. cancel the OCO; poll `get_order` until status is `CANCELED`;
2. place `MARKET_ON_CLOSE` sell, duration DAY, for the broker quantity.

If the cancel is not confirmed by 15:55 ET, leave the OCO working, alert, and
sell at market at the next session's first poll. NYSE's MOC cutoff is 15:50,
so 15:45 leaves margin for the confirmation round trip.

The paper engine's `time_exit_no_data` (a ticker with no bars for the whole
horizon) has no real equivalent; a halted or delisted holding stays in the
account and is alerted on every EOD until the owner resolves it.

## 4. Broker truth and idempotency

### 4.1 Booking from Schwab

Each tick, `broker.sync()` pulls orders entered in the last 100 calendar days
(the oldest GTC entry can be 64 sessions old) and the account's positions and
balances. Every journaled order is matched by `orderId`. Schwab statuses map
as:

| Schwab status | ledger meaning |
|---|---|
| `NEW`, `ACCEPTED`, `QUEUED`, `WORKING`, `PENDING_ACTIVATION`, `PENDING_ACKNOWLEDGEMENT`, `AWAITING_STOP_CONDITION`, `AWAITING_PARENT_ORDER`, `AWAITING_CONDITION`, `AWAITING_RELEASE_TIME` | working |
| `FILLED` | filled; book every execution leg (price, quantity, time) to `fills.jsonl` |
| `CANCELED`, `EXPIRED`, `REJECTED`, `REPLACED` | terminal; a reject is alerted |
| `PENDING_CANCEL`, `PENDING_REPLACE` | transitional; take no action on that order this tick |
| `AWAITING_MANUAL_REVIEW`, `AWAITING_UR_OUT`, `PENDING_RECALL`, `UNKNOWN` | anomaly; no new entries until resolved, alert |

Partial fills: a working order with `filledQuantity > 0` books its legs as
they appear (legs are keyed by `(orderId, leg time, quantity)` so re-reads are
idempotent). An OCO parent reports its two children; the filled child closes
the position, the other is canceled by Schwab.

Positions: the broker's position list is the truth for quantity. Entry price,
`gap_date`, `gap_pct`, `prior_close`, `stop_price`, `tp_price`, `entry_date`
and `exit_session` are bot metadata kept in the registry.

### 4.2 Intent journal and crash safety

Schwab assigns `orderId` server-side and has no client order id. The journal
makes submission idempotent anyway:

1. append `{"intent_id", "ts", "ticker", "side", "type", "qty", "price", "stop", "state": "sending"}`;
2. call `place_order`; read `orderId` from the response `Location` header;
3. append `{"intent_id", "orderId", "state": "sent"}`;
4. later ticks append terminal states as Schwab reports them.

Recovery on the next tick for any intent still in `sending`: scan working
orders for one with identical (symbol, instruction, orderType, quantity,
price, stopPrice) entered after the intent's timestamp and not in the journal;
adopt it. If none exists, the place call never landed; the diff re-issues it.
The bot never places a second working order for the same ticker and side.
The bot cancels only `orderId`s in its own journal and sells only tickers in
its own registry; every other order and position in the account is the
owner's and is never read for anything but the exclusion list and the
no-margin guard.

## 5. Tick flow

`gap_bot_tick.sh` already gates by ET window and holds a lock. When
`GAPBOT_EXECUTION` is not `off`, it runs `run_execution.py` after the paper
runner, under the same lock, with its own timeout (intraday 50 s total for
both runners is too tight; the intraday budget becomes 25 s paper + 30 s
execution, EOD keeps the unit's 1800 s ceiling).

Intraday mode (09:31–16:00 ET, every minute):

1. refuse unless the switch and ARMED file agree (§6.1); refuse if the real
   ledger's `last_run_date` is not the previous session (stale; exits still
   managed, no new entries);
2. `broker.sync()`: orders, positions, balances; book fills; detect anomalies;
3. compute `bot_equity`; evaluate kill rules; halt if any trips (§6.2);
4. one batched quote pull on the watched names (same `fetch_quotes` as the
   paper poller, separate call so the two bots stay independent);
5. consume touched non-resting watches;
6. build the desired book (§3), diff against actual, run per-order sanity,
   execute actions through `broker.py`, journal each one;
7. on an exit session after 15:45, run the MOC sequence (§3.5);
8. persist state atomically, write `intraday.json`.

EOD mode (17:00–23:30 ET, minute % 5, after the paper EOD run):

1. same refusals as intraday plus the paper runner's own staleness rule;
2. read `data/live/bars/<session>.json` (pull if absent);
3. `broker.sync()`; book fills; refresh the owner-held exclusion list;
4. `engine.register_and_advance` on the real state with the exclusion list;
5. re-size and re-side every desired entry order; cancel/place as needed;
6. snapshot equity; update the high-water mark; store the session's closing
   equity as tomorrow's daily-loss baseline; set `last_run_date`.

Exits are managed in every mode, including halted and stale: an open position
always has either a working OCO or a pending MOC.

## 6. Safety

### 6.1 The switch

`GAPBOT_EXECUTION` ∈ `off` (default when unset), `dryrun`, `live`.

- `off`: the tick never invokes `run_execution.py`; `broker.py` is never
  imported by any running process.
- `dryrun`: the full tick runs, but `broker.py` replaces `place_order` with
  `preview_order` (Schwab validates the order without placing it) and
  `cancel_order` with a log line. Every would-be action and Schwab's preview
  response go to `orders-dryrun.jsonl`. Nothing is placed, ever.
- `live`: additionally requires `~/.gapbot/ARMED` (mode 0600) containing JSON
  `{"account_hash": …, "capital": …, "max_shares": …}`. The hash must equal
  the pinned account's; `capital` must equal `GAPBOT_CAPITAL`; `max_shares`
  caps every entry quantity. Missing or mismatched → refuse, alert, exit 1.

`broker.py`'s constructor evaluates all of this and raises before any client
is built. There is no code path from the paper runners to `broker.py`.

### 6.2 Kill rules, enforced in code

| rule | trigger | action |
|---|---|---|
| drawdown | `bot_equity <= 0.80 × hwm` | HALT |
| daily loss | `bot_equity <= 0.95 × previous session's closing equity` | HALT |
| position cap | bot positions ≥ `slot_limit`, and never more than 9 | refuse new entries |
| order rate | > 10 place/cancel actions in one tick or > 40 in one session | HALT |
| no-margin guard | Σ notional of working bot buy orders + the new order > `currentBalances.cashBalance` | skip the entry, log; alert once per session |
| reconciliation anomaly | unknown status, unmatched intent older than one tick, position/registry mismatch the rules above cannot resolve | no new entries, alert |
| stale ledger | real `last_run_date` is not the previous session | no new entries, exits managed |
| corporate action suspected | Schwab `prior_close` differs from the stored `prior_close` by more than 20% for a pending watch or a position | cancel the watch's order and drop it; alert on a position, no automatic action |

HALT means: cancel every bot entry order, keep every exit order, place no new
entries, write `data/live/real/HALT`, set `halted` in state, send one alert.
The bot stays halted across ticks and restarts until the owner deletes the
file. The daily-loss rule arms once per session; the owner's deletion of the
file is the only resume path for both rules.

Per-order sanity, checked in `reconcile.py` before any place call, all must
hold: limit/stop within 20% of the last quote; quantity ≥ 1 and ≤
`max_shares`; buy notional ≤ 1.1 × `bot_equity / SIZE_DIV`; sell quantity ≤
the bot's registered quantity for that ticker; instruction is never
`SELL_SHORT` or `BUY_TO_COVER`; the ticker is not owner-held for a buy. A
failed check drops that action and alerts; it never halts by itself.

### 6.3 Funding

The account currently has a $0 cash balance and a margin debit. The
no-margin guard will skip every entry until the owner deposits the allocation
in cash (or sells enough to leave `cashBalance ≥ GAPBOT_CAPITAL`). The bot
never increases a debit; the owner's own trades can still consume the cash,
in which case entries skip and the session alert says so.

## 7. Operations

- **Token age** (`live/tokenage.py`): the tick logs the refresh-token age on
  the session's first poll; at ≥ 5 days it also emails once per day. The token
  dies at 7; the owner re-logs with `etf-bot/scripts/schwab/schwab_login.py`.
  Resting orders at Schwab keep protecting positions while the bot is blind.
- **Failure alert**: a `tick FAILED` for either runner emails, debounced to
  one per hour per mode. `send_alert` never raises; a dead mailer degrades to
  a log line and a spool retried on later ticks.
- **Alert config**: `~/.gapbot/alerts.json`, same schema as the wheel bot's
  `~/.wheelbot/alerts.json` (SMTP host/port/user/password/from/to); the owner
  copies it on the VPS.
- **Mirror**: `live/sync.py` excludes `real/` and `bars/`. The public
  `gap-bot-state` repo never carries real account data. Verified by a test
  that lists the rsync result.
- **Deploy**: tarball with `COPYFILE_DISABLE=1`, back up `live/`, `scripts/`,
  units first; full gate on the VPS under pandas 3.0.5; `GAPBOT_EXECUTION`
  stays `off` in the unit until the owner says otherwise.

## 8. Go-live stages and finish lines

Each stage starts only on the owner's explicit go.

| stage | switch | finish line |
|---|---|---|
| 1 build | `off` | all new unit tests pass locally and on the VPS; existing gates unchanged under both profiles (`verify_live_engine` MATCH 174 / +11.45% / −14.59%, `verify_catch_up`, `verify_intraday` anchor and ramp, `verify_ramp_profile` 259 / +330.8% / −17.0%, pytest) |
| 2 dry-run on the VPS | `dryrun` | ≥ 10 sessions; every would-be order passed `preview_order` (zero preview rejects); the dry-run's intended entries equal the paper bot's resting set on every session; zero anomalies |
| 3 shadow | `live`, `max_shares = 1` | ≥ 20 sessions or ≥ 5 completed round trips, whichever is later; ≥ 90% of paper-booked entries on resting names got a real fill the same session; mean entry fill ≤ `gap_open × 1.001`; every exit executed by the designed order (OCO leg or MOC); zero unexplained halts; one real round trip's P&L reconciles to the cent against Schwab's transaction history |
| 4 full size | `live`, `max_shares` removed | the owner sets `GAPBOT_CAPITAL`; the ramp profile's merit is a separate decision on paper evidence |

The fill-assumption number from stage 3 (real fills ÷ paper fills on resting
names) is the input the owner asked for before the first full-size dollar.

## 9. Testing

- `scripts/test_execution.py` (pytest) on an in-memory fake broker that
  implements the same methods as `broker.py` (orders with statuses, positions,
  balances, fills that can be scripted as full, partial, rejected, or delayed):
  desired-book construction for each profile; the diff for every transition
  (new watch, watch leaves resting set, slot freed intraday, expiry, owner
  collision); side selection below/at/above `gap_open`; partial fill → OCO on
  filled quantity and remainder cancel; crash between place and journal →
  adoption; crash after cancel → convergence; every kill rule and its HALT
  side effects; the MOC sequence including an unconfirmed cancel; the
  corporate-action detector; the switch and ARMED checks (`broker.py` refuses
  to construct); the mirror exclusion.
- `scripts/verify_execution_parity.py`: replays a session's `bars/<date>.json`
  through both the paper engine and the real desired-book builder and asserts
  the paper resting set equals the set of desired entry orders.
- Engine extraction is covered by the existing gates; they must print the same
  values before and after.
- The VPS venv has schwab-py; the local venv does not, so order-spec builders
  are tested against literal expected dicts locally and `preview_order` runs
  only on the VPS in stage 2.

## 10. Known divergences from paper, accepted

| divergence | why | effect |
|---|---|---|
| whole shares, min 1, at about $1k per slot | API orders are whole shares | names above the slot size are skipped; sizing is lumpy at this capital |
| a touched resting order may not fill | queue position | measured in stage 3; paper books the fill |
| stop-limit limit = `gap_open × 1.001` | a stop-limit at exactly `gap_open` could miss a fast rise | up to 10 bps worse entry, inside the modeled slippage |
| disaster stop is a plain STOP | a stop-limit might not fill on a gap-through | worse than `stop_price` on a gap-through; paper books `stop_price` |
| time exit via MOC | the closing auction is the real "close" | fills at the official close; paper books the bar close |
| equity marked with last quote | same as the intraday paper poller | none beyond what paper already does |
| no `time_exit_no_data` | a halted holding cannot be sold | alerted daily, manual |
| owner-held tickers excluded from registration | shared account | a small, time-varying universe cut |

Splits: both the backtest's yfinance bars and Schwab's history are
split-adjusted (yfinance OHLC is always split-adjusted; `auto_adjust=False`
only leaves dividends out), so the 2026-10-06 note's "unadjusted" worry does
not apply to bar data. The live exposure is only an order or position held
across a split, handled by the corporate-action rule.

## 11. Open items for the owner

- `GAPBOT_CAPITAL` amount, and funding the cash balance before stage 3.
- Copy `~/.wheelbot/alerts.json` to `~/.gapbot/alerts.json` on the VPS.
- The ramp-vs-anchor profile question stays on paper evidence; this layer
  follows whatever `GAPBOT_PROFILE` the unit sets.
