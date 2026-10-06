"""The anchor config, frozen here so the live bot trades exactly what the
backtest validated -- no knob here should ever drift from
`11 Gap Bot/_STATUS.md` without that file being updated in the same breath.

Every number traces to a dated finding in the vault:
  - ENTRY (2026-09-08, owner ruling): resting-limit fill -- a watch fills
    at gap_open only on a day the price traded through it (low <= gap_open
    <= high), else keeps resting -- and rest-top-N priority: only the
    `free`-slots biggest-gap live watches have an order resting on a
    given day. Both are the rules a real order can execute; the original
    "fill on high >= gap_open, biggest-gap-first among the day's touches"
    booked 276 of its 508 fills at a price the stock opened above and
    needed the close to know the day's touches. Anchor under the traded
    rule: 174 trades, +11.45% / 2yr, maxDD -14.59% --
    scripts/priority_lookahead_check.py, verified by
    scripts/verify_live_engine.py. The rule lives in live/engine.py
    (limit_touched, resting_orders); PRIORITY / ENTRY_STYLE below name it.
  - stops (7/11/16/26/37): 8yr-derived, unleaked, scored once held-out
    2026-09-07 -- code/gap-bot/scripts/confirm_2yr_heldout_stops.py.
    Derived under the ORIGINAL fill rule; not re-derived for the new one.
  - MAX_SLOTS=20: swept 10/20/30/40 x alpha/gap_desc on 10yr under the
    original rule, 20 sat on a broad plateau --
    scripts/slot_priority_sensitivity_10yr.py. Gap-size ordering carried
    into rest-top-N; the slot count was not re-swept under the new rule.
  - EXCLUDE_BELOW=2.0 (drop the 1-2% bucket): beats blending it in on
    every axis, 10yr confirm -- scripts/confirm_10yr.py
  - HORIZON=63 (flat, not per-bucket): a per-bucket timeout was tested
    against the corrected stops and came back a wash -- kept on standby,
    not adopted -- scripts/confirm_heldout_horizon_B.py
  - SLIPPAGE_BPS=10, COMMISSION=0.0: Schwab charges $0 on stock trades;
    10bps is a conservative per-fill slippage stand-in, stress-tested up
    to 50bps without going negative -- scripts/wick_fill_realism_slippage.py
  - MAX_PCT_OF_ADV=0.03: barely bites on S&P 500 liquidity, net positive
    where it does -- scripts/liquidity_floor_check.py
"""
import numpy as np

CAPITAL = 100_000.0
MAX_SLOTS = 20
HORIZON = 63
PRIORITY = "rest_top_n_by_gap"   # was "gap_desc" (whole-day sort) until 2026-09-08
ENTRY_STYLE = "resting_limit"    # was "wick" (fill on high >= gap_open) until 2026-09-08
EXCLUDE_BELOW = 2.0          # 1-2% bucket excluded: scope is (2.0, inf)
COMMISSION_PER_TRADE = 0.0
SLIPPAGE_BPS = 10.0
MAX_PCT_OF_ADV = 0.03
ADV_LOOKBACK = 20

# 8yr-derived, unleaked stop widths (% below entry). Bucket boundaries are
# on ABSOLUTE gap size; a candidate below EXCLUDE_BELOW never reaches the
# stop lookup because it's filtered out earlier.
BUCKETS = [("1-2%", 1.0, 2.0, 7.0), ("2-3%", 2.0, 3.0, 11.0),
           ("3-5%", 3.0, 5.0, 16.0), ("5-10%", 5.0, 10.0, 26.0),
           ("10%+", 10.0, np.inf, 37.0)]


def stop_for_abs_gap(abs_gap):
    for _, lo, hi, stop in BUCKETS:
        if lo <= abs_gap < hi:
            return stop
    return None


def bucket_name_for(abs_gap):
    for name, lo, hi, _ in BUCKETS:
        if lo <= abs_gap < hi:
            return name
    return None


# Three instrument variants trading the IDENTICAL signal above (same
# buckets, stops, slots, priority, horizon) -- only what you buy against a
# wick-fill differs. Each gets its own $100k paper account (live/paths.py),
# so this is three concurrent bots sharing one detector, not one bot with
# three exits. See live/fillers.py for the position economics each name
# below maps to.
VARIANTS = {
    "stock": {
        # unchanged anchor config, live since 2026-09-07.
        "commission_per_trade": COMMISSION_PER_TRADE,
    },
    "call": {
        # Real-chain-data result (vault-71, 2026-09-08,
        # scripts/backtest_long_call_real.py): EVERY (dte, delta) config
        # in {14,21,30,45,60} x {0.3,0.5,0.7} LOST money on the 2yr anchor,
        # real bid/ask, real EOD chains. 14 DTE / 0.30 delta lost the
        # least (-0.41%/trade vs the stock anchor's +1.82%/trade on the
        # same trade subset) -- short DTE minimizes the single biggest
        # damage source found (options expiring before the stock trade
        # even exits: 46-54 of 69 ten-percent-plus-bucket calls did).
        # Forward-tested anyway per owner ruling 2026-09-08: the backtest's
        # chain data was EOD-only, costing ~0.8pt/trade just from missing
        # the stock strategy's own intraday wick-entry timing -- live
        # forward testing gets real same-day quotes and can settle whether
        # that handicap was doing the killing. Expect this to lose money;
        # that is the honest prior, not a bug if it does.
        "target_dte": 14,
        "target_delta": 0.30,
        "commission_per_contract": 0.65,
    },
    "spread": {
        # Real-chain-data result landed (vault-85, 2026-09-08,
        # scripts/debit_spread_real_backtest.py, 4,091 real-quote trades
        # after coverage/liquidity gates): the spread ALSO loses money --
        # 87 of 92 real (bucket, dte, width) cells negative. Real quoted
        # spreads (median 7.8% long leg / 13.9% short leg) are the
        # dominant cost, several times the first-order pass's flat-3%
        # friction assumption -- this flipped the two buckets the
        # first-order model called profitable (5-10%, 10%+) to negative.
        # 7 DTE was the least-bad DTE across the grid (same pattern as
        # the call: shorter DTE avoids the expiry-before-exit tax),
        # though even the best real cell (3-5% bucket, 7 DTE) is
        # ~breakeven at best (+0.12%/trade, n=83) -- not a real edge,
        # inside noise. Forward-tested anyway per owner ruling
        # 2026-09-08, BUT NOTE: unlike the call, the spread's damage
        # (real bid/ask cost) is a liquidity tax, not an EOD-timing
        # artifact -- live intraday fills are less likely to recover
        # this than they might for the call. Flagged to the owner,
        # decision on whether to keep running this variant is theirs.
        "target_dte": 7,
        "long_delta": 0.50,
        "commission_per_contract": 0.65,
    },
}
