"""The anchor config, frozen here so the live bot trades exactly what the
backtest validated -- no knob here should ever drift from
`11 Gap Bot/_STATUS.md` without that file being updated in the same breath.

Every number traces to a dated finding in the vault:
  - stops (7/11/16/26/37): 8yr-derived, unleaked, scored once held-out
    2026-09-07 -- code/gap-bot/scripts/confirm_2yr_heldout_stops.py
  - priority=gap_desc, MAX_SLOTS=20: swept 10/20/30/40 x alpha/gap_desc on
    10yr, ROBUST -- gap_desc wins at every slot count, 20 sits on a broad
    plateau -- scripts/slot_priority_sensitivity_10yr.py
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
PRIORITY = "gap_desc"
ENTRY_STYLE = "wick"
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
