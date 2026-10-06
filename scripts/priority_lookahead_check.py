"""Two realism questions the intraday poller (live/intraday.py, 2026-09-08)
surfaced about the anchor's ENTRY rule, measured on the same 2yr window
and the same engine helpers, so the numbers are comparable to the anchor.

Q1. Intraday look-ahead in the priority rule. On a session where more
    prior watches touch gap_open than there are free slots, the backtest
    fills the BIGGEST gaps among that day's touches -- it sees the whole
    day at once. A real-time bot can't: at any moment it only knows which
    watches have touched SO FAR. Two real-time-feasible policies scored:
      rest_top_n  -- each session, rest limit orders on the `free`
                     biggest-gap live watches; only those can fill; a
                     touch on any other watch is a missed wick (watch
                     consumed, as the backtest consumes every touch).
                     Deterministic from daily bars. This is what real
                     order placement would actually do.
      first_touch -- fill touches in the order they happen. Daily bars
                     don't know the order, so it's random over N seeds:
                     an expected value, not a point estimate.
Q2. Fills at a price never traded. Step 2b fills at gap_open whenever
    h >= gap_open; if the day's LOW is also above gap_open (the stock
    gapped up through the level overnight), a resting limit buy at
    gap_open would not fill at all, and the backtest booked an entry
    below the day's range. Counted, not modeled: how many anchor fills
    have l > gap_open, and how many have o > gap_open.

Reproduces the ORIGINAL anchor first (n=508, +72.34%) as the sanity
check that the replay loop below is the pre-2026-09-08 day-step, so the
policy rows are apples to apples. -> results/priority_lookahead_check.csv

Outcome (owner ruling 2026-09-08): the "rest_top_n + resting limit
semantics" row became the engine's actual rule (live/engine.py) and the
new anchor; scripts/verify_live_engine.py now checks the engine against
that row. This script keeps its own inlined copy of the old 2b logic on
purpose -- it is the record of what the old number assumed.
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from live.config import CAPITAL, HORIZON, MAX_SLOTS, EXCLUDE_BELOW, VARIANTS, stop_for_abs_gap, bucket_name_for
from live.engine import close_position, fill_slots, SLIP
from live.fillers import make_filler
from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView

RESULTS = Path(__file__).parent.parent / "results"
FILLER = make_filler("stock", VARIANTS["stock"])
SEEDS = 5


def load(years: int = 2):
    tickers_data, longest = {}, None
    for p in sorted(BARS_DIR.glob("*.parquet")):
        d = load_ticker(p)
        if d is None:
            continue
        tickers_data[p.stem] = TickerView(d)
        if longest is None or len(d) > len(longest):
            longest = d["Date"]
    calendar = pd.DatetimeIndex(sorted(longest))
    calendar = calendar[calendar >= calendar[-1] - pd.DateOffset(years=years)]
    days = []
    for day in calendar:
        bars = {}
        for tk, tv in tickers_data.items():
            row = tv.get(day)
            if row is None:
                continue
            o, h, l, c = row
            pr = tv.get_prior_close(day)
            bars[tk] = {"o": float(o), "h": float(h), "l": float(l), "c": float(c),
                        "prior_close": None if pr is None else float(pr), "adv": tv.get_adv(day)}
        days.append((day.isoformat()[:10], bars))
    return days


def replay(days, policy: str, rng: random.Random | None = None, open_if_above: bool = False,
           limit_semantics: bool = False, skip_if_open_past_tp: bool = False, return_detail: bool = False):
    """engine.step_one_day, inlined only where the policy has to intervene
    (which touched watches become candidates, and in what order).
    `open_if_above`: when the session OPENS above gap_open, a resting limit
    buy at gap_open can't fill there -- book the entry at the open instead
    (the price a real market-on-touch order would get). tp stays at
    prior_close; the stop stays a % below the (now higher) entry.
    `limit_semantics`: model the entry as a resting limit buy at gap_open
    -- it fills only on a day where l <= gap_open <= h (the price actually
    traded there), at gap_open, and otherwise KEEPS resting until HORIZON.
    The anchor's `h >= gap_open` fills even when the whole day traded
    above the level."""
    cash, open_positions, pending = CAPITAL, {}, {}
    trades, equity = [], []
    touches = touch_low_above = touch_open_above = 0   # every touch on an eligible watch
    fill_low_above = fill_open_above = 0               # among positions actually opened
    for today, bars in days:
        # 1) exits -- verbatim engine logic
        for tk in list(open_positions):
            pos = open_positions[tk]
            days_held = pos["days_held"] + 1
            row = bars.get(tk)
            exit_price = reason = None
            if row is None:
                if days_held >= HORIZON:
                    exit_price, reason = pos["entry"], "time_exit_no_data"
                else:
                    pos["days_held"] = days_held
                    continue
            else:
                if row["l"] <= pos["stop_price"]:
                    exit_price, reason = pos["stop_price"], "stop"
                elif row["h"] >= pos["tp_price"]:
                    exit_price, reason = pos["tp_price"], "tp_gap_filled"
                elif days_held >= HORIZON:
                    exit_price, reason = row["c"], "time_exit"
                if exit_price is None:
                    pos["days_held"] = days_held
                    continue
            proceeds, trade = close_position(pos, tk, exit_price, reason, days_held, today, FILLER)
            cash += proceeds
            trades.append(trade)
            del open_positions[tk]
        # 2a) register -- verbatim
        for tk, row in bars.items():
            if tk in open_positions or tk in pending:
                continue
            pr = row.get("prior_close")
            if pr is None or pr <= 0:
                continue
            gap_pct = (row["o"] - pr) / pr * 100
            if gap_pct >= -1.0 or -gap_pct < EXCLUDE_BELOW:
                continue
            pending[tk] = {"gap_open": row["o"], "prior_close": pr, "gap_pct": gap_pct,
                           "gap_date": today, "days_waited": 0}
        # 2b) advance + touches -- verbatim, but the policy decides which
        # touches become candidates
        free = MAX_SLOTS - len(open_positions)
        live = []
        for tk in list(pending):
            w = pending[tk]
            if w["gap_date"] == today:
                continue
            w["days_waited"] += 1
            if w["days_waited"] > HORIZON:
                del pending[tk]
                continue
            if tk in open_positions:
                del pending[tk]
                continue
            live.append(tk)
        eligible = set(live)
        if policy == "rest_top_n":
            eligible = set(sorted(live, key=lambda t: (pending[t]["gap_pct"], t))[:max(free, 0)])
        candidates = []
        for tk in live:
            row = bars.get(tk)
            if row is None:
                continue
            w = pending[tk]
            if limit_semantics:
                touched = row["l"] <= w["gap_open"] <= row["h"]
            else:
                touched = row["h"] >= w["gap_open"]
            if touched:
                if tk in eligible:
                    touches += 1
                    touch_low_above += row["l"] > w["gap_open"]
                    touch_open_above += row["o"] > w["gap_open"]
                    abs_gap = -w["gap_pct"]
                    entry = max(w["gap_open"], row["o"]) if open_if_above else w["gap_open"]
                    if skip_if_open_past_tp and entry >= w["prior_close"]:
                        # stop-order variant, sane version: the gap is already
                        # filled at the open, nothing left to capture -- cancel
                        # the stop instead of buying above the target.
                        del pending[tk]
                        continue
                    candidates.append((tk, entry, w["prior_close"], bucket_name_for(abs_gap),
                                       stop_for_abs_gap(abs_gap), w["gap_pct"], w["gap_date"]))
                del pending[tk]  # a touch consumes the watch, eligible or not
        # 3) fill
        before = set(open_positions)
        if policy == "first_touch":
            rng.shuffle(candidates)
            for c in candidates:  # one at a time so fill_slots' own gap-desc sort can't reorder
                cash = fill_slots([c], cash, open_positions, today, bars, FILLER)
        else:
            cash = fill_slots(candidates, cash, open_positions, today, bars, FILLER)
        for tk in open_positions:
            if tk in before:
                continue
            entry = open_positions[tk]["entry"] / (1 + SLIP)
            fill_low_above += bars[tk]["l"] > entry + 1e-9
            fill_open_above += bars[tk]["o"] > entry + 1e-9
        # 4) mark
        equity.append(cash + sum(FILLER.mark(p, bars.get(t, {}).get("c", p["entry"])) for t, p in open_positions.items()))
    eq = pd.Series(equity)
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    tdf = pd.DataFrame(trades)
    if return_detail:
        return tdf, pd.Series(equity, index=[d for d, _ in days])
    return {"n_trades": len(tdf), "win_pct": round(100 * (tdf["pnl_pct"] > 0).mean(), 1),
            "return_pct": round((eq.iloc[-1] / CAPITAL - 1) * 100, 2), "maxdd_pct": round(dd, 2),
            "touches": touches, "touch_low_above": touch_low_above, "touch_open_above": touch_open_above,
            "fill_low_above": fill_low_above, "fill_open_above": fill_open_above}


def main():
    days = load()
    rows = []
    anchor = replay(days, "gap_desc")
    rows.append({"policy": "gap_desc (anchor, sees whole day)", **anchor})
    ok = anchor["n_trades"] == 508 and abs(anchor["return_pct"] - 72.34) < 0.05
    print(f"anchor reproduction: {'MATCH' if ok else 'MISMATCH -- replay loop drifted from engine, numbers below not comparable'}")
    rows.append({"policy": "gap_desc + entry at open when open > gap_open (price fix only)", **replay(days, "gap_desc", open_if_above=True)})
    rows.append({"policy": "gap_desc + resting limit semantics (fill only if l <= gap_open <= h, else keep resting)", **replay(days, "gap_desc", limit_semantics=True)})
    rows.append({"policy": "rest_top_n (orders on the `free` biggest gaps, real-time feasible)", **replay(days, "rest_top_n")})
    rows.append({"policy": "rest_top_n + STOP-order fill (entry = max(gap_open, open) on h >= gap_open)", **replay(days, "rest_top_n", open_if_above=True)})
    rows.append({"policy": "rest_top_n + STOP-order fill, cancelled when the open is already >= tp", **replay(days, "rest_top_n", open_if_above=True, skip_if_open_past_tp=True)})
    rows.append({"policy": "rest_top_n + resting limit semantics (ANCHOR since 2026-09-08)", **replay(days, "rest_top_n", limit_semantics=True)})
    for seed in range(SEEDS):
        rows.append({"policy": f"first_touch (random order, seed {seed})", **replay(days, "first_touch", random.Random(seed))})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "priority_lookahead_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    ft = df[df.policy.str.startswith("first_touch")]
    print(f"\nfirst_touch mean over {SEEDS} seeds: return {ft.return_pct.mean():.2f}%  maxDD {ft.maxdd_pct.mean():.2f}%  n {ft.n_trades.mean():.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
