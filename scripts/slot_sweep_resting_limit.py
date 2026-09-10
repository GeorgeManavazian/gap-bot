"""MAX_SLOTS sweep under the resting-limit fill rule (orders rest on the
top-N gaps at the open, filled only where the day's range covers the entry).
Under this rule 204 of 503 sessions in the 2yr window have more touches than
free slots, so the slot count is a binding constraint.

Same 2yr held-out window, same everything else; only live.engine.MAX_SLOTS
is patched per arm, and the replay goes through engine.step_one_day()
itself (the code run_daily.py calls), so n=20 must match the anchor
(174 / +11.45% / -14.59%) as the sanity check.

Reported per arm, beyond the anchor-table columns: the capital each slot
gets (sizing is equity / MAX_SLOTS, so more slots = smaller positions),
mean P&L per trade in dollars, average fraction of equity invested, and
how often the rest-top-N choice had to leave a touch unfilled.
Measurement only -- live/config.py is not changed here.
-> results/slot_sweep_resting_limit.csv (2yr)
-> results/slot_sweep_resting_limit_10yr.csv (`... 10`, direction only)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from live import engine
from live.config import CAPITAL, HORIZON, VARIANTS
from live.fillers import make_filler
from slot_and_priority_sweep import BARS_DIR, load_ticker, TickerView

RESULTS = Path(__file__).parent.parent / "results"
SLOTS = [5, 10, 15, 20, 30, 40, 50]   # 5 and 15 included to see the shape below 20
ANCHOR = {"slots": 20, "n_trades": 174, "return_pct": 11.45, "maxdd_pct": -14.59}
WINDOW_YEARS = int(sys.argv[1]) if len(sys.argv) > 1 else 2   # `... 10` = direction-only 10yr check


def load(years: int = 2):
    """Daily bars for every ticker over the last `years` years, as a list of
    (date, {ticker: bar}) in calendar order, the shape engine.step_one_day()
    consumes."""
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


def run(days, n_slots: int) -> dict:
    engine.MAX_SLOTS = n_slots
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades, equity, invested = [], [], []
    contended = 0
    for today, bars in days:
        # count, before stepping, how many live watches would touch today vs
        # free slots (after today's exits) -- the sessions where rest-top-N
        # had to leave a real touch unfilled.
        exits_today = 0
        for tk, pos in state["open_positions"].items():
            r = bars.get(tk)
            dh = pos["days_held"] + 1
            exits_today += (dh >= HORIZON) if r is None else (
                r["l"] <= pos["stop_price"] or r["h"] >= pos["tp_price"] or dh >= HORIZON)
        free = n_slots - len(state["open_positions"]) + exits_today
        touches = sum(1 for tk, w in state["pending"].items()
                      if w["gap_date"] != today and w["days_waited"] + 1 <= HORIZON
                      and tk in bars and engine.limit_touched(bars[tk], w["gap_open"]))
        contended += touches > free
        res = engine.step_one_day(state, today, bars, filler)
        trades.extend(res["trades"])
        equity.append(res["equity"])
        invested.append(1 - state["cash"] / res["equity"] if res["equity"] else 0.0)
    eq = pd.Series(equity)
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    tdf = pd.DataFrame(trades)
    ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    return {
        "slots": n_slots, "n_trades": len(tdf),
        "win_pct": round(100 * (tdf["pnl_pct"] > 0).mean(), 1),
        "return_pct": round(ret, 2), "maxdd_pct": round(dd, 2),
        "ret_over_maxdd": round(ret / -dd, 2) if dd < 0 else None,
        "capital_per_slot_at_start": round(CAPITAL / n_slots),
        "mean_pnl_per_trade_usd": round(tdf["pnl_dollar"].mean(), 0),
        "mean_pnl_per_trade_pct": round(tdf["pnl_pct"].mean(), 2),
        "avg_pct_equity_invested": round(100 * sum(invested) / len(invested), 1),
        "max_open_at_once": int(pd.Series([s for s in _open_counts(days, n_slots)]).max()),
        "sessions_touches_gt_free": contended,
    }


def _open_counts(days, n_slots):
    """Second cheap pass just for the peak number of simultaneously open
    positions -- tells whether the cap was ever actually reached."""
    engine.MAX_SLOTS = n_slots
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    for today, bars in days:
        engine.step_one_day(state, today, bars, filler)
        yield len(state["open_positions"])


def main():
    days = load(WINDOW_YEARS)
    print(f"{WINDOW_YEARS}yr window: {len(days)} sessions, {days[0][0]} -> {days[-1][0]}")
    rows = [run(days, n) for n in SLOTS]
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if WINDOW_YEARS == 2 else f"_{WINDOW_YEARS}yr"
    df.to_csv(RESULTS / f"slot_sweep_resting_limit{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    if WINDOW_YEARS != 2:
        print("\n(10yr = direction-only: survivorship-biased universe)")
        return 0
    a = df[df.slots == ANCHOR["slots"]].iloc[0]
    ok = (a.n_trades == ANCHOR["n_trades"] and abs(a.return_pct - ANCHOR["return_pct"]) < 0.05
          and abs(a.maxdd_pct - ANCHOR["maxdd_pct"]) < 0.05)
    print("\nanchor reproduction at 20 slots:", "MATCH" if ok else "MISMATCH -- do not trust the other rows")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
