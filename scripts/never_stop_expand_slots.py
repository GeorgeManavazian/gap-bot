"""Never-stop, expand-the-book variant (owner idea, 2026-10-05).

Rule under test: 10 base slots, NO stop loss, NO time exit. A position still
open after TRIGGER sessions does not close; it gets its own extra slot
(MAX_SLOTS = BASE + number of such "stuck" positions), so the 10 normal slots
keep trading around it. When the stuck position finally hits its take profit
(prior close) the book shrinks back to BASE. Entry rule, priority, slippage,
liquidity floor and the 2yr window are the honest-rule anchor's, untouched.

Everything goes through live.engine.step_one_day(); this script only patches
module globals per day (same approach as slot_sweep_honest_rule.py):
  - stop_for_abs_gap -> 100  (stop price 0, a low can never touch it)
  - days_held is capped below HORIZON before each step (no time exit); the
    real age is tracked here from entry_date
  - MAX_SLOTS is set each morning to BASE + stuck count
  - fill_slots is replaced by a copy that sizes off SIZE_DIV instead of
    MAX_SLOTS, so an extra slot can carry a full-size position (arm A)

Arms (2yr held-out by default; `... 10` = 10yr, direction-only):
  D  anchor-10     : 10 slots, stops 7/11/16/26/37, 63d exit  (must be 76 / +25.16% / -11.09%)
  E  10 nostop+63d : 10 slots, no stop, 63d time exit kept     (isolates the stop)
  C  10 nostop-never: 10 slots, no stop, no time exit, NO expansion (isolates expansion)
  A  expand, size 1/10 : the rule as stated, each position equity/10 (cash can run out)
  B  expand, size 1/(10+stuck): engine-native sizing, positions shrink as the book grows
  A20/A45 : arm A with trigger 20 / 45 sessions
Mark-to-market equity, so unrealized losses on stuck positions count in
return and drawdown. -> results/never_stop_expand_slots[_10yr].csv (+ arm-A trade log)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from live import engine
from live import config
from live.config import CAPITAL, HORIZON, VARIANTS
from live.fillers import make_filler
import priority_lookahead_check as _plc

# The cached parquet stores Date as datetime64[ms]; with this env's numpy the
# TickerView lookup (np.datetime64(Timestamp) -> us) never matches an ms key,
# so every session loads with zero bars. Normalize to us (what
# np.datetime64(Timestamp) yields) here, locally.
_orig_load_ticker = _plc.load_ticker


def _load_ticker_ns(path):
    d = _orig_load_ticker(path)
    if d is not None:
        d["Date"] = d["Date"].astype("datetime64[us]")
    return d


_plc.load_ticker = _load_ticker_ns
load = _plc.load

RESULTS = Path(__file__).parent.parent / "results"
WINDOW_YEARS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
BASE = 10
_ORIG_STOP = engine.stop_for_abs_gap
_ORIG_FILL = engine.fill_slots
SIZE = {"div": BASE}   # mutable: fill_slots below reads this


def fill_slots_sized(fill_candidates, cash, open_positions, today, bars, filler,
                     chain_provider=None, extra_fields=None):
    """engine.fill_slots, but position size = equity / SIZE['div'] (or
    equity / MAX_SLOTS when div is None). Everything else identical."""
    fill_candidates.sort(key=lambda x: (-abs(x[5]), x[0]))
    free = engine.MAX_SLOTS - len(open_positions)
    filled = 0
    for tk, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date in fill_candidates:
        if filled >= free:
            break
        mtm = sum(filler.mark(pp, bars.get(p, {}).get("c", pp["entry"]), chain_provider)
                  for p, pp in open_positions.items())
        equity_now = cash + mtm
        size_dollars = equity_now / (SIZE["div"] or engine.MAX_SLOTS)
        if engine.MAX_PCT_OF_ADV is not None:
            adv = bars.get(tk, {}).get("adv")
            if adv is not None and size_dollars > engine.MAX_PCT_OF_ADV * adv:
                continue
        entry_price_slipped = entry_price * (1 + engine.SLIP)
        quote = filler.price_entry(tk, entry_price_slipped, size_dollars, chain_provider, tp_price=tp_price)
        if quote is None:
            continue
        if quote.cash_cost > cash:
            continue
        cash -= quote.cash_cost
        open_positions[tk] = {
            "ticker": tk, "entry": entry_price_slipped,
            "stop_price": entry_price_slipped * (1 - stop_w / 100), "tp_price": tp_price,
            "days_held": 0, "units": quote.units, "bucket": bucket,
            "gap_pct": round(gap_pct, 3), "gap_date": gap_date, "entry_date": today,
            "position_dollars": size_dollars, "cash_cost": quote.cash_cost,
            **quote.extra, **(extra_fields or {}),
        }
        filled += 1
    return cash


def run(days, name, *, stops, time_exit, expand, trigger=30, size_div=BASE):
    idx = {d: i for i, (d, _) in enumerate(days)}
    engine.stop_for_abs_gap = _ORIG_STOP if stops else (lambda g: 100.0)
    engine.fill_slots = fill_slots_sized
    SIZE["div"] = size_div
    filler = make_filler("stock", VARIANTS["stock"])
    state = {"cash": CAPITAL, "open_positions": {}, "pending": {}, "last_run_date": None}
    trades, equity, invested, n_open, n_stuck = [], [], [], [], []
    for today, bars in days:
        ti = idx[today]
        stuck = 0
        for pos in state["open_positions"].values():
            if ti - idx[pos["entry_date"]] >= trigger:
                stuck += 1
            if not time_exit:
                pos["days_held"] = min(pos["days_held"], HORIZON - 3)
        engine.MAX_SLOTS = BASE + (stuck if expand else 0)
        res = engine.step_one_day(state, today, bars, filler)
        for t in res["trades"]:
            t["hold_sessions"] = ti - idx[t["entry_date"]]
        trades.extend(res["trades"])
        equity.append(res["equity"])
        invested.append(1 - state["cash"] / res["equity"] if res["equity"] else 0.0)
        n_open.append(len(state["open_positions"]))
        n_stuck.append(stuck)
    eq = pd.Series(equity)
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    tdf = pd.DataFrame(trades)
    # still-open positions at the end, marked at last close
    last_ti = len(days) - 1
    openp = state["open_positions"]
    unreal = [(p["last"] / p["entry"] - 1) * 100 for p in openp.values()]
    unreal_usd = sum(p["units"] * p["last"] - p["cash_cost"] for p in openp.values())
    row = {
        "arm": name, "closed_trades": len(tdf),
        "win_pct_closed": round(100 * (tdf["pnl_pct"] > 0).mean(), 1) if len(tdf) else None,
        "return_pct": round(ret, 2), "maxdd_pct": round(dd, 2),
        "ret_over_maxdd": round(ret / -dd, 2) if dd < 0 else None,
        "open_at_end": len(openp),
        "open_underwater": sum(u < 0 for u in unreal),
        "worst_open_pct": round(min(unreal), 1) if unreal else None,
        "unrealized_usd": round(unreal_usd),
        "max_open": max(n_open), "max_stuck": max(n_stuck),
        "avg_pct_invested": round(100 * sum(invested) / len(invested), 1),
        "max_pct_invested": round(100 * max(invested), 1),
        "median_hold_closed": float(tdf["hold_sessions"].median()) if len(tdf) else None,
        "max_hold_closed": int(tdf["hold_sessions"].max()) if len(tdf) else None,
    }
    return row, tdf, openp


def main():
    days = load(WINDOW_YEARS)
    print(f"{WINDOW_YEARS}yr window: {len(days)} sessions, {days[0][0]} -> {days[-1][0]}")
    arms = [
        ("D anchor-10 (stops, 63d)",        dict(stops=True,  time_exit=True,  expand=False)),
        ("E 10 nostop, 63d exit",           dict(stops=False, time_exit=True,  expand=False)),
        ("C 10 nostop, never exit, no expand", dict(stops=False, time_exit=False, expand=False)),
        ("A expand@30, size 1/10",          dict(stops=False, time_exit=False, expand=True)),
        ("B expand@30, size 1/(10+stuck)",  dict(stops=False, time_exit=False, expand=True, size_div=None)),
        ("A20 expand@20, size 1/10",        dict(stops=False, time_exit=False, expand=True, trigger=20)),
        ("A45 expand@45, size 1/10",        dict(stops=False, time_exit=False, expand=True, trigger=45)),
    ]
    rows, a_trades, a_open, e_trades = [], None, None, None
    for name, kw in arms:
        row, tdf, openp = run(days, name, **kw)
        rows.append(row)
        if name.startswith("E "):
            e_trades = tdf
        if name.startswith("A expand@30"):
            a_trades, a_open = tdf, openp
    engine.stop_for_abs_gap = _ORIG_STOP
    engine.fill_slots = _ORIG_FILL
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if WINDOW_YEARS == 2 else f"_{WINDOW_YEARS}yr"
    df.to_csv(RESULTS / f"never_stop_expand_slots{suffix}.csv", index=False)
    if a_trades is not None:
        a_trades.to_csv(RESULTS / f"never_stop_expand_slots_armA_trades{suffix}.csv", index=False)
    if e_trades is not None:
        e_trades.to_csv(RESULTS / f"never_stop_63d_exit_10slots_trades{suffix}.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))
    if a_open:
        print("\narm A positions still open at end (entry -> last, %):")
        for tk, p in sorted(a_open.items(), key=lambda kv: kv[1]["last"] / kv[1]["entry"]):
            print(f"  {tk:6s} entered {p['entry_date']}  {(p['last']/p['entry']-1)*100:7.1f}%")
    if WINDOW_YEARS == 2:
        d = df.iloc[0]
        ok = d.closed_trades == 76 and abs(d.return_pct - 25.16) < 0.05 and abs(d.maxdd_pct + 11.09) < 0.05
        print("\narm D reproduces the 10-slot sweep row:", "MATCH" if ok else "MISMATCH -- do not trust the other rows")
        return 0 if ok else 1
    print("\n(10yr = direction-only; survivorship-biased universe, and never-stop is exactly the rule that bias flatters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
