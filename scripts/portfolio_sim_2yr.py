"""Boil-it-down portfolio simulation: one $100k account, max 20 concurrent
positions (slots), walked day-by-day across all 10 years and all 502
tickers -- not per-trade stats pooled, an actual equity curve. Two arms,
EVERYTHING else held identical so the only difference is the price stop:

  WITH SAFETY NET     -- data-derived per-bucket stop (8/11/15/24/35%),
                         same as the last test.
  WITHOUT SAFETY NET  -- no price stop at all.

Both arms share: same signal (down gap >=1%), same entry (buy Open),
same target (TP at prior close), same 20-slot cap, same equal-weight
sizing off CURRENT equity, same 63-trading-day time exit if neither stop
nor TP fires (a position can't be held forever -- the slot has to free up
eventually in both arms, so this is held constant, not a new variable).

Signal selection on a day with more candidates than open slots: filled in
ticker-alphabetical order (a plain, stated, non-cherry-picked tie-break --
no ranking model, this is deliberately the simplest possible rule)."""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
HORIZON = 63
CAPITAL = 100_000.0
MAX_SLOTS = 20
WINDOW_YEARS = 2  # last 2 years only, same universe/rules/sizing as the 10yr run

BUCKET_STOPS = [(1.0, 2.0, 8.0), (2.0, 3.0, 11.0), (3.0, 5.0, 15.0),
               (5.0, 10.0, 24.0), (10.0, np.inf, 35.0)]


def stop_for_gap(abs_gap):
    for lo, hi, stop in BUCKET_STOPS:
        if lo <= abs_gap < hi:
            return stop
    return None


def load_ticker(path):
    df = pd.read_parquet(path)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    need = {"Date", "Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if len(df) < HORIZON + 5:
        return None
    return df


def run_sim(use_stop: bool, tickers_data: dict, calendar: pd.DatetimeIndex,
           flat_stop: float | None = None):
    """flat_stop overrides the per-bucket data-derived stop with one flat
    number (e.g. 50.0) applied to every gap size, when use_stop is True."""
    cash = CAPITAL
    open_positions = {}   # ticker -> dict(entry, stop_price, tp_price, entry_day_idx, size_shares)
    equity_curve = []
    closed_trades = []    # list of pnl_pct realized

    for day_idx, day in enumerate(calendar):
        # 1) process exits on open positions using today's bar
        for tk in list(open_positions):
            pos = open_positions[tk]
            df = tickers_data[tk]
            row = df.get(day)
            if row is None:
                continue  # no data today for this ticker (halt/holiday mismatch), skip
            o, h, l, c = row
            days_held = day_idx - pos["entry_day_idx"]
            exit_price = None
            if use_stop and l <= pos["stop_price"]:
                exit_price = pos["stop_price"]
            elif h >= pos["tp_price"]:
                exit_price = pos["tp_price"]
            elif days_held >= HORIZON:
                exit_price = c  # time exit, forced close at today's close
            if exit_price is not None:
                pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                proceeds = pos["size_shares"] * exit_price
                cash += proceeds
                closed_trades.append(pnl_pct)
                del open_positions[tk]

        # 2) look for new signals today, fill open slots in ticker-alpha order
        free_slots = MAX_SLOTS - len(open_positions)
        if free_slots > 0:
            candidates = []
            for tk in sorted(tickers_data):
                if tk in open_positions:
                    continue
                df = tickers_data[tk]
                today_row = df.get(day)
                prior_row = df.get_prior_close(day)
                if today_row is None or prior_row is None:
                    continue
                o, h, l, c = today_row
                prior_close = prior_row
                if prior_close <= 0:
                    continue
                gap_pct = (o - prior_close) / prior_close * 100
                if gap_pct >= -1.0:
                    continue
                stop_w = flat_stop if flat_stop is not None else stop_for_gap(abs(gap_pct))
                if stop_w is None:
                    continue
                candidates.append((tk, o, prior_close, stop_w))
            for tk, o, prior_close, stop_w in candidates[:free_slots]:
                mtm_now = 0.0
                for p, ppos in open_positions.items():
                    prow = tickers_data[p].get(day)
                    pc = prow[3] if prow is not None else ppos["entry"]
                    mtm_now += ppos["size_shares"] * pc
                equity_now = cash + mtm_now
                size_dollars = equity_now / MAX_SLOTS
                size_shares = size_dollars / o
                cash -= size_shares * o
                open_positions[tk] = {
                    "entry": o, "stop_price": o * (1 - stop_w / 100),
                    "tp_price": prior_close, "entry_day_idx": day_idx,
                    "size_shares": size_shares,
                }

        # 3) mark equity for the curve
        mtm = 0.0
        for tk, pos in open_positions.items():
            row = tickers_data[tk].get(day)
            c = row[3] if row is not None else pos["entry"]
            mtm += pos["size_shares"] * c
        equity_curve.append(cash + mtm)

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100
    return {
        "n_trades": len(closed_trades),
        "avg_win_pct": np.mean(closed_trades) if closed_trades else None,
        "win_rate_pct": 100 * np.mean([p > 0 for p in closed_trades]) if closed_trades else None,
        "max_drawdown_pct": max_dd,
        "final_equity": eq.iloc[-1],
        "total_return_pct": (eq.iloc[-1] / CAPITAL - 1) * 100,
        "equity_curve": eq,
    }


class TickerView:
    """O(1) date -> (o,h,l,c) lookup, plus prior-close lookup, over one
    ticker's bars. Avoids pandas per-row overhead in the day loop."""
    def __init__(self, df):
        self.dates = df["Date"].values
        self.idx = {d: i for i, d in enumerate(self.dates)}
        self.o = df["Open"].values
        self.h = df["High"].values
        self.l = df["Low"].values
        self.c = df["Close"].values

    def get(self, day):
        i = self.idx.get(np.datetime64(day))
        if i is None:
            return None
        return (self.o[i], self.h[i], self.l[i], self.c[i])

    def get_prior_close(self, day):
        i = self.idx.get(np.datetime64(day))
        if i is None or i == 0:
            return None
        return self.c[i - 1]


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    tickers_data = {}
    longest_dates = None
    for p in files:
        d = load_ticker(p)
        if d is None:
            continue
        tv = TickerView(d)
        tickers_data[p.stem] = tv
        if longest_dates is None or len(d) > len(longest_dates):
            longest_dates = d["Date"]
    # master calendar = the ticker with the most trading days on file (AAPL/
    # MSFT-class mega-caps have the fullest, most reliable 10y daily record;
    # no S&P500 index ETF is itself a constituent so none is in this store)
    calendar = pd.DatetimeIndex(sorted(longest_dates))
    if WINDOW_YEARS is not None:
        cutoff = calendar[-1] - pd.DateOffset(years=WINDOW_YEARS)
        calendar = calendar[calendar >= cutoff]
        # prior-close lookups for day 1 of the window still hit each
        # ticker's real history before the cutoff -- TickerView holds the
        # FULL 10yr series regardless of what slice of `calendar` we walk,
        # so the very first day's gap is computed against a real prior
        # close, not a cold-start artifact.
    print(f"{len(tickers_data)} tickers, {len(calendar)} trading days "
          f"({calendar[0].date()} -> {calendar[-1].date()})"
          f"{f'  [{WINDOW_YEARS}yr window]' if WINDOW_YEARS else ''}")

    arms = {}
    print("\nrunning WITH safety net (data-derived 8-35%) ...")
    arms["tight (8-35%, data)"] = run_sim(True, tickers_data, calendar)
    print("running flat 50% stop ...")
    arms["flat 50%"] = run_sim(True, tickers_data, calendar, flat_stop=50.0)
    print("running flat 65% stop ...")
    arms["flat 65%"] = run_sim(True, tickers_data, calendar, flat_stop=65.0)
    print("running flat 80% stop ...")
    arms["flat 80%"] = run_sim(True, tickers_data, calendar, flat_stop=80.0)
    print("running WITHOUT safety net ...")
    arms["no stop"] = run_sim(False, tickers_data, calendar)

    print("\n" + "=" * 130)
    header = "".join(f"{name:>20s}" for name in arms)
    print(f"{'':28s}{header}")
    print("=" * 130)
    for k, label in [("n_trades", "trades taken"),
                     ("avg_win_pct", "avg win/trade (%)"),
                     ("win_rate_pct", "win rate (%)"),
                     ("max_drawdown_pct", "MAX DRAWDOWN (%)"),
                     ("total_return_pct", "total return, 10yr (%)"),
                     ("final_equity", "final equity ($)")]:
        row = "".join(f"{(f'{arms[name][k]:,.2f}' if arms[name][k] is not None else 'n/a'):>20s}"
                      for name in arms)
        print(f"{label:28s}{row}")

    for name, r in arms.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("%", "pct")
        r["equity_curve"].to_csv((HERE.parent / "results" / f"equity_curve_2yr_{safe}.csv"))
    print(f"\nequity curves saved.")


if __name__ == "__main__":
    main()
