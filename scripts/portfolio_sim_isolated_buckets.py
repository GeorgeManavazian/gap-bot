"""Isolate each gap-size bucket into its OWN portfolio -- only 1-2% gaps,
only 2-3%, etc, never mixed. Same $100k account, same 20-slot equal-weight
sizing, same per-bucket data-derived stop, same 63-day time exit. The only
variable is: does trading ONE consistent flavor of signal behave more like
a fixed-edge game than the mixed pool did?
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
HORIZON = 63
CAPITAL = 100_000.0
MAX_SLOTS = 20

BUCKETS = [("1-2%", 1.0, 2.0, 8.0), ("2-3%", 2.0, 3.0, 11.0),
           ("3-5%", 3.0, 5.0, 15.0), ("5-10%", 5.0, 10.0, 24.0),
           ("10%+", 10.0, np.inf, 35.0)]


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


class TickerView:
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


def run_sim(tickers_data, calendar, lo, hi, stop_w, use_stop=True):
    cash = CAPITAL
    open_positions = {}
    equity_curve = []
    closed_trades = []

    for day_idx, day in enumerate(calendar):
        for tk in list(open_positions):
            pos = open_positions[tk]
            row = tickers_data[tk].get(day)
            if row is None:
                continue
            o, h, l, c = row
            days_held = day_idx - pos["entry_day_idx"]
            exit_price = None
            if use_stop and l <= pos["stop_price"]:
                exit_price = pos["stop_price"]
            elif h >= pos["tp_price"]:
                exit_price = pos["tp_price"]
            elif days_held >= HORIZON:
                exit_price = c
            if exit_price is not None:
                pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                cash += pos["shares"] * exit_price
                closed_trades.append(pnl_pct)
                del open_positions[tk]

        free = MAX_SLOTS - len(open_positions)
        if free > 0:
            candidates = []
            for tk in sorted(tickers_data):
                if tk in open_positions:
                    continue
                df = tickers_data[tk]
                tr = df.get(day)
                pr = df.get_prior_close(day)
                if tr is None or pr is None or pr <= 0:
                    continue
                o, h, l, c = tr
                gap_pct = (o - pr) / pr * 100
                abs_gap = -gap_pct  # down gaps only, positive magnitude
                if not (lo <= abs_gap < hi) or gap_pct >= -1.0:
                    continue
                candidates.append((tk, o, pr))
            for tk, o, pr in candidates[:free]:
                mtm = sum(
                    (tickers_data[p].get(day) or (0, 0, 0, pp["entry"]))[3] * pp["shares"]
                    for p, pp in open_positions.items()
                )
                equity_now = cash + mtm
                shares = (equity_now / MAX_SLOTS) / o
                cash -= shares * o
                open_positions[tk] = {
                    "entry": o, "stop_price": o * (1 - stop_w / 100),
                    "tp_price": pr, "entry_day_idx": day_idx, "shares": shares,
                }

        mtm = sum(
            (tickers_data[tk].get(day) or (0, 0, 0, pos["entry"]))[3] * pos["shares"]
            for tk, pos in open_positions.items()
        )
        equity_curve.append(cash + mtm)

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    trades = pd.Series(closed_trades)
    return {
        "n_trades": len(trades),
        "avg_win_pct": trades.mean() if len(trades) else None,
        "std_pct": trades.std() if len(trades) else None,
        "win_rate_pct": 100 * (trades > 0).mean() if len(trades) else None,
        "max_drawdown_pct": dd.min() * 100,
        "total_return_pct": (eq.iloc[-1] / CAPITAL - 1) * 100,
        "final_equity": eq.iloc[-1],
        "sharpe_like": (trades.mean() / trades.std()) if len(trades) and trades.std() else None,
        "equity_curve": eq,
    }


def main(window_years=None, use_stop=True):
    files = sorted(BARS_DIR.glob("*.parquet"))
    tickers_data = {}
    longest_dates = None
    for p in files:
        d = load_ticker(p)
        if d is None:
            continue
        tickers_data[p.stem] = TickerView(d)
        if longest_dates is None or len(d) > len(longest_dates):
            longest_dates = d["Date"]
    calendar = pd.DatetimeIndex(sorted(longest_dates))
    if window_years is not None:
        cutoff = calendar[-1] - pd.DateOffset(years=window_years)
        calendar = calendar[calendar >= cutoff]
    label = f"{window_years}yr" if window_years else "10yr (full)"
    print(f"\n{'='*110}\nISOLATED BUCKETS -- {label} window, stop={'ON' if use_stop else 'OFF'}, "
          f"{calendar[0].date()} -> {calendar[-1].date()}\n{'='*110}")

    rows = []
    for name, lo, hi, stop_w in BUCKETS:
        r = run_sim(tickers_data, calendar, lo, hi, stop_w, use_stop=use_stop)
        rows.append({
            "bucket": name, "stop_%": stop_w, "n_trades": r["n_trades"],
            "avg_win_%": round(r["avg_win_pct"], 3) if r["avg_win_pct"] is not None else None,
            "std_%": round(r["std_pct"], 3) if r["std_pct"] is not None else None,
            "win_rate_%": round(r["win_rate_pct"], 1) if r["win_rate_pct"] is not None else None,
            "avg/std (consistency)": round(r["sharpe_like"], 3) if r["sharpe_like"] is not None else None,
            "max_drawdown_%": round(r["max_drawdown_pct"], 2),
            "total_return_%": round(r["total_return_pct"], 2),
            "final_equity": round(r["final_equity"], 2),
        })
        r["equity_curve"].to_csv((HERE.parent / "results" / f"equity_curve_isolated_{name.replace('%','pct').replace('-','_').replace('+','plus')}_{label.replace(' ','')}.csv"))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv((HERE.parent / "results" / f"isolated_buckets_{label.replace(' ','').replace('(','').replace(')','')}.csv"), index=False)
    return t


if __name__ == "__main__":
    main(window_years=None, use_stop=True)
    main(window_years=2, use_stop=True)
