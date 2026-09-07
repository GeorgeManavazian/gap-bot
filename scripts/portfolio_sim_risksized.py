"""Risk-based position sizing, as specified: max loss per trade = RISK_PCT of
CURRENT equity (2% of $100k = $2,000 to start; scales with equity as the
account compounds -- flagged below, this is a design choice, not what was
literally said, which was a flat $2,000).

  position_$ = risk_$ / stop_distance_%

This REQUIRES a stop (you can't define "max loss" without a loss point), so
it uses the per-bucket data-derived stop from the earlier test (8/11/15/24/
35% by gap size). Tighter stop -> bigger dollar position for the same fixed
risk; wider stop -> smaller position. That also means the worst-performing
bucket from the earlier expectancy test (the 1-2% tight-stop bucket) gets
the LARGEST position size here -- flagged plainly in the report, not buried.

No artificial slot count. Capacity is now whatever real dollar constraint
falls out: a new position opens only if there's enough uncommitted cash to
cover it. On a cluster day, this can exhaust cash after just a few large
positions -- a genuinely different concentration behavior than the flat
20-slot model, and the point of rerunning it this way.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
HORIZON = 63
CAPITAL = 100_000.0
RISK_PCT = 0.02  # 2% of CURRENT equity risked per trade
WINDOW_YEARS = None  # set per run

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


def run_sim(tickers_data, calendar, use_stop=True):
    cash = CAPITAL
    open_positions = {}
    equity_curve = []
    closed_trades = []
    max_concurrent = 0
    skipped_no_cash = 0

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
                closed_trades.append({"pnl_pct": pnl_pct,
                                      "bucket": pos["bucket"],
                                      "position_dollars": pos["position_dollars"]})
                del open_positions[tk]

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
            if gap_pct >= -1.0:
                continue
            stop_w = stop_for_gap(abs(gap_pct))
            if stop_w is None:
                continue
            mtm = sum(
                (tickers_data[p].get(day) or (0, 0, 0, pp["entry"]))[3] * pp["shares"]
                for p, pp in open_positions.items()
            )
            equity_now = cash + mtm
            risk_dollars = equity_now * RISK_PCT
            position_dollars = risk_dollars / (stop_w / 100)
            if position_dollars > cash:
                skipped_no_cash += 1
                continue
            shares = position_dollars / o
            cash -= position_dollars
            open_positions[tk] = {
                "entry": o, "stop_price": o * (1 - stop_w / 100),
                "tp_price": pr, "entry_day_idx": day_idx, "shares": shares,
                "bucket": stop_w, "position_dollars": position_dollars,
            }
        max_concurrent = max(max_concurrent, len(open_positions))

        mtm = sum(
            (tickers_data[tk].get(day) or (0, 0, 0, pos["entry"]))[3] * pos["shares"]
            for tk, pos in open_positions.items()
        )
        equity_curve.append(cash + mtm)

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    trades_df = pd.DataFrame(closed_trades)
    return {
        "n_trades": len(closed_trades),
        "avg_win_pct": trades_df["pnl_pct"].mean() if len(trades_df) else None,
        "win_rate_pct": 100 * (trades_df["pnl_pct"] > 0).mean() if len(trades_df) else None,
        "max_drawdown_pct": dd.min() * 100,
        "total_return_pct": (eq.iloc[-1] / CAPITAL - 1) * 100,
        "final_equity": eq.iloc[-1],
        "max_concurrent_positions": max_concurrent,
        "avg_position_dollars": trades_df["position_dollars"].mean() if len(trades_df) else None,
        "median_position_dollars": trades_df["position_dollars"].median() if len(trades_df) else None,
        "max_position_dollars": trades_df["position_dollars"].max() if len(trades_df) else None,
        "n_skipped_no_cash": skipped_no_cash,
        "equity_curve": eq,
        "trades_df": trades_df,
    }


def main(window_years=None, label=""):
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
    print(f"\n=== {label or 'full window'}: {calendar[0].date()} -> {calendar[-1].date()} "
          f"({len(calendar)} days) ===")

    r = run_sim(tickers_data, calendar, use_stop=True)
    print(f"trades taken:              {r['n_trades']:,}")
    print(f"avg win/trade:             {r['avg_win_pct']:.2f}%")
    print(f"win rate:                  {r['win_rate_pct']:.1f}%")
    print(f"max drawdown:              {r['max_drawdown_pct']:.2f}%")
    print(f"total return:              {r['total_return_pct']:.2f}%")
    print(f"final equity:              ${r['final_equity']:,.2f}")
    print(f"max concurrent positions:  {r['max_concurrent_positions']}")
    print(f"avg position size:         ${r['avg_position_dollars']:,.2f}")
    print(f"median position size:      ${r['median_position_dollars']:,.2f}")
    print(f"max position size ever:    ${r['max_position_dollars']:,.2f}")
    print(f"entries skipped (no cash): {r['n_skipped_no_cash']:,}")

    print("\nposition size + count by bucket (stop%):")
    td = r["trades_df"]
    for lo, hi, stop in BUCKET_STOPS:
        sub = td[td["bucket"] == stop]
        if len(sub) == 0:
            continue
        print(f"  stop {stop:5.1f}%  n={len(sub):6d}  "
              f"avg pos ${sub['position_dollars'].mean():>10,.0f}  "
              f"avg pnl {sub['pnl_pct'].mean():+.2f}%")
    return r


if __name__ == "__main__":
    r10 = main(window_years=None, label="10-YEAR WINDOW")
    r10["equity_curve"].to_csv((HERE.parent / "results" / "equity_curve_risksized_10yr.csv"))
    r2 = main(window_years=2, label="2-YEAR WINDOW")
    r2["equity_curve"].to_csv((HERE.parent / "results" / "equity_curve_risksized_2yr.csv"))
