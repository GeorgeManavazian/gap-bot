"""18 backtests: 3 entry styles x 6 bucket scopes (5 isolated gap-size
buckets + 1 "all buckets combined"), each a real $100k account over the
last 2 years, same 20-slot equal-weight sizing and per-bucket data-derived
stop used throughout this study. Every trade logged to CSV.

ENTRY STYLES
  naive        buy at Open on the gap day itself (no confirmation).
  wick         wait for a LATER day where High re-touches the gap-day's own
               open, buy there, at that price.
  close_confirm  wait for a LATER day that CLOSES back above the gap-day's
               open (real reversal proof, not a wick), buy at the following
               day's open.

All three share: down gaps only (>=1%), target = prior close, per-bucket
stop from the earlier study (8/11/15/24/35%), 63-trading-day timeout on
BOTH the wait-to-enter clock and the wait-to-fill clock, 20 equal-weight
slots off a $100k account, ticker-alphabetical tie-break when more signals
than slots.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
LOGS_DIR = HERE.parent / "results" / "trade_logs_18"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 63
CAPITAL = 100_000.0
MAX_SLOTS = 20
WINDOW_YEARS = 2

BUCKETS = [("1-2%", 1.0, 2.0, 8.0), ("2-3%", 2.0, 3.0, 11.0),
           ("3-5%", 3.0, 5.0, 15.0), ("5-10%", 5.0, 10.0, 24.0),
           ("10%+", 10.0, np.inf, 35.0)]
ENTRY_STYLES = ["naive", "wick", "close_confirm"]


def stop_for_abs_gap(abs_gap):
    for name, lo, hi, stop in BUCKETS:
        if lo <= abs_gap < hi:
            return stop
    return None


def bucket_name_for(abs_gap):
    for name, lo, hi, stop in BUCKETS:
        if lo <= abs_gap < hi:
            return name
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


def bucket_matches(abs_gap, scope):
    if scope == "all":
        return 1.0 <= abs_gap
    lo, hi = scope
    return lo <= abs_gap < hi


def run_sim(tickers_data, calendar, entry_style, scope_lo_hi):
    """scope_lo_hi: (lo, hi) tuple for one bucket, or the string 'all'."""
    cash = CAPITAL
    open_positions = {}   # ticker -> position dict
    pending = {}          # ticker -> watch dict (wick/close_confirm only)
    equity_curve = []
    trades = []           # full log rows

    for day_idx, day in enumerate(calendar):
        # 1) exits on open positions
        for tk in list(open_positions):
            pos = open_positions[tk]
            row = tickers_data[tk].get(day)
            if row is None:
                continue
            o, h, l, c = row
            days_held = day_idx - pos["entry_day_idx"]
            exit_price, reason = None, None
            if l <= pos["stop_price"]:
                exit_price, reason = pos["stop_price"], "stop"
            elif h >= pos["tp_price"]:
                exit_price, reason = pos["tp_price"], "tp_gap_filled"
            elif days_held >= HORIZON:
                exit_price, reason = c, "time_exit"
            if exit_price is not None:
                pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                pnl_dollar = pos["shares"] * (exit_price - pos["entry"])
                cash += pos["shares"] * exit_price
                trades.append({
                    "ticker": tk, "gap_date": pos["gap_date"],
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "bucket": pos["bucket"], "gap_pct": pos["gap_pct"],
                    "entry_price": round(pos["entry"], 4),
                    "exit_price": round(exit_price, 4),
                    "exit_reason": reason,
                    "contracts_shares": round(pos["shares"], 4),
                    "position_dollars": round(pos["position_dollars"], 2),
                    "pnl_pct": round(pnl_pct, 3),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "days_held": days_held,
                })
                del open_positions[tk]

        # 2) update pending watches / find new gaps -> today's fill candidates
        fill_candidates = []  # (ticker, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date)

        if entry_style == "naive":
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
                abs_gap = -gap_pct
                if gap_pct >= -1.0 or not bucket_matches(abs_gap, scope_lo_hi):
                    continue
                stop_w = stop_for_abs_gap(abs_gap)
                fill_candidates.append((tk, o, pr, bucket_name_for(abs_gap),
                                        stop_w, gap_pct, day))

        else:  # wick or close_confirm: state machine over `pending`
            # 2a) register brand-new gaps into pending
            for tk in sorted(tickers_data):
                if tk in open_positions or tk in pending:
                    continue
                df = tickers_data[tk]
                tr = df.get(day)
                pr = df.get_prior_close(day)
                if tr is None or pr is None or pr <= 0:
                    continue
                o, h, l, c = tr
                gap_pct = (o - pr) / pr * 100
                abs_gap = -gap_pct
                if gap_pct >= -1.0 or not bucket_matches(abs_gap, scope_lo_hi):
                    continue
                pending[tk] = {"gap_open": o, "prior_close": pr,
                              "start_day_idx": day_idx, "gap_pct": gap_pct,
                              "gap_date": day, "confirmed_day_idx": None}

            # 2b) advance pending watches (skip the gap day itself)
            for tk in list(pending):
                w = pending[tk]
                if day_idx <= w["start_day_idx"]:
                    continue
                if day_idx - w["start_day_idx"] > HORIZON:
                    del pending[tk]  # never triggered in time
                    continue
                if tk in open_positions:
                    del pending[tk]
                    continue
                row = tickers_data[tk].get(day)
                if row is None:
                    continue
                o, h, l, c = row

                if entry_style == "wick":
                    if h >= w["gap_open"]:
                        abs_gap = -w["gap_pct"]
                        fill_candidates.append((tk, w["gap_open"], w["prior_close"],
                                                bucket_name_for(abs_gap),
                                                stop_for_abs_gap(abs_gap),
                                                w["gap_pct"], w["gap_date"]))
                        del pending[tk]

                elif entry_style == "close_confirm":
                    if w["confirmed_day_idx"] is None:
                        if c >= w["gap_open"]:
                            w["confirmed_day_idx"] = day_idx
                    elif day_idx == w["confirmed_day_idx"] + 1:
                        abs_gap = -w["gap_pct"]
                        fill_candidates.append((tk, o, w["prior_close"],
                                                bucket_name_for(abs_gap),
                                                stop_for_abs_gap(abs_gap),
                                                w["gap_pct"], w["gap_date"]))
                        del pending[tk]

        # 3) fill candidates into open slots, ticker-alpha order, equal-weight sizing
        fill_candidates.sort(key=lambda x: x[0])
        free = MAX_SLOTS - len(open_positions)
        for tk, entry_price, tp_price, bucket, stop_w, gap_pct, gap_date in fill_candidates[:max(free, 0)]:
            mtm = sum(
                (tickers_data[p].get(day) or (0, 0, 0, pp["entry"]))[3] * pp["shares"]
                for p, pp in open_positions.items()
            )
            equity_now = cash + mtm
            size_dollars = equity_now / MAX_SLOTS
            if size_dollars > cash:
                continue
            shares = size_dollars / entry_price
            cash -= size_dollars
            open_positions[tk] = {
                "entry": entry_price, "stop_price": entry_price * (1 - stop_w / 100),
                "tp_price": tp_price, "entry_day_idx": day_idx,
                "shares": shares, "bucket": bucket, "gap_pct": round(gap_pct, 3),
                "gap_date": gap_date, "entry_date": day,
                "position_dollars": size_dollars,
            }

        # 4) mark equity
        mtm = sum(
            (tickers_data[tk].get(day) or (0, 0, 0, pos["entry"]))[3] * pos["shares"]
            for tk, pos in open_positions.items()
        )
        equity_curve.append(cash + mtm)

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    tdf = pd.DataFrame(trades)
    return {
        "n_trades": len(tdf),
        "win_rate_pct": 100 * (tdf["pnl_pct"] > 0).mean() if len(tdf) else None,
        "avg_win_pct": tdf["pnl_pct"].mean() if len(tdf) else None,
        "total_pnl_dollar": tdf["pnl_dollar"].sum() if len(tdf) else 0.0,
        "max_drawdown_pct": dd.min() * 100,
        "total_return_pct": (eq.iloc[-1] / CAPITAL - 1) * 100,
        "final_equity": eq.iloc[-1],
        "trades_df": tdf,
        "equity_curve": eq,
    }


def main():
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
    cutoff = calendar[-1] - pd.DateOffset(years=WINDOW_YEARS)
    calendar = calendar[calendar >= cutoff]
    print(f"{len(tickers_data)} tickers, {len(calendar)} trading days, "
          f"{calendar[0].date()} -> {calendar[-1].date()}")

    scopes = [(name, (lo, hi)) for name, lo, hi, stop in BUCKETS] + [("all", "all")]
    summary_rows = []
    for style in ENTRY_STYLES:
        for scope_name, scope in scopes:
            print(f"\nrunning {style} / {scope_name} ...")
            r = run_sim(tickers_data, calendar, style, scope)
            fname = f"{style}_{scope_name.replace('%','pct').replace('-','_').replace('+','plus')}"
            r["trades_df"].to_csv(LOGS_DIR / f"{fname}.csv", index=False)
            r["equity_curve"].to_csv(LOGS_DIR / f"{fname}_equity.csv")
            summary_rows.append({
                "entry_style": style, "bucket_scope": scope_name,
                "n_trades": r["n_trades"],
                "win_rate_%": round(r["win_rate_pct"], 1) if r["win_rate_pct"] is not None else None,
                "avg_win_%": round(r["avg_win_pct"], 3) if r["avg_win_pct"] is not None else None,
                "max_drawdown_%": round(r["max_drawdown_pct"], 2),
                "total_return_%": round(r["total_return_pct"], 2),
                "final_equity_$": round(r["final_equity"], 2),
                "total_pnl_$": round(r["total_pnl_dollar"], 2),
            })
            print(f"  n={r['n_trades']}  win%={r['win_rate_pct']}  "
                  f"return={r['total_return_pct']:.2f}%  "
                  f"maxDD={r['max_drawdown_pct']:.2f}%  "
                  f"final=${r['final_equity']:,.2f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv((HERE.parent / "results" / "full_18_summary.csv"), index=False)
    print("\n" + "=" * 130)
    print("ALL 18 BACKTESTS -- SUMMARY")
    print("=" * 130)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
