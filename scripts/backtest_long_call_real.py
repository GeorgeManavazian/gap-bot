#!/usr/bin/env python
"""Long-call replacement for the gap bot's stock trades, priced off REAL daily chains.

For every trade in the 2yr anchor log, replace the shares with one long call
chosen on the entry day from the real EOD chain snapshot (etf-bot's ThetaData
store), mark it every day at mid off the real snapshots, and sell it at the bid
on the stock trade's real exit day. If the option expires first, it settles at
intrinsic on expiry and the trade ends there (no roll). Same signal, same
exit rules, different instrument.

Three arms on the IDENTICAL trade subset (trades with chain data):
  anchor      the stock anchor's own fills and P&L, restricted to the subset
  stock_eod   shares bought/sold at the chain's EOD underlying price on the same
              days -- the clean control, since the option can only fill at EOD
  call_<dte>_<delta>   the long call, notional-matched (contracts x 100 = shares)

Sweep: DTE target in {14, 21, 30, 45, 60}; delta target in {0.3, 0.5, 0.7}.
The store was pulled with max_dte=60, so nothing longer exists. Store ends 2026-08-07 (ThetaData lapsed); trades entering after
are excluded and trades still open then are truncated at the last snapshot in
EVERY arm alike.

Costs: buy at ask, sell at bid (real spread), $0.65/contract/side commission.
Fractional contracts, matching the anchor's fractional shares -- integer
lumpiness at a $5k slot is reported separately, not modelled.

Outputs (results/):
  long_call_real_scorecard.csv      per bucket x config: n, win rate, avg pnl,
                                    total return, maxDD, ret/maxDD
  long_call_real_trades.csv         every trade x config
  long_call_real_equity_<cfg>.csv   daily equity curve per config
  long_call_real_gaps.json          what was excluded and why
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
BARS = os.path.join(ROOT, "data", "daily_bars")
CHAINS = os.path.join(ROOT, "..", "etf-bot", "data", "options", "chains")
OI = os.path.join(ROOT, "..", "etf-bot", "data", "options", "open_interest")
ANCHOR = os.path.join(RESULTS, "confirm2yr_heldout_stops_exclude_1-2pct_friction.csv")

STORE_END = pd.Timestamp("2026-08-07")
START_EQUITY = 100_000.0
COMMISSION = 0.65          # per contract per side (Schwab)
FILL = os.environ.get("FILL", "ba")   # "ba" = buy ask / sell bid (default); "mid" = frictionless sensitivity
SUFFIX = "" if FILL == "ba" else f"_{FILL}"
DTES = [14, 21, 30, 45, 60]
DELTAS = [0.3, 0.5, 0.7]
CHAIN_COLS = ["expiration", "strike", "right", "bid", "ask", "underlying_price",
              "vendor_iv", "vendor_delta", "date", "volume"]

_chain_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
_oi_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
_bars_cache: dict[str, pd.Series | None] = {}


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")]


def load_chain(ticker: str, month: str) -> pd.DataFrame | None:
    key = (ticker, month)
    if key not in _chain_cache:
        path = os.path.join(CHAINS, ticker, f"{month}.parquet")
        if not os.path.exists(path):
            _chain_cache[key] = None
        else:
            c = pd.read_parquet(path, columns=CHAIN_COLS)
            c = c[c.right == "CALL"].copy()
            c["date"] = pd.to_datetime(c["date"])
            c["expiration"] = pd.to_datetime(c["expiration"])
            _chain_cache[key] = c
    return _chain_cache[key]


def load_oi(ticker: str, month: str) -> pd.DataFrame | None:
    key = (ticker, month)
    if key not in _oi_cache:
        path = os.path.join(OI, ticker, f"{month}.parquet")
        if not os.path.exists(path):
            _oi_cache[key] = None
        else:
            o = pd.read_parquet(path)
            o = o[o.right == "CALL"].copy()
            o["date"] = pd.to_datetime(o["date"])
            o["expiration"] = pd.to_datetime(o["expiration"])
            _oi_cache[key] = o
    return _oi_cache[key]


def load_bars(ticker: str) -> pd.Series | None:
    if ticker not in _bars_cache:
        path = os.path.join(BARS, f"{ticker}.parquet")
        if not os.path.exists(path):
            _bars_cache[ticker] = None
        else:
            b = pd.read_parquet(path)
            dcol = [c for c in b.columns if c.lower() == "date"]
            if dcol:
                b = b.set_index(dcol[0])
            b.index = pd.to_datetime(b.index)
            if not isinstance(b.index, pd.DatetimeIndex) or b.index.min().year < 2000:
                raise RuntimeError(f"{ticker}: bars index is not a real date index")
            col = [c for c in b.columns if c.lower() == "close"][0]   # unadjusted close (auto_adjust=False)
            _bars_cache[ticker] = b[col].astype(float)
    return _bars_cache[ticker]


def chain_window(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = [c for m in months_between(start, end) if (c := load_chain(ticker, m)) is not None]
    if not parts:
        return pd.DataFrame(columns=CHAIN_COLS)
    c = pd.concat(parts, ignore_index=True)
    return c[(c.date >= start) & (c.date <= end)]


def pick_contract(snap: pd.DataFrame, dte_target: int, delta_target: float):
    """Nearest expiry at or beyond the DTE target, then the call whose vendor
    delta is closest to target among quotable rows (bid > 0). Returns row or None."""
    if snap.empty:
        return None
    dte = (snap.expiration - snap.date).dt.days
    elig = snap[(dte >= dte_target) & (snap.bid > 0) & (snap.ask > 0)]
    if elig.empty:
        # nothing at/after target: fall back to the longest expiry available
        elig = snap[(snap.bid > 0) & (snap.ask > 0)]
        if elig.empty:
            return None
        far = (elig.expiration - elig.date).dt.days.max()
        elig = elig[(elig.expiration - elig.date).dt.days == far]
    else:
        near = (elig.expiration - elig.date).dt.days.min()
        elig = elig[(elig.expiration - elig.date).dt.days == near]
    d = (elig.vendor_delta - delta_target).abs()
    return elig.loc[d.idxmin()]


def simulate_trade(row, dte_target: int, delta_target: float, chain: pd.DataFrame, gaps: dict):
    """Returns dict with per-day marks (dict date->value per contract) and outcome."""
    entry = row.entry_date
    exit_ = min(row.exit_date, STORE_END)
    snap = chain[chain.date == entry]
    if snap.empty:
        gaps["no_entry_snapshot"] += 1
        return None
    c = pick_contract(snap, dte_target, delta_target)
    if c is None:
        gaps["no_quotable_contract"] += 1
        return None
    K, exp_ = c.strike, c.expiration
    S0 = c.underlying_price
    buy = c.ask if FILL == "ba" else (c.bid + c.ask) / 2
    series = chain[(chain.strike == K) & (chain.expiration == exp_) & (chain.date > entry) & (chain.date <= exit_)]
    series = series.sort_values("date")
    marks = {entry: (c.bid + c.ask) / 2}
    last_mark = marks[entry]
    bars = load_bars(row.ticker)
    end_reason = "stock_exit"
    sell = None
    last_date = entry
    for r in series.itertuples():
        if r.date >= exp_:
            break
        m = (r.bid + r.ask) / 2 if r.ask > 0 else max(r.bid, 0.0)
        marks[r.date] = m
        last_mark, last_date = m, r.date
        if r.date == exit_:
            sell = r.bid if FILL == "ba" else m
    if exp_ <= exit_:
        # expired before (or on) the stock's exit: settle at intrinsic on expiry
        end_reason = "expired_before_exit"
        s_exp = None
        on_exp = chain[chain.date == exp_]                 # the chain's own underlying on expiry day
        if not on_exp.empty:
            s_exp = float(on_exp.underlying_price.iloc[0])
        elif bars is not None:
            b = bars[(bars.index <= exp_) & (bars.index >= exp_ - pd.Timedelta(days=4))]
            if len(b):
                s_exp = float(b.iloc[-1])
        if s_exp is None:
            gaps["expiry_underlying_missing"] += 1
            s_exp = series.underlying_price.iloc[-1] if len(series) else S0
        sell = max(s_exp - K, 0.0)
        marks[exp_] = sell
        exit_day = exp_
    else:
        exit_day = exit_
        if sell is None:
            # no snapshot on the exit day: last available mark, flagged
            gaps["exit_day_missing_snapshot"] += 1
            sell = last_mark
            end_reason += "_lastmark"
        if row.exit_date > STORE_END:
            end_reason = "truncated_at_store_end"
    # underlying on the STOCK's exit day, for the stock_eod control (not the option's expiry)
    s_exit = None
    ex = chain[chain.date == exit_]
    if not ex.empty:
        s_exit = float(ex.underlying_price.iloc[0])
    elif bars is not None:
        b = bars[bars.index <= exit_]
        s_exit = float(b.iloc[-1]) if len(b) else None
    return dict(strike=K, expiration=exp_, dte=int((exp_ - entry).days), delta=float(c.vendor_delta),
                iv=float(c.vendor_iv), S0=float(S0), s_exit=s_exit, buy=float(buy), sell=float(sell),
                spread_pct=float((c.ask - c.bid) / ((c.ask + c.bid) / 2)), exit_day=exit_day,
                end_reason=end_reason, marks=marks, volume=int(c.volume) if pd.notna(c.volume) else 0)


def oi_of(ticker: str, date: pd.Timestamp, strike: float, exp_: pd.Timestamp):
    o = load_oi(ticker, date.strftime("%Y-%m"))
    if o is None:
        return np.nan
    m = o[(o.date == date) & (o.strike == strike) & (o.expiration == exp_)]
    return float(m.open_interest_asof_prior_close.iloc[0]) if len(m) else np.nan


def equity_curve(legs: list[dict], calendar: pd.DatetimeIndex) -> pd.Series:
    """legs: dicts with entry, exit_day, cost (cash out at entry), proceeds (cash in
    at exit), marks: {date: value}. Daily equity = cash + sum of open marks."""
    cash = pd.Series(0.0, index=calendar)
    open_val = pd.Series(0.0, index=calendar)
    for L in legs:
        cash[cash.index >= L["entry"]] -= L["cost"]
        cash[cash.index >= L["exit_day"]] += L["proceeds"]
        mk = pd.Series(L["marks"]).sort_index()
        mk = mk.reindex(calendar[(calendar >= L["entry"]) & (calendar < L["exit_day"])], method="ffill")
        open_val = open_val.add(mk.fillna(0.0) * L["units"], fill_value=0.0)
    return START_EQUITY + cash + open_val


def score(eq: pd.Series, pnl: pd.Series) -> dict:
    dd = eq / eq.cummax() - 1
    tr = eq.iloc[-1] / START_EQUITY - 1
    return dict(n_trades=int(len(pnl)), win_rate_pct=round(100 * (pnl > 0).mean(), 1) if len(pnl) else np.nan,
                avg_pnl_pct=round(pnl.mean(), 3) if len(pnl) else np.nan,
                std_pct=round(pnl.std(), 3) if len(pnl) > 1 else np.nan,
                total_return_pct=round(100 * tr, 2), max_drawdown_pct=round(100 * dd.min(), 2),
                ret_per_maxdd=round(tr / abs(dd.min()), 3) if dd.min() < 0 else np.nan,
                final_equity=round(eq.iloc[-1], 2))


def main():
    df = pd.read_csv(ANCHOR, parse_dates=["entry_date", "exit_date", "gap_date"])
    gaps = defaultdict(int)
    gaps["anchor_trades"] = len(df)
    missing_tk = sorted({t for t in df.ticker.unique() if not os.path.isdir(os.path.join(CHAINS, t))})
    gaps["tickers_missing_from_store"] = missing_tk
    gaps["trades_on_missing_tickers"] = int(df.ticker.isin(missing_tk).sum())
    gaps["trades_entering_after_store_end"] = int((df.entry_date > STORE_END).sum())
    gaps["trades_truncated_at_store_end"] = int(((df.entry_date <= STORE_END) & (df.exit_date > STORE_END)).sum())
    sub = df[~df.ticker.isin(missing_tk) & (df.entry_date <= STORE_END)].copy()
    calendar = pd.bdate_range(sub.entry_date.min(), STORE_END)

    # preload chains per trade window
    windows = {}
    for r in sub.itertuples():
        end = min(r.exit_date, STORE_END)
        windows[r.Index] = chain_window(r.ticker, r.entry_date, end + pd.Timedelta(days=7))
    print(f"chains loaded for {len(windows)} trades", file=sys.stderr)

    trade_rows, legs_by_cfg, scorecard = [], defaultdict(list), []

    # ---- arm: anchor subset (actual fills) ----
    anchor_legs, anchor_pnl = [], []
    stock_legs, stock_pnl, stock_meta = [], [], {}
    for r in sub.itertuples():
        exit_day = min(r.exit_date, STORE_END)
        bars = load_bars(r.ticker)
        marks = {}
        if bars is not None:
            b = bars[(bars.index >= r.entry_date) & (bars.index <= exit_day)]
            marks = {d: v / r.entry_price * r.entry_price for d, v in b.items()}  # share price path
        # anchor: truncated trades close at the bar close on STORE_END
        if r.exit_date > STORE_END:
            px = marks[max(k for k in marks if k <= exit_day)] if marks else r.entry_price
            pnl_pct = (px / r.entry_price - 1) * 100
        else:
            px, pnl_pct = r.exit_price, r.pnl_pct
        units = r.position_dollars / r.entry_price
        anchor_legs.append(dict(entry=r.entry_date, exit_day=exit_day, cost=r.position_dollars,
                                proceeds=units * px, marks=marks or {r.entry_date: r.entry_price}, units=units))
        anchor_pnl.append((r.Index, r.bucket, pnl_pct))
    eq = equity_curve(anchor_legs, calendar)
    eq.to_csv(os.path.join(RESULTS, f"long_call_real_equity_anchor_subset{SUFFIX}.csv"))
    ap = pd.DataFrame(anchor_pnl, columns=["idx", "bucket", "pnl_pct"]).set_index("idx")
    scorecard.append(dict(config="anchor_subset", bucket="ALL", **score(eq, ap.pnl_pct)))
    for bk, g in ap.groupby("bucket"):
        eqb = equity_curve([anchor_legs[i] for i in range(len(anchor_legs)) if sub.index[i] in g.index], calendar)
        scorecard.append(dict(config="anchor_subset", bucket=bk, **score(eqb, g.pnl_pct)))

    # ---- option arms + stock_eod control (needs the chain's EOD underlying) ----
    stock_eod_done = False
    for dte in DTES:
        for dl in DELTAS:
            cfg = f"call_{dte}d_{dl:.1f}"
            legs, pnl_rows, g = [], [], defaultdict(int)
            se_legs, se_pnl = [], []
            for r in sub.itertuples():
                sim = simulate_trade(r, dte, dl, windows[r.Index], g)
                if sim is None:
                    continue
                # contracts, fractional, sized on the chain's own underlying: the anchor's
                # entry_price is split-adjusted (yfinance) and strikes are not
                units = r.position_dollars / sim["S0"] / 100.0
                cost = units * 100 * sim["buy"] + units * COMMISSION
                proceeds = units * 100 * sim["sell"] - units * COMMISSION
                pnl_dollar = proceeds - cost
                pnl_pct_notional = pnl_dollar / r.position_dollars * 100
                pnl_pct_premium = pnl_dollar / cost * 100 if cost > 0 else np.nan
                legs.append(dict(entry=r.entry_date, exit_day=sim["exit_day"], cost=cost, proceeds=proceeds,
                                 marks={d: v * 100 for d, v in sim["marks"].items()}, units=units))
                pnl_rows.append(dict(idx=r.Index, config=cfg, ticker=r.ticker, bucket=r.bucket, entry_date=r.entry_date,
                                     exit_date=r.exit_date, exit_day=sim["exit_day"], stock_exit_reason=r.exit_reason,
                                     option_end=sim["end_reason"], days_held=r.days_held, strike=sim["strike"],
                                     expiration=sim["expiration"], dte_actual=sim["dte"], delta=sim["delta"], iv=sim["iv"],
                                     S0=sim["S0"], s_exit=sim["s_exit"], buy=sim["buy"], sell=sim["sell"],
                                     premium_pct_notional=sim["buy"] / sim["S0"] * 100, spread_pct=sim["spread_pct"] * 100,
                                     volume=sim["volume"], contracts=units, cost=cost,
                                     pnl_dollar=pnl_dollar, pnl_pct_notional=pnl_pct_notional, pnl_pct_premium=pnl_pct_premium,
                                     stock_pnl_pct=r.pnl_pct))
                if not stock_eod_done and sim["s_exit"] is not None:
                    su = r.position_dollars / sim["S0"]
                    bars = load_bars(r.ticker)
                    mk = {}
                    if bars is not None:
                        b = bars[(bars.index >= r.entry_date) & (bars.index <= sim["exit_day"])]
                        mk = {d: float(v) for d, v in b.items()}
                    se_legs.append(dict(entry=r.entry_date, exit_day=min(r.exit_date, STORE_END), cost=r.position_dollars,
                                        proceeds=su * sim["s_exit"], marks=mk or {r.entry_date: sim["S0"]}, units=su))
                    se_pnl.append(dict(idx=r.Index, bucket=r.bucket, pnl_pct=(sim["s_exit"] / sim["S0"] - 1) * 100))
            tr = pd.DataFrame(pnl_rows)
            trade_rows.append(tr)
            eq = equity_curve(legs, calendar)
            eq.to_csv(os.path.join(RESULTS, f"long_call_real_equity_{cfg}{SUFFIX}.csv"))
            scorecard.append(dict(config=cfg, bucket="ALL", **score(eq, tr.pnl_pct_notional),
                                  avg_pnl_pct_premium=round(tr.pnl_pct_premium.mean(), 1),
                                  avg_premium_pct_notional=round(tr.premium_pct_notional.mean(), 2),
                                  expired_before_exit=int((tr.option_end == "expired_before_exit").sum()),
                                  median_spread_pct=round(tr.spread_pct.median(), 1)))
            for bk, gb in tr.groupby("bucket"):
                idxs = set(gb.idx)
                eqb = equity_curve([L for L, pr in zip(legs, pnl_rows) if pr["idx"] in idxs], calendar)
                scorecard.append(dict(config=cfg, bucket=bk, **score(eqb, gb.pnl_pct_notional),
                                      avg_pnl_pct_premium=round(gb.pnl_pct_premium.mean(), 1),
                                      avg_premium_pct_notional=round(gb.premium_pct_notional.mean(), 2),
                                      expired_before_exit=int((gb.option_end == "expired_before_exit").sum()),
                                      median_spread_pct=round(gb.spread_pct.median(), 1)))
            gaps[f"{cfg}_gaps"] = dict(g)
            print(f"{cfg}: n={len(tr)} avg_pnl_notional={tr.pnl_pct_notional.mean():.2f}% "
                  f"win={100*(tr.pnl_pct_notional>0).mean():.1f}% expired_first={int((tr.option_end=='expired_before_exit').sum())}",
                  file=sys.stderr)
            if not stock_eod_done and se_legs:
                eq = equity_curve(se_legs, calendar)
                eq.to_csv(os.path.join(RESULTS, f"long_call_real_equity_stock_eod{SUFFIX}.csv"))
                sp = pd.DataFrame(se_pnl)
                scorecard.append(dict(config="stock_eod_control", bucket="ALL", **score(eq, sp.pnl_pct)))
                for bk, gb in sp.groupby("bucket"):
                    idxs = set(gb.idx)
                    eqb = equity_curve([L for L, pr in zip(se_legs, se_pnl) if pr["idx"] in idxs], calendar)
                    scorecard.append(dict(config="stock_eod_control", bucket=bk, **score(eqb, gb.pnl_pct)))
                stock_eod_done = True

    trades = pd.concat(trade_rows, ignore_index=True)
    # OI of the chosen contract on entry day (liquidity flag), one config's worth per trade is enough
    ref = trades[trades.config == "call_30d_0.5"]
    trades["oi_entry"] = np.nan
    oi_vals = {i: oi_of(r.ticker, r.entry_date, r.strike, r.expiration) for i, r in ref.set_index("idx").iterrows()}
    trades.loc[trades.config == "call_30d_0.5", "oi_entry"] = trades.loc[trades.config == "call_30d_0.5", "idx"].map(oi_vals)
    trades.to_csv(os.path.join(RESULTS, f"long_call_real_trades{SUFFIX}.csv"), index=False)
    sc = pd.DataFrame(scorecard)
    sc.to_csv(os.path.join(RESULTS, f"long_call_real_scorecard{SUFFIX}.csv"), index=False)
    gaps["scored_subset"] = int(len(sub))
    with open(os.path.join(RESULTS, f"long_call_real_gaps{SUFFIX}.json"), "w") as f:
        json.dump(gaps, f, indent=2, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(sc[sc.bucket == "ALL"].to_string(index=False))
    print(json.dumps({k: v for k, v in gaps.items() if not k.endswith("_gaps")}, indent=1, default=str))


if __name__ == "__main__":
    main()
