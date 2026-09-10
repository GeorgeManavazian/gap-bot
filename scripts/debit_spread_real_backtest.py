"""Real debit-call-spread backtest over the gap-bot's 508-trade 2yr anchor.

Long call ATM at entry_price, short call at width_frac*(target-entry_price)
above entry (target = entry_price/(1+gap_pct/100), the gap-fill/prior-close
level -- same formula as scripts/options_structure_overlay.py's first-order
pass). Uses REAL daily chain quotes from ../etf-bot/data/options/chains and
../etf-bot/data/options/open_interest -- not a static Black-Scholes guess.

Per (ticker, entry_date, target_dte): pick the real listed expiry nearest
target_dte from that day's chain (band search, discrete Friday/monthly
expiries only). Long strike = closest listed CALL strike to entry_price. Short
strike swept via width_frac against the SAME day's listed strikes. Liquidity
gate (both legs, entry day): open_interest>=10, volume>=4 -- the wheel bot's
live floors, not new numbers.

Daily mark = long_mid - short_mid using that day's real bid/ask when the
(expiration,strike,right) row exists; BS fallback (calendar/365, r=4%, own IV
carried forward from the contract's last real vendor_iv) only when a specific
day's row is missing -- rate is reported, not assumed away.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GAP = Path(__file__).resolve().parent.parent
# Option chains live in a sibling etf-bot checkout by default; override with ETF_BOT_DIR.
ETF = Path(os.environ.get("ETF_BOT_DIR", GAP.parent / "etf-bot"))
CHAINS = ETF / "data/options/chains"
OI = ETF / "data/options/open_interest"
ANCHOR = GAP / "results/confirm2yr_heldout_stops_exclude_1-2pct_friction.csv"

R = 0.04
MIN_OI = 10.0
MIN_VOL = 4.0
CAPITAL = 100_000.0
MAX_SLOTS = 20
DTE_GRID = [7, 14, 21, 30, 45, 60]
WIDTH_GRID = [0.5, 0.75, 1.0, 1.25]

_month_cache: dict[tuple[str, str], pd.DataFrame] = {}
_oi_month_cache: dict[tuple[str, str], pd.DataFrame] = {}


def _months_between(d0, d1):
    return pd.period_range(pd.Timestamp(d0).to_period("M"),
                            pd.Timestamp(d1).to_period("M"), freq="M")


def load_chain_months(ticker: str, start, end) -> pd.DataFrame | None:
    frames = []
    for p in _months_between(start, end):
        key = (ticker, str(p))
        if key not in _month_cache:
            fp = CHAINS / ticker / f"{p}.parquet"
            if fp.exists():
                d = pd.read_parquet(fp, columns=[
                    "symbol", "expiration", "strike", "right", "date",
                    "bid", "ask", "volume", "underlying_price", "vendor_iv"])
                d = d[d["right"] == "CALL"]
                _month_cache[key] = d
            else:
                _month_cache[key] = None
        if _month_cache[key] is not None:
            frames.append(_month_cache[key])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_oi_months(ticker: str, start, end) -> pd.DataFrame | None:
    frames = []
    for p in _months_between(start, end):
        key = (ticker, str(p))
        if key not in _oi_month_cache:
            fp = OI / ticker / f"{p}.parquet"
            if fp.exists():
                d = pd.read_parquet(fp)
                d = d[d["right"] == "CALL"]
                _oi_month_cache[key] = d
            else:
                _oi_month_cache[key] = None
        if _oi_month_cache[key] is not None:
            frames.append(_oi_month_cache[key])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sigma, r=R, q=0.0):
    if T <= 1e-9:
        return max(0.0, S - K)
    sigma = max(sigma, 1e-4)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


class TickerBook:
    """Per-ticker indexed view: fast (date, expiration, strike) lookups."""

    def __init__(self, ticker, chain_df, oi_df):
        self.ticker = ticker
        self.chain = chain_df
        self.oi = oi_df
        self.by_date = {} if chain_df is None else {
            d: g for d, g in chain_df.groupby("date")
        }
        self.row_idx = {} if chain_df is None else {
            (r.date, r.expiration, r.strike): r
            for r in chain_df.itertuples(index=False)
        }
        self.oi_idx = {} if oi_df is None else {
            (r.date, r.expiration, r.strike): r.open_interest_asof_prior_close
            for r in oi_df.itertuples(index=False)
        }
        # last-known vendor_iv per (expiration, strike), forward-fillable
        self._iv_carry = {}
        self._last_S = None

    def day_calls(self, date_str):
        return self.by_date.get(date_str)

    def row(self, date_str, expiration_str, strike):
        return self.row_idx.get((date_str, expiration_str, strike))

    def oi_at(self, date_str, expiration_str, strike):
        return self.oi_idx.get((date_str, expiration_str, strike))

    def real_S(self, date_str):
        """Real (unadjusted) underlying price ThetaData actually quoted that
        day, carried forward when the day is missing. NEVER the anchor CSV's
        entry_price/exit_price -- those are split/dividend-adjusted (yfinance
        daily_bars), while the option chain's strikes and underlying_price are
        contemporaneous unadjusted prices. A split between a trade's entry and
        today makes the two series differ by the split ratio (verified: APH
        2024-09-05, CSV entry_price=29.03 vs chain underlying_price=61.86 --
        Amphenol's 2025-03 2-for-1 split, backward-adjusted in the CSV's price
        history). Using the CSV price to pick strikes on a post-split name
        would select a strike at the wrong multiple entirely."""
        day = self.day_calls(date_str)
        if day is not None and len(day):
            u = day["underlying_price"].dropna()
            if len(u):
                self._last_S = float(u.iloc[0])
        return self._last_S

    def mid_or_fallback(self, date_str, expiration_str, strike, S_fallback, T_years):
        row = self.row(date_str, expiration_str, strike)
        if row is not None and row.bid == row.bid and row.ask == row.ask and (row.bid > 0 or row.ask > 0):
            iv = row.vendor_iv
            if iv == iv and iv > 0:
                self._iv_carry[(expiration_str, strike)] = iv
            return 0.5 * (row.bid + row.ask), False
        iv = self._iv_carry.get((expiration_str, strike))
        if iv is None or iv != iv or iv <= 0:
            iv = 0.30  # last resort: no real quote ever seen for this leg
        val = bs_call(S_fallback, strike, max(T_years, 0.0), iv)
        return val, True


def get_book(ticker, start, end, _books={}):
    key = ticker
    if key not in _books:
        chain_df = load_chain_months(ticker, start, end)
        oi_df = load_oi_months(ticker, start, end)
        _books[key] = TickerBook(ticker, chain_df, oi_df)
    return _books[key]


def pick_expiry_and_long_strike(book, entry_date_str, entry_price, target_dte):
    day = book.day_calls(entry_date_str)
    if day is None or not len(day):
        return None
    day = day.copy()
    day["exp_dt"] = pd.to_datetime(day["expiration"])
    dte = (day["exp_dt"] - pd.Timestamp(entry_date_str)).dt.days
    day = day.assign(dte=dte)
    lo, hi = max(1, target_dte - 3), target_dte + 5
    band = day[(day.dte >= lo) & (day.dte <= hi)]
    if not len(band):
        return None
    chosen_dte = band.dte.iloc[(band.dte - target_dte).abs().argmin()]
    exp_candidates = band[band.dte == chosen_dte]
    expiration = exp_candidates["expiration"].iloc[0]
    strikes = exp_candidates["strike"].unique()
    long_strike = strikes[np.argmin(np.abs(strikes - entry_price))]
    return expiration, float(long_strike), strikes, int(chosen_dte)


def pick_short_strike(strikes, long_strike, entry_price, target_price, width_frac):
    """Nearest listed strike to the width target, constrained to sit ABOVE the
    long strike (a spread with short<=long is not a debit call spread). When
    the coarse strike grid puts every candidate at or below the long strike
    (common for narrow width_frac on wide-strike-spacing names), floors to the
    next listed strike above -- flagged via the returned `floored` bool rather
    than silently produced as if it were the requested width."""
    target_short = entry_price + width_frac * (target_price - entry_price)
    strikes = np.asarray(sorted(set(strikes)))
    above = strikes[strikes > long_strike]
    if not len(above):
        return None, False
    nearest = strikes[np.argmin(np.abs(strikes - target_short))]
    if nearest > long_strike:
        return float(nearest), False
    return float(above[0]), True


def leg_liquidity(book, date_str, expiration, strike):
    """Liquidity facts for one leg. HARD gate: volume>=4 (the wheel bot's live
    floor, frozen.toml 2026-08-17) -- chain store volume is populated across
    the full DTE range it was pulled at. OI is reported, never gated: the OI
    store's historical months were pulled at the CLI's old default max_dte=30
    (verified on APH/2024-09 -- max dte present is 30, not the 60 the module
    docstring claims), so an OI join on a 45/60 DTE contract chosen from the
    chain store returns None most of the time for reasons that are a store
    vintage gap, not real illiquidity. Hard-refusing on missing OI would gut
    the sample instead of measuring the structure."""
    row = book.row(date_str, expiration, strike)
    if row is None:
        return None
    oi = book.oi_at(date_str, expiration, strike)
    bid, ask = float(row.bid), float(row.ask)
    mid = 0.5 * (bid + ask)
    vol = float(row.volume) if row.volume == row.volume else np.nan
    return {
        "row_present": True,
        "volume": vol,
        "volume_ok": vol == vol and vol >= MIN_VOL,
        "oi": float(oi) if oi is not None and oi == oi else np.nan,
        "oi_known": oi is not None and oi == oi,
        "spread_pct": (ask - bid) / mid * 100 if mid > 0 else np.nan,
        "zero_volume": (vol == 0),
    }


def next_trading_days(all_dates_sorted, start, end):
    return [d for d in all_dates_sorted if start <= d <= end]


def run_trade(book, row, target_dte, width_frac, calendar_days_by_ticker):
    entry_date = row.entry_date
    exit_date = row.exit_date
    entry_str = str(entry_date.date())

    S_entry = book.real_S(entry_str)
    if S_entry is None:
        return {"status": "no_underlying_price"}
    # gap_pct is scale-invariant; anchor it to the REAL (unadjusted) entry
    # price, not the CSV's split-adjusted entry_price -- see TickerBook.real_S.
    target_price = S_entry / (1 + row.gap_pct / 100)

    picked = pick_expiry_and_long_strike(book, entry_str, S_entry, target_dte)
    if picked is None:
        return {"status": "no_expiry"}
    expiration, long_strike, strikes, actual_dte = picked
    short_strike, width_floored = pick_short_strike(
        strikes, long_strike, S_entry, target_price, width_frac)
    if short_strike is None:
        return {"status": "no_strike_above_long"}

    long_liq = leg_liquidity(book, entry_str, expiration, long_strike)
    short_liq = leg_liquidity(book, entry_str, expiration, short_strike)
    if long_liq is None:
        return {"status": "illiquid_long_row_missing"}
    if short_liq is None:
        return {"status": "illiquid_short_row_missing"}
    if not (long_liq["volume_ok"] and short_liq["volume_ok"]):
        return {"status": "illiquid_low_volume",
                "long_volume": long_liq["volume"], "short_volume": short_liq["volume"]}

    long_row = book.row(entry_str, expiration, long_strike)
    short_row = book.row(entry_str, expiration, short_strike)
    entry_cost = float(long_row.ask - short_row.bid)
    if entry_cost <= 0:
        return {"status": "bad_entry_cost"}

    exp_dt = pd.Timestamp(expiration)
    effective_exit = min(exit_date, exp_dt)
    expiry_forced = exp_dt < exit_date

    dates = calendar_days_by_ticker
    hold_days = next_trading_days(dates, entry_date, effective_exit)
    if not hold_days:
        hold_days = [entry_date]

    daily_marks = []
    fallback_flags = []
    itm_short_flags = []
    S_last = S_entry
    for d in hold_days:
        d_str = str(d.date())
        S = book.real_S(d_str)
        if S is None:
            S = S_last  # no chain snapshot that day anywhere on this ticker
        S_last = S
        T_years = max((exp_dt - d).days, 0) / 365.0
        long_val, lf = book.mid_or_fallback(d_str, expiration, long_strike, S, T_years)
        short_val, sf = book.mid_or_fallback(d_str, expiration, short_strike, S, T_years)
        daily_marks.append(long_val - short_val)
        fallback_flags.append(lf or sf)
        itm_short_flags.append(S > short_strike)

    if effective_exit < exp_dt:
        exit_value = daily_marks[-1]
    else:
        S_exit = S_last  # real S as of the last held trading day (== effective_exit)
        exit_value = max(0.0, S_exit - long_strike) - max(0.0, S_exit - short_strike)

    pnl = exit_value - entry_cost
    width_dollars = short_strike - long_strike

    return {
        "status": "ok",
        "expiration": expiration, "actual_dte": actual_dte,
        "width_floored": width_floored,
        "long_strike": long_strike, "short_strike": short_strike,
        "long_volume": long_liq["volume"], "short_volume": short_liq["volume"],
        "long_oi": long_liq["oi"], "short_oi": short_liq["oi"],
        "oi_known_both_legs": long_liq["oi_known"] and short_liq["oi_known"],
        "long_spread_pct": long_liq["spread_pct"], "short_spread_pct": short_liq["spread_pct"],
        "zero_volume_either_leg": long_liq["zero_volume"] or short_liq["zero_volume"],
        "width_dollars": width_dollars,
        "entry_cost": entry_cost, "exit_value": exit_value, "pnl": pnl,
        "roc_pct": 100.0 * pnl / entry_cost,
        "max_value_reached": max(daily_marks + [exit_value]),
        "bs_fallback_any": any(fallback_flags),
        "itm_short_any": any(itm_short_flags),
        "expiry_forced_exit": expiry_forced,
        "effective_exit": effective_exit,
        "hold_days_used": len(hold_days),
        "daily_marks": daily_marks,
        "hold_dates": [d for d in hold_days],
    }


def main():
    df = pd.read_csv(ANCHOR, parse_dates=["gap_date", "entry_date", "exit_date"])
    df["target"] = df.entry_price / (1 + df.gap_pct / 100)

    tickers = sorted(df.ticker.unique())
    print(f"[{pd.Timestamp.now()}] {len(df)} trades, {len(tickers)} tickers, "
          f"DTE grid {DTE_GRID}, width grid {WIDTH_GRID}", flush=True)

    all_results = []
    for i, ticker in enumerate(tickers):
        sub = df[df.ticker == ticker]
        start = sub.entry_date.min() - pd.Timedelta(days=5)
        end = sub.exit_date.max() + pd.Timedelta(days=70)
        book = get_book(ticker, start, end)
        if book.chain is None:
            for _, row in sub.iterrows():
                for dte in DTE_GRID:
                    for wf in WIDTH_GRID:
                        all_results.append({
                            "ticker": ticker, "bucket": row.bucket, "trade_idx": row.name,
                            "target_dte": dte, "width_frac": wf, "status": "no_chain_data",
                        })
            continue
        calendar_days = sorted(book.by_date.keys())
        calendar_days = pd.to_datetime(pd.Series(calendar_days)).tolist()
        for _, row in sub.iterrows():
            for dte in DTE_GRID:
                for wf in WIDTH_GRID:
                    res = run_trade(book, row, dte, wf, calendar_days)
                    res.update({
                        "ticker": ticker, "bucket": row.bucket, "trade_idx": row.name,
                        "target_dte": dte, "width_frac": wf,
                        "gap_pct": row.gap_pct, "entry_date": row.entry_date,
                        "exit_date": row.exit_date, "exit_reason": row.exit_reason,
                        "stock_entry_price": row.entry_price,
                    })
                    all_results.append(res)
        if (i + 1) % 25 == 0:
            print(f"[{pd.Timestamp.now()}] {i+1}/{len(tickers)} tickers done", flush=True)

    res_df = pd.DataFrame(all_results)
    res_df.to_pickle(GAP / "results/_debit_spread_real_raw.pkl")
    print(f"[{pd.Timestamp.now()}] raw results: {len(res_df)} rows, "
          f"status counts:\n{res_df.status.value_counts()}", flush=True)


if __name__ == "__main__":
    main()
