"""Schwab live-data client for gap-bot -- same config file, same token, same
account as the wheel bot (one Schwab login serves both; this repo just reads
it). Copied rather than cross-imported from code/etf-bot/scripts/schwab/
schwab_client.py so gap-bot stays a self-contained repo. Data-only, no order
placement, matching the wheel bot's own doctrine."""
from __future__ import annotations
import json
import os
import time

import pandas as pd

CONFIG_PATH = os.path.expanduser("~/.schwab/config.json")
ADV_LOOKBACK = 20
FETCH_LOOKBACK_DAYS = 45  # calendar days; enough trading days for ADV(20) + buffer


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise SystemExit(f"No config at {CONFIG_PATH}. Same one the wheel bot uses.")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["token_path"] = os.path.expanduser(cfg["token_path"])
    return cfg


def get_client():
    from schwab.auth import client_from_token_file
    cfg = load_config()
    if not os.path.exists(cfg["token_path"]):
        raise SystemExit(f"No token at {cfg['token_path']}. Run schwab_login.py first "
                         f"(same login as the wheel bot -- one Schwab account).")
    return client_from_token_file(cfg["token_path"], cfg["app_key"], cfg["app_secret"])


def throttle(fn, *args, retries: int = 2, backoff: float = 1.0, **kwargs):
    r = fn(*args, **kwargs)
    attempts = 0
    while getattr(r, "status_code", None) in (429, 502) and attempts < retries:
        time.sleep(backoff)
        r = fn(*args, **kwargs)
        attempts += 1
    return r


def _ohlcv_from_json(payload: dict) -> pd.DataFrame:
    candles = payload.get("candles", []) or []
    if not candles:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(candles)
    df["Date"] = pd.to_datetime(df["datetime"], unit="ms").dt.normalize()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    ok = (df["Close"] > 0) & (df["Close"] != float("inf"))
    return df[ok].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def fetch_universe_bars(client, tickers: list[str], as_of=None) -> dict:
    """{ticker: DataFrame(Date,Open,High,Low,Close,Volume)} for the trailing
    FETCH_LOOKBACK_DAYS window ending `as_of` (default: now). One failed
    ticker is logged and skipped, never stops the run -- same doctrine as
    the wheel bot's LiveMarket (one bad symbol must not stop the bot)."""
    end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    start = end - pd.Timedelta(days=FETCH_LOOKBACK_DAYS)
    out, failed = {}, []
    for tk in tickers:
        try:
            r = throttle(client.get_price_history_every_day, tk,
                        start_datetime=start, end_datetime=end)
            if r.status_code != 200:
                failed.append((tk, f"HTTP {r.status_code}"))
                continue
            df = _ohlcv_from_json(r.json())
            if df.empty:
                failed.append((tk, "no usable bars"))
                continue
            out[tk] = df
        except Exception as e:
            failed.append((tk, str(e)))
    if failed:
        print(f"fetch_universe_bars: {len(failed)}/{len(tickers)} tickers failed "
              f"(kept going): {failed[:10]}{' ...' if len(failed) > 10 else ''}")
    return out


def latest_session_date(client, reference: str = "SPY"):
    """The most recent trading day Schwab actually has a candle for, via
    one cheap reference-ticker pull -- NOT the caller's wall-clock date.

    This exists to fix a real bug (found + repro'd 2026-09-07): the VPS
    runs in UTC, and "today" by wall-clock rolls over to the next
    calendar date partway through the ET trading day (e.g. 8pm ET is
    already past midnight UTC). Using wall-clock date as "today" both
    (a) let the bot think a brand-new day had started while the market
    was still in the SAME session, defeating the same-day-touch guard
    and phantom-filling every pending watch at its own gap-day open, and
    (b) on a market holiday, silently relabeled the last real session's
    stale bar as the holiday's date instead of recognizing no new
    session happened. Driving `today` off the data itself instead of the
    wall clock fixes both: a holiday or an early re-trigger returns the
    SAME session date as last time, which the caller's last_run_date
    check already treats as a no-op.

    Returns None if the reference pull fails outright (caller should
    treat that as "can't tell, skip this tick" rather than guessing)."""
    bars = fetch_universe_bars(client, [reference])
    df = bars.get(reference)
    if df is None or df.empty:
        return None
    return df.iloc[-1]["Date"].date()


def bars_for_today(universe_bars: dict) -> dict:
    """{ticker: {"o","h","l","c","prior_close","adv"}} for the LAST row of
    each ticker's frame -- "today" as Schwab currently sees it. `adv` is
    the trailing ADV_LOOKBACK-day average dollar volume, look-back only
    (today's own volume excluded), same definition as the backtest's
    TickerView.get_adv(). A ticker with fewer than ADV_LOOKBACK+1 rows in
    the fetch window gets adv=None (unfloored for that ticker, exactly
    like a newly-listed name in the backtest)."""
    out = {}
    for tk, df in universe_bars.items():
        if len(df) < 2:
            continue  # need at least a prior close
        last = df.iloc[-1]
        prior_close = float(df.iloc[-2]["Close"])
        dollar_vol = (df["Close"] * df["Volume"]).iloc[:-1]  # exclude today
        adv = float(dollar_vol.tail(ADV_LOOKBACK).mean()) if len(dollar_vol) >= ADV_LOOKBACK else None
        out[tk] = {"o": float(last["Open"]), "h": float(last["High"]),
                   "l": float(last["Low"]), "c": float(last["Close"]),
                   "prior_close": prior_close, "adv": adv}
    return out
