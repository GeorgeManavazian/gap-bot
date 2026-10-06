"""Live option chain quotes for the call/spread paper fillers, via the SAME
authenticated Schwab client schwab_data.py already uses for stock bars --
one login, one token, both data types. Independent of the etf-bot options
store (ThetaData, lapsed 2026-07-25): this is Schwab's own live chain
endpoint, a real-time broker quote, not a historical vendor pull, and it
is what makes forward-testing possible at all when the historical store
can't be refreshed.

Smoke-tested live 2026-09-08 (scripts/smoke_test_options.py, real SPY
chain) -- found and fixed one real bug in the process: Schwab's own
volatility/delta/gamma/theta/vega/rho fields come back as a -999.0
sentinel on every contract, liquid or not (bid/ask/OI/volume are fine).
Fixed by self-computing IV+delta from the real bid/ask mid via
Black-Scholes bisection, same fix pattern etf-bot's iv_solve.py already
uses for the wheel bot. Post-fix, pick_by_delta_dte(target=0.30) on a
real SPY chain returned delta=0.3068 -- correct. If schwab-py's call
signature or JSON shape ever changes, only this file is in question --
fillers.py never touches schwab-py directly, it only calls the
functions below."""
from __future__ import annotations
import datetime as dt
import math

from live.schwab_data import throttle

MAX_DTE_LOOKAHEAD = 90  # covers every variant's target_dte with room to pick a real listed expiry

# Confirmed live 2026-09-08 (scripts/smoke_test_options.py against a real
# account): Schwab's own volatility/delta/gamma/theta/vega/rho fields come
# back as the sentinel -999.0 on EVERY contract, liquid or not (bid/ask/OI/
# volume are all real and fine). This is the same gap etf-bot's
# iv_solve.py was built to route around for the wheel bot -- self-compute
# IV from the real bid/ask mid via Black-Scholes bisection, then delta from
# that, rather than trust the broken vendor field. Not cross-imported (gap-
# bot stays self-contained, same doctrine as schwab_data.py) -- copied in
# miniature here.


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs_call(S, K, T, r, q, sigma):
    if T <= 1e-6 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _call_delta(S, K, T, r, q, sigma):
    if T <= 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _norm_cdf(d1)


def _solve_iv_and_delta(mid, S, K, T, r, q):
    """Bisection IV solve off the real market mid, then delta from that IV.
    Returns None if the mid is non-positive or brackets a price BS can't
    match (deep-ITM intrinsic-only quotes, T too small) -- caller treats a
    None delta as 'can't select by delta', not a crash."""
    if mid <= 0 or T <= 1e-6 or S <= 0 or K <= 0:
        return None
    lo, hi = 0.01, 5.0
    if _bs_call(S, K, T, r, q, lo) > mid or _bs_call(S, K, T, r, q, hi) < mid:
        return None
    for _ in range(50):
        mid_sigma = (lo + hi) / 2
        price = _bs_call(S, K, T, r, q, mid_sigma)
        if price > mid:
            hi = mid_sigma
        else:
            lo = mid_sigma
    sigma = (lo + hi) / 2
    return _call_delta(S, K, T, r, q, sigma)


def fetch_chain(client, ticker: str, max_dte: int = MAX_DTE_LOOKAHEAD) -> dict | None:
    """{"underlying": float, "calls": [{"strike","expiry","dte","bid","ask",
    "delta","symbol","open_interest","volume"}]} for `ticker`, calls only
    (every filler here is calls or a call spread). None on any failure --
    caller skips this ticker today, same one-bad-symbol doctrine as
    fetch_universe_bars. A contract with no two-sided quote (bid or ask
    missing/zero) is dropped here so every filler downstream only ever
    sees quotable contracts."""
    try:
        r = throttle(client.get_option_chain, ticker,
                     contract_type=client.Options.ContractType.CALL,
                     to_date=dt.date.today() + dt.timedelta(days=max_dte))
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception:
        return None

    underlying = payload.get("underlyingPrice")
    # Schwab's own delta/volatility fields are a broken -999.0 sentinel on
    # every contract (confirmed live 2026-09-08) -- self-compute instead.
    # interestRate/dividendYield come back as percentages (e.g. 4.5, not
    # 0.045); fall back to a conservative rate if the field is ever absent.
    r_pct = payload.get("interestRate")
    q_pct = payload.get("dividendYield")
    r_rate = (r_pct / 100) if r_pct is not None else 0.045
    q_rate = (q_pct / 100) if q_pct is not None else 0.0

    today = dt.date.today()
    calls = []
    for exp_key, strikes in (payload.get("callExpDateMap") or {}).items():
        try:
            expiry = dt.date.fromisoformat(exp_key.split(":")[0])
        except ValueError:
            continue
        dte = (expiry - today).days
        if dte < 1:
            continue  # 0-DTE: IV solve is unstable at T~0, and no variant targets it
        T = dte / 365
        for strike_str, contracts in (strikes or {}).items():
            if not contracts:
                continue
            c = contracts[0]
            bid, ask = c.get("bid"), c.get("ask")
            if bid is None or ask is None or ask <= 0:
                continue
            try:
                strike = float(strike_str)
            except ValueError:
                continue
            bid, ask = float(bid), float(ask)
            delta = None
            if underlying is not None:
                delta = _solve_iv_and_delta((bid + ask) / 2, underlying, strike, T, r_rate, q_rate)
            calls.append({
                "strike": strike, "expiry": expiry.isoformat(), "dte": dte,
                "bid": bid, "ask": ask,
                "delta": delta, "symbol": c.get("symbol"),
                "open_interest": c.get("openInterest"), "volume": c.get("totalVolume"),
            })
    if not calls:
        return None
    return {"underlying": underlying, "calls": calls}


def pick_by_delta_dte(chain: dict | None, target_dte: int, target_delta: float) -> dict | None:
    """Nearest expiry >= target_dte (falls back to whatever's closest if
    nothing reaches it), then nearest |delta - target_delta| at that
    expiry. None if the chain has no delta field on anything at that
    expiry -- a filler should skip rather than guess a strike blind."""
    if not chain or not chain.get("calls"):
        return None
    candidates = [c for c in chain["calls"] if c["dte"] >= target_dte]
    pool = candidates if candidates else chain["calls"]
    best_dte = min(c["dte"] for c in pool)
    same_expiry = [c for c in pool if c["dte"] == best_dte and c.get("delta") is not None]
    if not same_expiry:
        return None
    return min(same_expiry, key=lambda c: abs(c["delta"] - target_delta))


def pick_by_strike_near(chain: dict | None, target_dte: int, target_strike: float | None) -> dict | None:
    """Same expiry-selection rule as pick_by_delta_dte, nearest strike to
    target_strike -- used for the spread's short leg, struck at the
    signal's own gap-fill target price."""
    if not chain or not chain.get("calls") or target_strike is None:
        return None
    candidates = [c for c in chain["calls"] if c["dte"] >= target_dte]
    pool = candidates if candidates else chain["calls"]
    best_dte = min(c["dte"] for c in pool)
    same_expiry = [c for c in pool if c["dte"] == best_dte]
    if not same_expiry:
        return None
    return min(same_expiry, key=lambda c: abs(c["strike"] - target_strike))


def quote_by_symbol(chain: dict | None, symbol: str) -> dict | None:
    """Re-find a previously-picked contract in a fresh chain pull, to mark
    or settle an already-open position. None if it's rolled off the chain
    (expired, delisted, no longer quoted) -- callers fall back to intrinsic
    value in that case, same as backtest_long_call_real.py's expiry
    handling."""
    if not chain:
        return None
    for c in chain.get("calls", []):
        if c.get("symbol") == symbol:
            return c
    return None


def make_chain_provider(client):
    """A ChainProvider (see fillers.py) that fetches live and memoizes for
    the calling day -- run_daily.py creates one fresh instance per tick so
    the cache never spans days."""
    cache: dict[str, dict | None] = {}

    def provider(ticker: str) -> dict | None:
        if ticker not in cache:
            cache[ticker] = fetch_chain(client, ticker)
        return cache[ticker]

    return provider
