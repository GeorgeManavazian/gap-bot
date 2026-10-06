"""A synthetic, no-network stand-in for schwab_options.fetch_chain, for
--dry-run wiring tests of the call/spread fillers. This is NOT a pricing
validation -- flat 40% IV, no smile, no real market spread -- it exists
only to exercise price_entry/mark/close_value end to end (contract
selection, cash math, position bookkeeping) before ever calling the real
Schwab endpoint. Same shape as schwab_options.fetch_chain's return value,
so make_chain_provider below is a drop-in swap with the real one."""
from __future__ import annotations
import math

FLAT_IV = 0.40
RATE = 0.045
DTES = [7, 14, 21, 30, 45, 60, 90]
STRIKE_STEPS = [round(1 + s / 100, 2) for s in range(-30, 31, 5)]  # +-30% around spot, 5% steps


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs_call(S, K, T, r, sigma):
    if T <= 1e-6:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _call_delta(S, K, T, r, sigma):
    if T <= 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def synthetic_chain(ticker: str, spot: float) -> dict | None:
    if spot is None or spot <= 0:
        return None
    calls = []
    for dte in DTES:
        T = dte / 365
        for mult in STRIKE_STEPS:
            K = round(spot * mult, 2)
            if K <= 0:
                continue
            mid = _bs_call(spot, K, T, RATE, FLAT_IV)
            if mid < 0.02:
                continue
            delta = _call_delta(spot, K, T, RATE, FLAT_IV)
            spread = max(mid * 0.06, 0.02)  # synthetic 6%-of-mid round-trip spread
            calls.append({
                "strike": K, "expiry": f"fixture+{dte}d", "dte": dte,
                "bid": round(max(mid - spread / 2, 0.01), 2),
                "ask": round(mid + spread / 2, 2),
                "delta": round(delta, 4),
                "symbol": f"{ticker}_FIX_{dte}_{K}", "open_interest": 100, "volume": 10,
            })
    return {"underlying": spot, "calls": calls} if calls else None


def make_fixture_chain_provider(spot_lookup):
    """spot_lookup: ticker -> current stock price (float) or None, e.g. a
    closure over the day's fixture bars. Memoized per call like the real
    provider, so a --dry-run tick fetches each ticker's synthetic chain
    once."""
    cache: dict[str, dict | None] = {}

    def provider(ticker: str) -> dict | None:
        if ticker not in cache:
            cache[ticker] = synthetic_chain(ticker, spot_lookup(ticker))
        return cache[ticker]

    return provider
