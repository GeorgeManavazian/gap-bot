"""Per-variant position economics -- how the SAME wick-fill signal turns
into a position, how it's marked day to day, and what it's worth when the
shared exit trigger fires. The trigger (stop/tp/timeout) is always computed
off the underlying STOCK price in engine.py, unchanged across variants --
only the instrument bought against that signal differs, and that's entirely
contained in the filler below.

A filler that can't price an entry (no listed contract, no quote, illiquid)
returns None from price_entry() and the candidate is skipped for THAT
variant only -- the stock variant and the other option variant still see
and can take the same signal. This is why three variants run as three
fully independent step_one_day() calls with three independent states,
rather than one shared position list: their accepted-candidate sets can
diverge on any given day.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

# A ChainProvider fetches (and should cache for the caller's lifetime, i.e.
# once per ticker per day) live option-chain data for one ticker. Returns
# None on any failure -- same one-bad-symbol-must-not-stop-the-bot doctrine
# as schwab_data.fetch_universe_bars. Signature: (ticker: str) -> dict | None
ChainProvider = Callable[[str], Optional[dict]]


@dataclass
class EntryQuote:
    units: float                    # shares, or contracts (fractional -- paper account)
    cash_cost: float                # cash debited now, commission included
    extra: dict = field(default_factory=dict)  # filler-specific fields stored on the position


class StockFiller:
    """The live anchor config, unchanged -- exists so engine.py has exactly
    one code path for all three variants instead of a stock special case."""
    name = "stock"

    def __init__(self, commission_per_trade: float = 0.0):
        self.commission = commission_per_trade

    def price_entry(self, ticker, entry_price, size_dollars, chain_provider=None, tp_price=None) -> EntryQuote | None:
        return EntryQuote(units=size_dollars / entry_price,
                           cash_cost=size_dollars + self.commission, extra={})

    def mark(self, position, stock_price, chain_provider=None) -> float:
        return position["units"] * stock_price

    def close_value(self, position, exit_price, exit_reason, chain_provider=None) -> float:
        return position["units"] * exit_price - self.commission


class CallFiller:
    """Buy a call instead of the stock, notional-matched (contracts x 100
    ~= size_dollars worth of underlying). Config is real-chain-data-derived,
    not guessed: see live/config.py VARIANTS['call'] for provenance --
    every config LOST money on the 2yr real-chain backtest (vault-71,
    2026-09-08), 14 DTE / 0.30 delta lost the least. Forward-tested anyway
    per owner ruling 2026-09-08, to see whether live intraday wick fills
    recover the -0.8pt EOD-fill handicap the historical backtest was stuck
    with (chain data there was EOD-only; this filler gets same-day live
    quotes).  Expect this to lose money; that's the honest prior."""
    name = "call"

    def __init__(self, target_dte: int, target_delta: float, commission_per_contract: float = 0.65):
        self.target_dte = target_dte
        self.target_delta = target_delta
        self.commission = commission_per_contract

    def price_entry(self, ticker, entry_price, size_dollars, chain_provider, tp_price=None) -> EntryQuote | None:
        from live import schwab_options as so
        chain = chain_provider(ticker)
        c = so.pick_by_delta_dte(chain, self.target_dte, self.target_delta)
        if c is None or c["ask"] <= 0:
            return None  # no quotable contract today -- skip this ticker for the call ledger
        contracts = size_dollars / (entry_price * 100)
        cash_cost = contracts * c["ask"] * 100 + self.commission
        return EntryQuote(units=contracts, cash_cost=cash_cost,
                           extra={"symbol": c["symbol"], "strike": c["strike"], "expiry": c["expiry"]})

    def mark(self, position, stock_price, chain_provider) -> float:
        from live import schwab_options as so
        chain = chain_provider(position["ticker"]) if chain_provider else None
        c = so.quote_by_symbol(chain, position["symbol"]) if chain else None
        if c is not None:
            mid = (c["bid"] + c["ask"]) / 2
            return position["units"] * mid * 100
        return position["units"] * max(stock_price - position["strike"], 0.0) * 100

    def close_value(self, position, exit_price, exit_reason, chain_provider) -> float:
        from live import schwab_options as so
        chain = chain_provider(position["ticker"]) if chain_provider else None
        c = so.quote_by_symbol(chain, position["symbol"]) if chain else None
        if c is not None and c["bid"] > 0:
            proceeds = position["units"] * c["bid"] * 100
        else:
            # rolled off the chain (expired, delisted) -- settle at intrinsic,
            # matching backtest_long_call_real.py's expiry handling
            proceeds = position["units"] * max(exit_price - position["strike"], 0.0) * 100
        return proceeds - self.commission


class SpreadFiller:
    """Long an ATM-ish call, short a call struck at the signal's OWN
    gap-fill target (tp_price) -- the short leg needs no free width
    parameter, it's the level the trade is already betting on. PLACEHOLDER
    config: see live/config.py VARIANTS['spread'] -- vault-85's real-chain
    backtest was still running when this was written. Update target_dte
    there the moment real numbers land; do not treat this as validated."""
    name = "spread"

    def __init__(self, target_dte: int, long_delta: float = 0.50, commission_per_contract: float = 0.65):
        self.target_dte = target_dte
        self.long_delta = long_delta
        self.commission = commission_per_contract

    def price_entry(self, ticker, entry_price, size_dollars, chain_provider, tp_price=None) -> EntryQuote | None:
        from live import schwab_options as so
        chain = chain_provider(ticker)
        long_c = so.pick_by_delta_dte(chain, self.target_dte, self.long_delta)
        if long_c is None or long_c["ask"] <= 0:
            return None
        short_c = so.pick_by_strike_near(chain, self.target_dte, tp_price)
        if short_c is None or short_c["bid"] <= 0 or short_c["strike"] <= long_c["strike"]:
            return None  # no usable short leg above the long strike today
        contracts = size_dollars / (entry_price * 100)
        debit = (long_c["ask"] - short_c["bid"]) * contracts * 100
        if debit <= 0:
            return None  # priced as a credit or free -- data looks bad, skip rather than trust it
        cash_cost = debit + 2 * self.commission
        return EntryQuote(units=contracts, cash_cost=cash_cost, extra={
            "long_symbol": long_c["symbol"], "long_strike": long_c["strike"],
            "short_symbol": short_c["symbol"], "short_strike": short_c["strike"],
            "expiry": long_c["expiry"],
        })

    def _leg_quotes(self, position, chain_provider):
        from live import schwab_options as so
        chain = chain_provider(position["ticker"]) if chain_provider else None
        long_c = so.quote_by_symbol(chain, position["long_symbol"]) if chain else None
        short_c = so.quote_by_symbol(chain, position["short_symbol"]) if chain else None
        return long_c, short_c

    def mark(self, position, stock_price, chain_provider) -> float:
        long_c, short_c = self._leg_quotes(position, chain_provider)
        if long_c is not None and short_c is not None:
            long_mid = (long_c["bid"] + long_c["ask"]) / 2
            short_mid = (short_c["bid"] + short_c["ask"]) / 2
            return position["units"] * (long_mid - short_mid) * 100
        long_intrinsic = max(stock_price - position["long_strike"], 0.0)
        short_intrinsic = max(stock_price - position["short_strike"], 0.0)
        return position["units"] * (long_intrinsic - short_intrinsic) * 100

    def close_value(self, position, exit_price, exit_reason, chain_provider) -> float:
        long_c, short_c = self._leg_quotes(position, chain_provider)
        if long_c is not None and short_c is not None and long_c["bid"] > 0:
            proceeds = position["units"] * (long_c["bid"] - short_c["ask"]) * 100
        else:
            long_intrinsic = max(exit_price - position["long_strike"], 0.0)
            short_intrinsic = max(exit_price - position["short_strike"], 0.0)
            proceeds = position["units"] * (long_intrinsic - short_intrinsic) * 100
        return proceeds - 2 * self.commission


def make_filler(variant: str, cfg: dict):
    if variant == "stock":
        return StockFiller(commission_per_trade=cfg.get("commission_per_trade", 0.0))
    if variant == "call":
        return CallFiller(cfg["target_dte"], cfg["target_delta"],
                           cfg.get("commission_per_contract", 0.65))
    if variant == "spread":
        return SpreadFiller(cfg["target_dte"], cfg.get("long_delta", 0.50),
                             cfg.get("commission_per_contract", 0.65))
    raise ValueError(f"unknown variant: {variant}")
