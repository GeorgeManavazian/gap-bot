"""Gap-fill study across the S&P 500, 10 years of daily bars.

DEFINITIONS
  gap_pct = (Open[t] - Close[t-1]) / Close[t-1] * 100
  Down gap: Open[t] < Close[t-1] (gap_pct < 0). "Fills" when a later day's
    High reaches back UP to Close[t-1].
  Up gap:   Open[t] > Close[t-1] (gap_pct > 0). "Fills" when a later day's
    Low reaches back DOWN to Close[t-1].
  Buckets (by |gap_pct|): 1-2%, 3-5%, 5-10% -- exactly as asked. The 2-5%
    and >10% ranges are reported separately as a sanity check, not silently
    dropped.
  Fill timing buckets: same day (day 0's own High/Low already reaches it --
    yes, this can happen: a gap down that reverses intraday), same week
    (within 5 trading days incl. day 0), same month (within 21 trading days),
    same quarter (within 63 trading days), never (not filled within 63
    trading days -- reported as its own honest bucket, not assumed a loss).

THE TRADING RULE (down-gaps only, exactly as specified): on a down-gap day,
buy at Open[t]; take-profit at Close[t-1] (the pre-gap level). Profit is
computed ONLY for trades that actually filled within the 63-trading-day
window; trades that never filled in that window are reported separately as
unresolved, with their actual price move at the 63-day mark so nothing is
swept under the rug.

Stock only -- no options, no slippage/commission model (all explicitly out
of scope per the ask). This is a plain historical-frequency study, not a
promotable bench artifact -- it never touches the wheel-bot repo.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
HORIZON = 63  # trading days ~ 1 quarter

BUCKETS = [("1-2%", 1.0, 2.0), ("2-3%", 2.0, 3.0), ("3-5%", 3.0, 5.0),
           ("5-10%", 5.0, 10.0), ("10%+", 10.0, np.inf)]


def load_ticker(path: Path) -> pd.DataFrame | None:
    df = pd.read_parquet(path)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    need = {"Date", "Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < HORIZON + 5:
        return None
    return df


def find_gaps(df: pd.DataFrame, ticker: str) -> list[dict]:
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    dates = df["Date"].values
    n = len(df)
    out = []
    for t in range(1, n - 1):  # need at least 1 day of history after for "never filled" to mean something; last row has no forward data
        prior_close = c[t - 1]
        if prior_close <= 0:
            continue
        gap_pct = (o[t] - prior_close) / prior_close * 100
        if abs(gap_pct) < 1.0:
            continue
        direction = "down" if gap_pct < 0 else "up"
        end = min(t + HORIZON, n - 1)
        fill_day = None
        for k in range(t, end + 1):
            if direction == "down":
                if h[k] >= prior_close:
                    fill_day = k - t
                    break
            else:
                if l[k] <= prior_close:
                    fill_day = k - t
                    break
        horizon_reached = end - t  # actual trading days available, may be < HORIZON near the end of history
        if fill_day is None:
            last_close = c[end]
            unresolved_move_pct = (last_close - o[t]) / o[t] * 100
        else:
            unresolved_move_pct = None
        trade_pnl_pct = None
        if fill_day is not None:
            # long entered at Open[t], TP at prior_close, for DOWN gaps only
            if direction == "down":
                trade_pnl_pct = (prior_close - o[t]) / o[t] * 100
        out.append({
            "ticker": ticker, "date": dates[t], "gap_pct": gap_pct,
            "abs_gap_pct": abs(gap_pct), "direction": direction,
            "fill_day": fill_day, "horizon_reached": horizon_reached,
            "unresolved_move_pct": unresolved_move_pct,
            "trade_pnl_pct": trade_pnl_pct,
        })
    return out


def bucket_of(abs_gap):
    for name, lo, hi in BUCKETS:
        if lo <= abs_gap < hi:
            return name
    return None


def main():
    files = sorted(BARS_DIR.glob("*.parquet"))
    print(f"{len(files)} ticker files found")
    all_gaps = []
    n_loaded = 0
    for p in files:
        df = load_ticker(p)
        if df is None:
            continue
        n_loaded += 1
        all_gaps.extend(find_gaps(df, p.stem))
    print(f"{n_loaded} tickers usable, {len(all_gaps)} gap events >=1% found")

    g = pd.DataFrame(all_gaps)
    g["bucket"] = g["abs_gap_pct"].apply(bucket_of)
    g.to_parquet((HERE.parent / "results" / "all_gaps.parquet"), index=False)

    def fill_timing_row(sub):
        n = len(sub)
        if n == 0:
            return None
        same_day = (sub["fill_day"] == 0).sum()
        same_week = (sub["fill_day"] <= 4).sum()      # incl day 0, within 5 trading days
        same_month = (sub["fill_day"] <= 20).sum()    # within 21 trading days
        same_quarter = (sub["fill_day"] <= 62).sum()  # within 63 trading days
        never = sub["fill_day"].isna().sum()
        return {
            "n_gaps": n,
            "same_day_%": round(100 * same_day / n, 1),
            "same_week_%": round(100 * same_week / n, 1),
            "same_month_%": round(100 * same_month / n, 1),
            "same_quarter_%": round(100 * same_quarter / n, 1),
            "never_in_quarter_%": round(100 * never / n, 1),
            "median_fill_days": sub["fill_day"].median(),
        }

    print("\n" + "=" * 100)
    print("FILL-RATE TABLE  (n = 503 S&P500 tickers, 10yr daily bars)")
    print("=" * 100)
    rows = []
    for direction in ["down", "up"]:
        for name, lo, hi in BUCKETS:
            sub = g[(g["direction"] == direction) & (g["bucket"] == name)]
            r = fill_timing_row(sub)
            if r:
                r["direction"] = direction
                r["bucket"] = name
                rows.append(r)
    fill_table = pd.DataFrame(rows)[["direction", "bucket", "n_gaps", "same_day_%",
                                     "same_week_%", "same_month_%",
                                     "same_quarter_%", "never_in_quarter_%",
                                     "median_fill_days"]]
    print(fill_table.to_string(index=False))
    fill_table.to_csv((HERE.parent / "results" / "fill_rate_table.csv"), index=False)

    # --- the trading-rule simulation: down gaps only ------------------------
    print("\n" + "=" * 100)
    print("TRADING RULE: buy Open on a down-gap day, TP at prior close")
    print("(only trades that FILLED within one quarter are counted as resolved)")
    print("=" * 100)
    rows2 = []
    down = g[g["direction"] == "down"]
    for name, lo, hi in BUCKETS:
        sub = down[down["bucket"] == name]
        n = len(sub)
        if n == 0:
            continue
        resolved = sub[sub["fill_day"].notna()]
        unresolved = sub[sub["fill_day"].isna()]
        row = {
            "bucket": name,
            "n_gaps": n,
            "n_resolved": len(resolved),
            "resolved_%": round(100 * len(resolved) / n, 1),
            "avg_profit_%_per_resolved_trade": round(resolved["trade_pnl_pct"].mean(), 3) if len(resolved) else None,
            "median_days_held_resolved": resolved["fill_day"].median() if len(resolved) else None,
            "n_unresolved": len(unresolved),
            "avg_move_%_at_quarter_mark_unresolved": round(unresolved["unresolved_move_pct"].mean(), 3) if len(unresolved) else None,
        }
        rows2.append(row)
    trade_table = pd.DataFrame(rows2)
    print(trade_table.to_string(index=False))
    trade_table.to_csv((HERE.parent / "results" / "trading_rule_table.csv"), index=False)

    # overall (all down gaps >=1%, pooled) headline
    resolved_all = down[down["fill_day"].notna()]
    print(f"\nPOOLED (all down gaps >=1%, n={len(down)}):")
    print(f"  resolved within 1 quarter: {len(resolved_all)} "
          f"({100*len(resolved_all)/len(down):.1f}%)")
    print(f"  avg profit per resolved trade: {resolved_all['trade_pnl_pct'].mean():.3f}%")
    print(f"  median days held: {resolved_all['fill_day'].median():.0f}")
    win_rate = (resolved_all["trade_pnl_pct"] > 0).mean() * 100
    print(f"  win rate among resolved (should be ~100% by construction -- TP-only exit): {win_rate:.1f}%")


if __name__ == "__main__":
    main()
