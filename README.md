# Gap Bot — research workspace

Stock-only (no options) gap-fill strategy research. Started 2026-09-05.
Full narrative + findings live in the vault: **`vault/11 Gap Bot/_STATUS.md`**
— read that first. This README is just the code map.

## Layout

```
data/
  sp500_constituents.csv   current S&P 500 list (pulled from a public GitHub
                           mirror of the Wikipedia constituents table)
  daily_bars/              one parquet per ticker, 10yr daily OHLCV via
                           yfinance (503 tickers, ~57MB, pulled 2026-09-05)
scripts/                   every analysis script, in the order they were
                           built (see below)
results/                   every CSV/parquet output, plus:
  trade_logs_18/           full per-trade logs, 3 entry styles x 6 bucket
                           scopes, 2yr window -- one row per real trade
  equity_curves/           daily equity series for every backtest variant run
```

## Re-running anything

All scripts are standalone, run from anywhere:
```
cd ~/Documents/Trading/code/etf-bot   # any repo with the shared .venv-live works
.venv-live/bin/python ~/Documents/Trading/code/gap-bot/scripts/<script>.py
```
No new download needed for anything except `download_bars.py` itself — every
other script reads the cached `data/daily_bars/` parquets. Re-run
`download_bars.py` to refresh to a later date; it's resumable (skips tickers
already cached).

## Script order (what each one answers)

1. `download_bars.py` — pulls the 10yr daily bar cache. Run once, or to refresh.
2. `analyze_gaps.py` — the base gap-fill frequency study: how often, how fast,
   by gap size and direction. → `fill_rate_table.csv`, `trading_rule_table.csv`
3. `sizing_stats.py` — signal frequency per day, clustering on panic days,
   max-adverse-excursion (informs stop placement). → `signals_per_day.csv`,
   `mae_table.csv`
4. `stop_loss_sim.py` — per-trade expectancy with a stop-loss added, at
   several widths. → `stop_sim_data_derived.csv`, `stop_sweep_flat.csv`
5. `portfolio_sim.py` / `portfolio_sim_2yr.py` — full $100k account,
   equal-weight 20-slot sizing, WITH vs WITHOUT a stop, 10yr and 2yr.
6. `portfolio_sim_risksized.py` — same, but sized by fixed-dollar risk /
   stop distance instead of equal-weight (found WORSE — see _STATUS).
7. `portfolio_sim_isolated_buckets.py` — each gap-size bucket run ALONE
   (no other bucket competing for slots). The clean read on which bucket
   sizes are actually good.
8. `reentry_analysis.py` — the corrected entry timing: wait for a WICK back
   up to the gap-day open before buying, instead of buying blind at the open.
9. `reentry_confirmed_close.py` — a STRONGER version: wait for a full closed
   day back above the gap open (found this kills the edge — see _STATUS).
10. `full_18_backtest.py` — THE key result. 3 entry styles (naive / wick /
    close_confirm) x 6 bucket scopes (5 isolated + all-combined), each its
    own real $100k account, full trade logs. → `results/trade_logs_18/`,
    `full_18_summary.csv`
11. `slot_and_priority_sweep.py` — found that the 20-slot cap wasn't the
    problem; the ticker-alphabetical tie-break was. Sweeps slot count x
    tie-break rule (alpha vs biggest-gap-first). → `slot_priority_sweep_summary.csv`

## The one-line status

**Best result found so far:** wick entry, all buckets, 20 slots, tie-break by
biggest gap first: **+73.5% / 2yr, max DD −17.7%, ret/maxDD 4.17** — beats
every isolated bucket and every other configuration tested. Not yet
confirmed on the full 10-year window, and zero slippage/commission modeled.
See the vault status note for the full validation checklist before this
becomes a live bot.
