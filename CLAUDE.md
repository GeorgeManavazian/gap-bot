# gap-bot — briefing

**What:** S&P 500 gap-down mean-reversion. Buy after a ≥2% down-gap once price reclaims the gap-day open; exit at gap fill (prior close), per-bucket stop, or 63 trading days. Stock only, no options. Own code, own data — independent of etf-bot.

**Read first:** `~/Documents/Trading/vault/11 Gap Bot/_STATUS.md` (the handoff). Then `README.md` here (thesis + measured results). Vault rules: `~/Documents/Trading/vault/_CLAUDE.md`.

**Known facts (2026-09-13):** 503 tickers, 230,803 gap events, 10 yrs daily bars (yfinance). Held-out 2024-09→2026-09: +11.45%, maxDD −14.59%, 174 trades, 64.4% win, 20 slots, 10 bps slippage. Stops fail on correlated panic days. Options variants lose to stock. Paper-ran 2026-09-07→09-12, paused.

**Update 2026-10-05:** nothing proven better than the live config. Best lead = 10 slots + one 50% disaster stop + 63d exit (small, unproven; +24.2% vs +21.1% median rolling 2yr, return/DD tie). Five ranking features failed a one-shot lockbox. The +11.45% anchor is a cold-start number (+25.7% when already running, same dates): compare variants on paired rolling windows. Env bug: parquet `Date` is datetime64[ms], breaks TickerView in the etf-bot venv (cast to `[us]`). Details: vault `11 Gap Bot/2026-10-05 — …` note.

**Rule:** results go in `results/` and get mirrored to the vault section. A finding that isn't in `_STATUS.md` doesn't exist next session.
