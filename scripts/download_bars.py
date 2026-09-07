"""Download 10 years of daily OHLCV for every current S&P 500 ticker via
yfinance. Chunked (Yahoo rate-limits large single requests), resumable (skips
any ticker that already has a cached parquet), retries on failure, writes one
parquet per ticker to daily_bars/. Pure research pull -- nothing here touches
the wheel bot repo or its venv beyond importing already-installed yfinance."""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
BARS_DIR = HERE.parent / "data" / "daily_bars"
BARS_DIR.mkdir(parents=True, exist_ok=True)

END = pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)

tickers = pd.read_csv(HERE.parent / "data" / "sp500_constituents.csv")["Symbol"].tolist()
# yfinance wants '-' not '.' for share classes (BRK.B -> BRK-B)
tickers = [t.replace(".", "-") for t in tickers]

todo = [t for t in tickers if not (BARS_DIR / f"{t}.parquet").exists()]
print(f"{len(tickers)} tickers total, {len(todo)} still to fetch "
      f"({len(tickers) - len(todo)} already cached)")

CHUNK = 25
failed = []
for i in range(0, len(todo), CHUNK):
    chunk = todo[i:i + CHUNK]
    ok_this_chunk = []
    for attempt in range(3):
        try:
            data = yf.download(chunk, start=START, end=END, group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
            break
        except Exception as e:
            print(f"  chunk {i//CHUNK}: download error attempt {attempt+1}: {e}")
            time.sleep(5)
    else:
        print(f"  chunk {i//CHUNK}: FAILED after 3 attempts, tickers: {chunk}")
        failed.extend(chunk)
        continue

    for t in chunk:
        try:
            if len(chunk) == 1:
                df = data
            else:
                df = data[t] if t in data.columns.get_level_values(0) else None
            if df is None or df.empty or df["Close"].dropna().empty:
                failed.append(t)
                continue
            df = df.dropna(how="all").reset_index()
            df.to_parquet(BARS_DIR / f"{t}.parquet", index=False)
            ok_this_chunk.append(t)
        except Exception as e:
            print(f"    {t}: save error {e}")
            failed.append(t)

    n_cached = len(list(BARS_DIR.glob("*.parquet")))
    print(f"  chunk {i//CHUNK+1}/{(len(todo)+CHUNK-1)//CHUNK}: "
          f"{len(ok_this_chunk)}/{len(chunk)} ok ({n_cached} total cached)")
    time.sleep(1.5)

print(f"\ndone. {len(list(BARS_DIR.glob('*.parquet')))} tickers cached. "
      f"{len(failed)} failed: {failed}")
if failed:
    (HERE / "failed_tickers.txt").write_text("\n".join(failed))
