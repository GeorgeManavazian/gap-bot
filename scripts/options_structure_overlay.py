"""Overlay option structures on the gap bot's 508 anchor trades.

First-order model: option value at entry and at exit from Black-Scholes using
the trade's actual entry/exit prices and days held; IV = 20d realized vol at
entry x 1.15 (floor 20%), held flat. Ignores path (a stop hit intra-hold, the
put's path value) and IV change. Good enough for "how much does theta eat".
"""
import numpy as np, pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
pd.set_option('display.width', 220)

R = 0.04
RES = '/Users/georgiemanavazian/Documents/Trading/code/gap-bot/results/confirm2yr_heldout_stops_exclude_1-2pct_friction.csv'
BARS = '/Users/georgiemanavazian/Documents/Trading/code/gap-bot/data/daily_bars/'
STOP = {'2-3%': .11, '3-5%': .16, '5-10%': .26, '10%+': .37}
OPT_FRICTION = 0.03   # per side, % of premium (bid-ask on S&P single names)

def bs(S, K, T, sig, kind):
    if T <= 0:
        return max(0.0, S - K) if kind == 'c' else max(0.0, K - S)
    d1 = (log(S / K) + (R + sig * sig / 2) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
    if kind == 'c':
        return S * norm.cdf(d1) - K * exp(-R * T) * norm.cdf(d2)
    return K * exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

df = pd.read_csv(RES, parse_dates=['entry_date', 'exit_date', 'gap_date'])

# realized-vol proxy at entry
cache = {}
def rv(tk, date):
    if tk not in cache:
        try:
            b = pd.read_parquet(BARS + tk + '.parquet')
            col = [c for c in b.columns if c.lower() == 'close'][0]
            dcol = [c for c in b.columns if c.lower() == 'date']
            if dcol: b = b.set_index(dcol[0])
            b.index = pd.to_datetime(b.index)
            assert b.index.min().year >= 2000, tk + ': bars index is not dates'
            cache[tk] = np.log(b[col]).diff()
        except Exception as e:
            cache[tk] = None
    s = cache[tk]
    if s is None: return np.nan
    w = s[s.index < date].tail(20)
    return w.std() * sqrt(252) if len(w) >= 15 else np.nan

df['rv20'] = [rv(t, d) for t, d in zip(df.ticker, df.entry_date)]
print('rv20 coverage', df.rv20.notna().mean().round(3), df.rv20.describe(percentiles=[.25, .5, .75]).round(2).to_dict())
df['iv'] = (df.rv20.fillna(df.rv20.median()) * 1.15).clip(lower=0.20)
df['target'] = df.entry_price / (1 + df.gap_pct / 100)          # prior close = gap-fill level
df['stop_k'] = df.entry_price * (1 - df.bucket.map(STOP))
df['T_held'] = df.days_held / 252

def sim(row, dte_days, iv_mult=1.0):
    S0, S1, iv = row.entry_price, row.exit_price, row.iv * iv_mult
    T0 = dte_days / 365; T1 = max(T0 - row.days_held / 252, 0)
    out = {}
    # A. long ATM call
    c0 = bs(S0, S0, T0, iv, 'c'); c1 = bs(S1, S0, T1, iv, 'c')
    out['call_prem'] = c0 / S0
    out['call_pnl'] = (c1 * (1 - OPT_FRICTION) - c0 * (1 + OPT_FRICTION)) / S0
    out['call_theta'] = (c0 - bs(S0, S0, T1, iv, 'c')) / S0        # decay if price had not moved
    # B. debit call spread ATM / target
    K2 = row.target
    s0 = c0 - bs(S0, K2, T0, iv, 'c'); s1 = c1 - bs(S1, K2, T1, iv, 'c')
    out['spr_prem'] = s0 / S0
    out['spr_width'] = (K2 - S0) / S0
    out['spr_pnl'] = (s1 * (1 - OPT_FRICTION) - s0 * (1 + OPT_FRICTION)) / S0
    out['spr_capture'] = s1 / (K2 - S0) if row.exit_reason == 'tp_gap_filled' else np.nan
    # C. stock + protective put at stop strike
    p0 = bs(S0, row.stop_k, T0, iv, 'p'); p1 = bs(S1, row.stop_k, T1, iv, 'p')
    out['put_prem'] = p0 / S0
    out['put_pnl'] = (p1 * (1 - OPT_FRICTION) - p0 * (1 + OPT_FRICTION)) / S0
    out['stockput_pnl'] = row.pnl_pct / 100 + out['put_pnl']
    return pd.Series(out)

for dte in (45, 90):
    r = df.apply(sim, axis=1, dte_days=dte)
    d = pd.concat([df, r], axis=1)
    print(f'\n===== DTE {dte} (calendar), IV = rv20 x 1.15 =====')
    print('per-trade avg, % of stock notional:')
    tot = pd.DataFrame({
        'stock_pnl': [df.pnl_pct.mean() / 100],
        'call_pnl': [d.call_pnl.mean()], 'call_prem': [d.call_prem.mean()], 'call_theta_paid': [d.call_theta.mean()],
        'spread_pnl': [d.spr_pnl.mean()], 'spread_prem': [d.spr_prem.mean()], 'spread_capture_at_fill': [d.spr_capture.mean()],
        'stock+put_pnl': [d.stockput_pnl.mean()], 'put_prem': [d.put_prem.mean()],
    }).T
    print((tot * 100).round(2))
    print('\nreturn on CAPITAL deployed (stock=100% notional; call/spread=premium; stock+put=100%+prem):')
    cap = pd.DataFrame({
        'stock': [df.pnl_pct.sum() / 100 / len(df)],
        'call': [d.call_pnl.sum() / d.call_prem.sum()],
        'spread': [d.spr_pnl.sum() / d.spr_prem.sum()],
        'stock+put': [d.stockput_pnl.sum() / (1 + d.put_prem).sum()],
    }).T
    print((cap * 100).round(1))
    print('\nby bucket, avg pnl % notional:')
    print((d.groupby('bucket')[['pnl_pct']].mean() / 100 * 100).round(2).join(
        (d.groupby('bucket')[['call_pnl', 'call_theta', 'spr_pnl', 'spr_capture', 'stockput_pnl', 'put_prem']].mean() * 100).round(2)).join(
        d.groupby('bucket')[['days_held']].median()))
    print('\nby exit reason:')
    print((d.groupby('exit_reason')[['call_pnl', 'call_theta', 'spr_pnl', 'stockput_pnl']].mean() * 100).round(2).join(
        d.groupby('exit_reason')[['pnl_pct']].mean().round(2)))
    print('\nwin rate: stock', (df.pnl_pct > 0).mean().round(3), 'call', (d.call_pnl > 0).mean().round(3),
          'spread', (d.spr_pnl > 0).mean().round(3), 'stock+put', (d.stockput_pnl > 0).mean().round(3))
    # worst-case / tails
    print('p10 pnl: stock', np.percentile(df.pnl_pct, 10).round(2), 'call', (np.percentile(d.call_pnl, 10) * 100).round(2),
          'spread', (np.percentile(d.spr_pnl, 10) * 100).round(2), 'stock+put', (np.percentile(d.stockput_pnl, 10) * 100).round(2))
    if dte == 90:
        # IV sensitivity on the call
        for m in (0.8, 1.3):
            rr = df.apply(sim, axis=1, dte_days=dte, iv_mult=m)
            print(f'  IV x{m}: call avg {rr.call_pnl.mean()*100:.2f}%  prem {rr.call_prem.mean()*100:.2f}%  spread avg {rr.spr_pnl.mean()*100:.2f}%  stock+put {rr.stockput_pnl.mean()*100:.2f}%')
        # theta as share of gross winner edge
        w = d[d.pnl_pct > 0]
        print(f'  winners: stock avg {w.pnl_pct.mean():.2f}%  call avg {w.call_pnl.mean()*100:.2f}%  theta paid {w.call_theta.mean()*100:.2f}%')
        l = d[d.pnl_pct <= 0]
        print(f'  losers : stock avg {l.pnl_pct.mean():.2f}%  call avg {l.call_pnl.mean()*100:.2f}%  spread {l.spr_pnl.mean()*100:.2f}%  stock+put {l.stockput_pnl.mean()*100:.2f}%')
        d.to_csv('/private/tmp/claude-501/-Users-georgiemanavazian-Documents-Trading-vault/9e5e4de8-8208-48f5-83c7-869fb0b374c2/scratchpad/options_overlay_90dte.csv', index=False)
