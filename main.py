# main.py (realistic stat-arb backtest)
from src.pair_prep import load_pair, compute_returns, rolling_hedge_ratio, adf_test
from src.strategy import run_grid
from src.metrics import performance_from_equity
import pandas as pd
import os

STOCK_A = 'ABB'
STOCK_B = 'INDHOTEL'
TIMEFRAMES = ['5min', '15min', '30min', '1h', '1d']

OUT_DIR = './results'
os.makedirs(OUT_DIR, exist_ok=True)

# Max hold per timeframe
MAX_HOLD_MAP = {
    '5min': 5,
    '15min': 8,
    '30min': 12,
    '1h': 15,
    '1d': 20
}

all_summary = []

for tf in TIMEFRAMES:
    print(f"\n=== TIMEFRAME: {tf} ===")
    try:
        joined = load_pair(STOCK_A, STOCK_B, tf)
    except FileNotFoundError as e:
        print("Missing files for timeframe", tf, e)
        continue

    returns = compute_returns(joined)
    
    # --- Rolling hedge ratio ---
    beta_series = rolling_hedge_ratio(returns, STOCK_A, STOCK_B, window=60)
    aligned = returns.loc[beta_series.index].copy()
    spread = aligned[STOCK_A] - beta_series * aligned[STOCK_B]
    spread_df = spread.to_frame(name='spread')
    
    print(f"Rolling hedge ratio (last value): {beta_series.iloc[-1]:.4f}")

    # --- ADF test: only trade if spread is stationary ---
    adf = adf_test(spread_df['spread'])
    print("ADF:", adf)
    if adf['p-value'] > 0.05:
        print("Spread non-stationary. Skipping timeframe.")
        continue

    # --- Run grid search ---
    max_hold = MAX_HOLD_MAP.get(tf, 10)
    results = run_grid(spread_df['spread'], timeframe=tf, max_hold=max_hold)

    # --- Evaluate results ---
    best_by_ann = None
    for r in results:
        strat_df = r['strategy_df']
        equity = (1 + strat_df['strategy_ret']).cumprod()
        metrics = performance_from_equity(equity, strat_df['strategy_ret'], tf)
        r.update(metrics)

        if best_by_ann is None or r['annual_return'] > best_by_ann['annual_return']:
            best_by_ann = r

    if best_by_ann:
        print("Best config:", {
            'lookback': best_by_ann['lookback'],
            'z': best_by_ann['z'],
            'annual_return': round(best_by_ann['annual_return'], 3),
            'sharpe': round(best_by_ann['sharpe'], 3),
            'max_dd': round(best_by_ann['max_drawdown'], 3),
            'num_trades': best_by_ann['num_trades'],
            'win_rate': round(best_by_ann['win_rate'], 3),
            'avg_profit': round(best_by_ann['avg_profit'], 4),
            'avg_loss': round(best_by_ann['avg_loss'], 4)
        })

        summary = {
            'timeframe': tf,
            'beta': beta_series,
            'ADF': adf,
            'best_lookback': best_by_ann['lookback'],
            'best_z': best_by_ann['z'],
            'annual_return': best_by_ann['annual_return'],
            'sharpe': best_by_ann['sharpe'],
            'max_dd': best_by_ann['max_drawdown'],
            'num_trades': best_by_ann['num_trades'],
            'win_rate': best_by_ann['win_rate'],
            'avg_profit': best_by_ann['avg_profit'],
            'avg_loss': best_by_ann['avg_loss']
        }
        all_summary.append(summary)

        # Save outputs
        best_by_ann['equity_curve'].to_csv(f"{OUT_DIR}/equity_{tf}.csv")
        best_by_ann['strategy_df'].to_csv(f"{OUT_DIR}/strategy_{tf}.csv")
    else:
        print("No valid results for this timeframe.")

# Save summary
pd.DataFrame(all_summary).to_csv(f"{OUT_DIR}/summary.csv", index=False)
print("Done. Results in", OUT_DIR)
