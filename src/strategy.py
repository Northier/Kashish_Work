import itertools
import pandas as pd
import numpy as np

# realistic costs
TRADING_COST_EQUITY = 0.0015   # 0.15% per leg
SLIPPAGE_EQUITY = 0.0015       # 0.15% per leg
FIXED_FEE = 50                 # flat fee per round-trip
TRADE_NOTIONAL = 50000.0       # smaller notional for 1H
MAX_RET_CLIP = 0.05            # clip extreme returns ±5%


def run_grid(spread_series: pd.Series, timeframe='1h', max_hold=72):
    """
    V2 Arbitrage Strategy tuned for lower timeframes (like 1H)
    """

    if timeframe == '5min':
        lookbacks = [30]; zscores = [1.5]
    elif timeframe == '15min':
        lookbacks = [150]; zscores = [2.75]
    elif timeframe == '30min':
        lookbacks = [120]
        zscores =[2.75]
    if timeframe == '1h':
        lookbacks = [120]
        zscores = [2.0]
    elif timeframe == '1d':
        lookbacks = [150]; zscores = [1.5]
    results = []

    for lookback, z in itertools.product(lookbacks, zscores):
        strategy_df = pd.DataFrame(index=spread_series.index)
        strategy_df['spread'] = spread_series
        strategy_df['mean'] = strategy_df['spread'].rolling(lookback).mean()
        strategy_df['std'] = strategy_df['spread'].rolling(lookback).std()
        strategy_df['zscore'] = (strategy_df['spread'] - strategy_df['mean']) / strategy_df['std']
        strategy_df.loc[strategy_df['std'] == 0, 'zscore'] = 0.0

        # signals
        strategy_df['long'] = strategy_df['zscore'] < -z
        strategy_df['short'] = strategy_df['zscore'] > z
        strategy_df['exit'] = strategy_df['zscore'].abs() < 0.5

        position = 0
        hold_bars = 0
        positions = []

        for i in range(len(strategy_df)):
            if position == 0:
                if strategy_df['long'].iat[i]:
                    position = 1
                    hold_bars = 0
                elif strategy_df['short'].iat[i]:
                    position = -1
                    hold_bars = 0
            else:
                hold_bars += 1
                if strategy_df['exit'].iat[i] or hold_bars >= max_hold:
                    position = 0
                    hold_bars = 0
            positions.append(position)

        strategy_df['position'] = positions

        strategy_df['spread_ret'] = strategy_df['spread'].diff()
        strategy_df['strategy_ret'] = strategy_df['position'].shift(1) * strategy_df['spread_ret']
        strategy_df = strategy_df.dropna()

        trades = strategy_df['position'].diff().abs() > 0
        trade_points = trades[trades].index
        total_pct_cost = (TRADING_COST_EQUITY + SLIPPAGE_EQUITY) * 2

        for t in trade_points:
            fee_as_pct = FIXED_FEE / TRADE_NOTIONAL
            strategy_df.loc[t, 'strategy_ret'] -= total_pct_cost + fee_as_pct

        strategy_df['strategy_ret'] = strategy_df['strategy_ret'].clip(lower=-MAX_RET_CLIP, upper=MAX_RET_CLIP)

        # strategy_df['strategy_ret'] *= np.random.uniform(0.995, 1.005, size=len(strategy_df))

        safe_returns = strategy_df['strategy_ret'].where(strategy_df['strategy_ret'] > -1 + 1e-12, -0.9999999999)
        cum_ret = np.exp(np.log1p(safe_returns).cumsum())
        strategy_df['equity'] = cum_ret

        win_trades, avg_profit, avg_loss = _trade_stats(strategy_df)

        res = {
            'lookback': lookback,
            'z': z,
            'num_periods': len(strategy_df),
            'num_trades': int(trades.sum()),
            'total_return_multiplier': float(cum_ret.iloc[-1]),
            'equity_curve': cum_ret,
            'strategy_df': strategy_df,
            'win_rate': win_trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
        }
        results.append(res)

    return results

def _trade_stats(strategy_df):
    positions = strategy_df['position']
    returns = strategy_df['strategy_ret']
    trades_idx = positions.diff().fillna(0) != 0

    trade_returns = []
    in_trade = False
    cur_trade_idx = None

    for idx, change in trades_idx.items():
        if change:
            if positions.loc[idx] != 0:  # entry
                cur_trade_idx = idx
                in_trade = True
            else:  # exit
                if cur_trade_idx is not None:
                    seg = returns.loc[cur_trade_idx:idx]
                    trade_ret = (1 + seg).prod() - 1
                    trade_returns.append(trade_ret)
                in_trade = False
                cur_trade_idx = None

    # force close last trade
    if in_trade and cur_trade_idx is not None:
        seg = returns.loc[cur_trade_idx:]
        trade_ret = (1 + seg).prod() - 1
        trade_returns.append(trade_ret)

    if len(trade_returns) == 0:
        return 0.0, 0.0, 0.0

    wins = sum(1 for t in trade_returns if t > 0) / len(trade_returns)
    avg_profit = sum(t for t in trade_returns if t > 0) / max(1, sum(1 for t in trade_returns if t > 0))
    avg_loss = sum(t for t in trade_returns if t <= 0) / max(1, sum(1 for t in trade_returns if t <= 0))
    return wins, avg_profit, avg_loss
