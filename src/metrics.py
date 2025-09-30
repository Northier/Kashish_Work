# src/metrics.py
import numpy as np
import pandas as pd
from .utils import annualize_return, annualized_vol, annualization_factor

def performance_from_equity(equity_curve: pd.Series, strategy_returns: pd.Series, timeframe: str):
    """
    equity_curve: (1+returns).cumprod() series (multiplier, e.g., starts near 1)
    strategy_returns: pd.Series same index (simple returns)
    """
    n_periods = len(strategy_returns)
    final = float(equity_curve.iloc[-1])
    total_return = final
    ann_ret = annualize_return(total_return, n_periods, timeframe)
    ann_vol = annualized_vol(strategy_returns, timeframe)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    # max drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = drawdown.min()
    return {
        'total_return_multiplier': total_return,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }
