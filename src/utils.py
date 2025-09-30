# src/utils.py
from math import sqrt

TIMEFRAME_MAP = {
    '5min': '5Min',
    '15min': '15Min',
    '30min': '30Min',
    '1h': '1H',
    '1d': '1D'
}

BARS_PER_DAY = {
    '5Min': 78,
    '15Min': 26,
    '30Min': 13,
    '1H': 6.5,
    '1D': 1
}

def normalize_timeframe(tf: str) -> str:
    tf_low = tf.lower()
    return TIMEFRAME_MAP.get(tf_low, tf)

def annualization_factor(timeframe: str) -> float:
    tf = normalize_timeframe(timeframe)
    bars_per_day = BARS_PER_DAY.get(tf, 1)
    return 252 * bars_per_day  # 252 trading days per year

def annualize_return(total_return: float, n_periods: int, timeframe: str) -> float:
    if n_periods <= 0:
        return 0.0
    af = annualization_factor(timeframe)
    return total_return ** (af / n_periods) - 1

def annualized_vol(returns_series, timeframe: str):
    af = annualization_factor(timeframe)
    return returns_series.std() * (af ** 0.5)
