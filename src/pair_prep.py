# src/pair_prep.py
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from .utils import normalize_timeframe
from src.data_load import load_data  # your provided loader

def load_pair(stock1: str, stock2: str, timeframe: str):
    tf = normalize_timeframe(timeframe)
    df1 = load_data(stock1, tf)
    df2 = load_data(stock2, tf)
    # expect 'Close' column; adapt here if different
    s1 = df1['Close'].rename(stock1)
    s2 = df2['Close'].rename(stock2)
    joined = pd.concat([s1, s2], axis=1).dropna()
    return joined

def compute_returns(joined_prices: pd.DataFrame):
    # simple returns like in PDF
    returns = joined_prices.pct_change().dropna()
    return returns

# src/pair_prep.py

def rolling_hedge_ratio(returns: pd.DataFrame, target: str, other: str, window: int = 60):
    """
    Compute rolling OLS hedge ratio (beta) over a given window.
    returns: DataFrame with two columns (target, other)
    """

    betas = []
    idxs = []

    for i in range(window, len(returns)):
        y = returns[target].iloc[i-window:i]
        X = sm.add_constant(returns[other].iloc[i-window:i])
        model = sm.OLS(y, X).fit()
        beta = model.params[other]
        betas.append(beta)
        idxs.append(returns.index[i])

    beta_series = pd.Series(betas, index=idxs, name="beta")
    return beta_series


def spread_from_hedge(returns, target: str, other: str, beta: float):
    spread = returns[target] - beta * returns[other]
    return spread.to_frame(name='spread')

def adf_test(series, **kwargs):
    # returns adf result tuple
    res = adfuller(series.dropna(), **kwargs)
    return {'ADF Statistic': res[0], 'p-value': res[1], 'Used Lag': res[2], 'Nobs': res[3]}
