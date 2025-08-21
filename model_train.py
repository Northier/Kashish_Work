import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

TIMEFRAMES = ['5min', '15min', '30min', '1hr', '1D']
MODEL_PATH = "xgb_stock_trend.json"
SCALER_PATH = "xgb_scaler.gz"
TEST_SIZE = 0.1
PRED_PERIOD = 7  

def load_all_timeframes(timeframes=TIMEFRAMES):
    data_list = []
    for tf in timeframes:
        print(f"Processing {tf} Data")
        files = glob(f"resampled_{tf}_NSE_*.csv")
        for file in files:
            df = pd.read_csv(file, parse_dates=['DateTime'])
            df = df.sort_values('DateTime').reset_index(drop=True)
            df['Timeframe'] = tf
            data_list.append(df)
    data = pd.concat(data_list, ignore_index=True).sort_values('DateTime').reset_index(drop=True)
    print("Final combined shape:", data.shape)
    return data

def calculate_dynamic_thresholds(df, pred_period=PRED_PERIOD, window=50, upper_q=0.66, lower_q=0.33):
    returns = df['Close'].pct_change(periods=pred_period).dropna()
    rolling_up = returns.rolling(window).quantile(upper_q)
    rolling_down = returns.rolling(window).quantile(lower_q)
    UP_TH = rolling_up.iloc[-1] if not pd.isna(rolling_up.iloc[-1]) else returns.quantile(upper_q)
    DOWN_TH = rolling_down.iloc[-1] if not pd.isna(rolling_down.iloc[-1]) else returns.quantile(lower_q)
    return UP_TH, DOWN_TH

def compute_indicators(df, period=PRED_PERIOD):
    df = df.copy()
    span1 = max(7, period)
    span2 = span1 * 2

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # EMA
    df['EMA_1'] = df['Close'].ewm(span=span1, adjust=False).mean()
    df['EMA_2'] = df['Close'].ewm(span=span2, adjust=False).mean()

    # Rolling volatility
    df['rolling_vol_1'] = df['log_return'].rolling(max(span1, 2)).std()
    df['rolling_vol_2'] = df['log_return'].rolling(max(span2, 2)).std()

    # Bollinger Bands
    df['BBM'] = df['Close'].rolling(max(span1,2)).mean()
    tr = pd.concat([df['High'] - df['Low'],
                    (df['High'] - df['Close'].shift(1)).abs(),
                    (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(max(span1,2)).mean()
    df['BBU'] = df['BBM'] + 2 * df['ATR']
    df['BBL'] = df['BBM'] - 2 * df['ATR']

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(max(span1,2)).mean()
    avg_loss = loss.rolling(max(span1,2)).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_short = df['Close'].ewm(span=12, adjust=False).mean()
    ema_long = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_short - ema_long

    # Stochastic
    low_min = df['Low'].rolling(max(span1,2)).min()
    high_max = df['High'].rolling(max(span1,2)).max()
    df['StochK'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['StochD'] = df['StochK'].rolling(3).mean()

    # CCI
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    MA_TP = TP.rolling(max(span1,2)).mean()
    MD = TP.rolling(max(span1,2)).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (TP - MA_TP) / (0.015 * MD)

    # ADX
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(max(span1,2)).sum() / tr.rolling(max(span1,2)).sum()
    minus_di = 100 * pd.Series(minus_dm).rolling(max(span1,2)).sum() / tr.rolling(max(span1,2)).sum()
    df['ADX'] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)  # avoid div0

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def add_features(df, period=PRED_PERIOD):
    df = df.copy()
    for lag in range(1, period + 1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    for win in [period, period*2, period*3]:
        df[f'rolling_mean_{win}'] = df['Close'].rolling(max(win,2)).mean()
        df[f'rolling_std_{win}'] = df['Close'].rolling(max(win,2)).std()
    df = compute_indicators(df, period)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def preprocess(data, period=PRED_PERIOD):
    data = add_features(data, period)
    UP_TH, DOWN_TH = calculate_dynamic_thresholds(data, period)
    print(f"Auto thresholds -> UP_TH: {UP_TH:.4f}, DOWN_TH: {DOWN_TH:.4f}")
    
    data['Return'] = data['Close'].pct_change(period)
    data['Trend'] = data['Return'].apply(lambda x: 1 if x > UP_TH else (2 if x < DOWN_TH else 0))

    base_features = ['Open','High','Low','Close','Volume','log_return']
    lag_features = [f'Close_lag_{lag}' for lag in range(1, period + 1)]
    rolling_features = [f'rolling_mean_{win}' for win in [period, period*2, period*3]] + \
                       [f'rolling_std_{win}' for win in [period, period*2, period*3]]
    indicators = ['EMA_1','EMA_2','rolling_vol_1','rolling_vol_2',
                  'BBL','BBM','BBU','ATR','RSI','MACD','StochK','StochD','CCI','ADX']

    all_features = base_features + lag_features + rolling_features + indicators
    X = data[all_features].copy()
    y = data['Trend'].astype(int).copy()

    X.fillna(method='bfill', inplace=True)
    X.fillna(method='ffill', inplace=True)
    y = y[X.index]
    return data, X, y, all_features


def train_pipeline(period=PRED_PERIOD):
    data = load_all_timeframes()
    data, X, y, features = preprocess(data, period)

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    print("Original class distribution:", Counter(y_train))

    counts = Counter(y_train)
    majority_count = max(counts.values())
    sampling_strategy = {cls: majority_count for cls, cnt in counts.items() if cnt < majority_count}

    if sampling_strategy:
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
        print("Resampled class distribution:", Counter(y_train_res))
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Hold','Buy','Sell']))
    joblib.dump(model, MODEL_PATH)

    return model, scaler, features

def calculate_trading_profit(df, pred_period=PRED_PERIOD):
    profits = []
    position = None
    entry_price = 0
    entry_time = None

    for idx, row in df.iterrows():
        if row['Trend'] == 1 and position is None:  # BUY signal
            position = 'long'
            entry_price = row['Close']
            entry_time = row['DateTime']
        elif row['Trend'] == 2 and position == 'long':  # SELL signal
            sell_idx = idx + pred_period
            if sell_idx >= len(df):
                sell_idx = len(df) - 1
            exit_price = df.iloc[sell_idx]['Close']
            exit_time = df.iloc[sell_idx]['DateTime']
            profit = exit_price - entry_price
            profits.append({
                'Symbol': df['Symbol'].iloc[0],
                'Timeframe': df['Timeframe'].iloc[0],
                'EntryTime': entry_time,
                'ExitTime': exit_time,
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'Profit': profit
            })
            position = None
    return pd.DataFrame(profits)

def total_profit(profit_df):
    return profit_df['Profit'].sum()

if __name__ == "__main__":
    model, scaler, features = train_pipeline(PRED_PERIOD)
    data = load_all_timeframes()
    data, X, y, _ = preprocess(data, PRED_PERIOD)

    all_symbols = data['Symbol'].unique()
    all_profits = []

    for sym in all_symbols:
        df_sym = data[data['Symbol'] == sym].copy()
        for tf in df_sym['Timeframe'].unique():
            df_tf = df_sym[df_sym['Timeframe'] == tf].copy().reset_index(drop=True)
            profit_df = calculate_trading_profit(df_tf, PRED_PERIOD)
            all_profits.append(profit_df)

    final_profit_df = pd.concat(all_profits, ignore_index=True)
    print("Total Profit:", total_profit(final_profit_df))
    final_profit_df.to_csv("trading_profit.csv", index=False)
