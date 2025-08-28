import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, balanced_accuracy_score
import xgboost as xgb
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TIMEFRAMES = ['5min', '15min', '30min', '1hr', '1D']
TEST_SIZE = 0.1
PRED_PERIOD = 7
ALLOW_SHORT = True
QTY = 1
USE_PRED = True
TAKE_PROFIT = 0.01    # +1%
STOP_LOSS = -0.005    # -0.5%

# -------------------------------
# Data loading
# -------------------------------
def load_all_timeframes(timeframes=TIMEFRAMES):
    data_list = []
    for tf in timeframes:
        print(f"Processing {tf} Data")
        files = glob(f"resampled_{tf}_NSE*.csv")
        for file in files:
            df = pd.read_csv(file, parse_dates=['DateTime'])
            df = df.sort_values('DateTime').reset_index(drop=True)
            df['Timeframe'] = tf
            data_list.append(df)
    data = pd.concat(data_list, ignore_index=True).sort_values('DateTime').reset_index(drop=True)
    print("Final combined shape:", data.shape)
    return data

# -------------------------------
# Dynamic thresholds
# -------------------------------
def calculate_dynamic_thresholds(df, pred_period=PRED_PERIOD, window=50, upper_q=0.66, lower_q=0.33):
    future_returns = (df['Close'].shift(-pred_period) - df['Close']) / df['Close']
    returns = future_returns.dropna()
    rolling_up = returns.rolling(window).quantile(upper_q)
    rolling_down = returns.rolling(window).quantile(lower_q)
    UP_TH = rolling_up.iloc[-1] if not pd.isna(rolling_up.iloc[-1]) else returns.quantile(upper_q)
    DOWN_TH = rolling_down.iloc[-1] if not pd.isna(rolling_down.iloc[-1]) else returns.quantile(lower_q)
    return UP_TH, DOWN_TH

# -------------------------------
# Feature engineering
# -------------------------------
def compute_indicators(df, period=PRED_PERIOD):
    df = df.copy()
    span1 = max(7, period)
    span2 = span1 * 2
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['EMA_1'] = df['Close'].ewm(span=span1, adjust=False).mean()
    df['EMA_2'] = df['Close'].ewm(span=span2, adjust=False).mean()
    df['rolling_vol_1'] = df['log_return'].rolling(max(span1, 2)).std()
    df['rolling_vol_2'] = df['log_return'].rolling(max(span2, 2)).std()
    df['BBM'] = df['Close'].rolling(max(span1, 2)).mean()
    tr = pd.concat([df['High'] - df['Low'],
                    (df['High'] - df['Close'].shift(1)).abs(),
                    (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(max(span1, 2)).mean()
    df['BBU'] = df['BBM'] + 2 * df['ATR']
    df['BBL'] = df['BBM'] - 2 * df['ATR']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(max(span1, 2)).mean()
    avg_loss = loss.rolling(max(span1, 2)).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_short = df['Close'].ewm(span=12, adjust=False).mean()
    ema_long = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_short - ema_long

    low_min = df['Low'].rolling(max(span1, 2)).min()
    high_max = df['High'].rolling(max(span1, 2)).max()
    df['StochK'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['StochD'] = df['StochK'].rolling(3).mean()

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def add_features(df, period=PRED_PERIOD):
    df = df.copy()
    for lag in range(1, period + 1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    for win in [period, period*2, period*3]:
        df[f'rolling_mean_{win}'] = df['Close'].rolling(max(win, 2)).mean()
        df[f'rolling_std_{win}'] = df['Close'].rolling(max(win, 2)).std()
    df = compute_indicators(df, period)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def preprocess(data, period=PRED_PERIOD, save_csv=True, filename="trend_prediction.csv"):
    data = add_features(data, period)
    UP_TH, DOWN_TH = calculate_dynamic_thresholds(data, pred_period=period)
    print(f"Dynamic thresholds -> UP_TH: {UP_TH:.4f}, DOWN_TH: {DOWN_TH:.4f}")

    data['FutureClose'] = data['Close'].shift(-period)
    data = data.dropna(subset=['FutureClose']).copy()
    data['Return'] = (data['FutureClose'] - data['Close']) / data['Close']
    data['Trend'] = data['Return'].apply(lambda x: 1 if x > UP_TH else (2 if x < DOWN_TH else 0))

    base_features = ['Open','High','Low','Close','Volume','log_return']
    lag_features = [f'Close_lag_{lag}' for lag in range(1, period + 1)]
    rolling_features = [f'rolling_mean_{win}' for win in [period, period*2, period*3]] + \
                       [f'rolling_std_{win}' for win in [period, period*2, period*3]]
    indicators = ['EMA_1','EMA_2','rolling_vol_1','rolling_vol_2','BBL','BBM','BBU','ATR','RSI','MACD','StochK','StochD']
    all_features = base_features + lag_features + rolling_features + indicators
    X = data[all_features].copy()
    y = data['Trend'].astype(int).copy()
    X.fillna(method='bfill', inplace=True)
    X.fillna(method='ffill', inplace=True)
    y = y[X.index]

    if save_csv:
        trend_df = data[['DateTime','Symbol','Open','High','Low','Close','Volume','Trend','Return']]
        trend_df.to_csv(filename, index=False)
        print(f"Trend predictions saved to {filename}")

    return data, X, y, all_features

# -------------------------------
# Feature selection with XGB
# -------------------------------
def select_features_with_xgb(X_train, y_train, top_k=20):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    selected_features = importance_df['feature'].head(top_k).tolist()
    print("Selected Features:", selected_features)
    return selected_features

# -------------------------------
# Sequence preparation
# -------------------------------
def create_sequence(X, y, time_stemps=10):
    Xs, ys = [], []
    for i in range(len(X)-time_stemps):
        Xs.append(X[i:(i+time_stemps)].values)
        ys.append(y.iloc[i + time_stemps])
    return np.array(Xs), np.array(ys)

# -------------------------------
# Training pipeline (walk-forward)
# -------------------------------
def train_pipeline_hybrid(period=PRED_PERIOD, top_k=20, time_stamps=10):
    data = load_all_timeframes()
    data, X, y, features = preprocess(data, period)

    split_idx = int(len(X)*(1-TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    select_features = select_features_with_xgb(X_train, y_train, top_k)
    X_train = X_train[select_features]
    X_test = X_test[select_features]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq, y_train_seq = create_sequence(pd.DataFrame(X_train_scaled, columns=select_features), y_train, time_stamps)
    X_test_seq, y_test_seq = create_sequence(pd.DataFrame(X_test_scaled, columns=select_features), y_test, time_stamps)

    # class weights
    cls_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    cls_weights = {i:w for i,w in enumerate(cls_weights)}

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_stamps, len(select_features))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('lstm_stock_trend.h5', save_best_only=True)

    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32,
              validation_split=0.1, callbacks=[early_stopping, model_checkpoint],
              class_weight=cls_weights)

    y_pred_prob = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print("Balanced Accuracy:", balanced_accuracy_score(y_test_seq, y_pred))
    print(classification_report(y_test_seq, y_pred, target_names=['Hold','Buy','Sell']))

    return model, scaler, select_features, data, y_test_seq, y_pred

# -------------------------------
# Backtesting with TP/SL
# -------------------------------
def calculate_profits_all_timeframes(data, symbols, timeframes, pred_period, allow_short, use_pred):
    profits = []
    for sym in symbols:
        for tf in timeframes:
            df_tf = data[(data["Symbol"] == sym) & (data["Timeframe"] == tf)].sort_values('DateTime').reset_index(drop=True)
            position = None
            entry_price, entry_time, entry_idx = 0, None, None
            for idx, row in df_tf.iterrows():
                trend = row['Pred_Trend'] if use_pred and not pd.isna(row['Pred_Trend']) else row['Trend']
                if trend == 1 and position is None:
                    position = 'long'
                    entry_idx = idx
                    entry_price = df_tf.iloc[idx+1]['Open'] if idx+1 < len(df_tf) else row['Close']
                    entry_time = df_tf.iloc[idx+1]['DateTime'] if idx+1 < len(df_tf) else row['DateTime']
                elif position == 'long':
                    change = (row['Close'] - entry_price)/entry_price
                    if change >= TAKE_PROFIT or change <= STOP_LOSS or idx >= entry_idx + pred_period:
                        exit_price = row['Close']
                        exit_time = row['DateTime']
                        profit = (exit_price - entry_price) * QTY
                        profits.append({'Symbol':sym,'Timeframe':tf,'Side':'LONG','EntryTime':entry_time,'ExitTime':exit_time,'EntryPrice':entry_price,'ExitPrice':exit_price,'Profit':profit})
                        position = None
                elif trend == 2 and position is None and allow_short:
                    position = 'short'
                    entry_idx = idx
                    entry_price = df_tf.iloc[idx+1]['Open'] if idx+1 < len(df_tf) else row['Close']
                    entry_time = df_tf.iloc[idx+1]['DateTime'] if idx+1 < len(df_tf) else row['DateTime']
                elif position == 'short':
                    change = (entry_price - row['Close'])/entry_price
                    if change >= TAKE_PROFIT or change <= -STOP_LOSS or idx >= entry_idx + pred_period:
                        exit_price = row['Close']
                        exit_time = row['DateTime']
                        profit = (entry_price - exit_price) * QTY
                        profits.append({'Symbol':sym,'Timeframe':tf,'Side':'SHORT','EntryTime':entry_time,'ExitTime':exit_time,'EntryPrice':entry_price,'ExitPrice':exit_price,'Profit':profit})
                        position = None
    return pd.DataFrame(profits)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    model, scaler, selected_features, data, y_test_seq, y_pred = train_pipeline_hybrid()
    split_idx = int(len(data) * (1 - TEST_SIZE))
    data['Pred_Trend'] = np.nan
    data.loc[split_idx+10:split_idx+10+len(y_pred)-1, 'Pred_Trend'] = y_pred

    all_symbols = data['Symbol'].unique()
    all_profits = []
    for sym in all_symbols:
        for tf in data['Timeframe'].unique():
            profits_df = calculate_profits_all_timeframes(data, [sym], [tf], PRED_PERIOD, ALLOW_SHORT, USE_PRED)
            all_profits.append(profits_df)

    if all_profits:
        final_profit_df = pd.concat(all_profits, ignore_index=True)
        print("Total Profit:", final_profit_df['Profit'].sum())
        final_profit_df.to_csv("Hybrid_Training_Profit.csv", index=False)
    else:
        print("No trades executed.")
