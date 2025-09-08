"""
STOCK MARKET TREND PREDICTION
XGBoost vs LSTM vs Hybrid
Includes: feature engineering, LSTM sequences, hybrid voting, backtest & plots
"""

from asyncio import Handle
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from xgboost import XGBClassifier, plot_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

PRED_PERIOD = 7
TIMEFRAMES = [ '1D']
CLASSES = ['Sideways', 'Up', 'Down']  # Trend classes
LSTM_EPOCHS = 15
LSTM_BATCH = 64


def safe_div(a, b):
    b = np.where(b==0, np.nan, b)
    return a / b

def add_price_derivatives(df, use_log=True, ema_span=7):
    df = df.copy()
    if 'DateTime' not in df.columns:
        raise ValueError("DateTime column required")
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    dt = df['DateTime'].diff().dt.total_seconds() # Time delta in seconds
    dt.iloc[0] = dt.median() if np.isfinite(dt.median()) else 1.0  # Handle first NaN value
    dt = dt.replace(0, np.nan).fillna(dt.median() if np.isfinite(dt.median()) else 1.0)  # Avoid division by zero

    series = np.log(df['Close'].clip(lower=1e-12)) if use_log else df['Close']  # Avoid log(0)
    
    df['dClose_dt'] = series.diff() / dt # First derivative
    df['d2Close_dt2'] = df['dClose_dt'].diff() / dt # Second derivative

    # Smooth derivatives
    df['dClose_dt_ema'] = df['dClose_dt'].ewm(span=ema_span, adjust=False).mean()
    df['d2Close_dt2_ema'] = df['d2Close_dt2'].ewm(span=ema_span, adjust=False).mean()

    # Handle infinities and NaNs
    for col in ['dClose_dt', 'd2Close_dt2', 'dClose_dt_ema', 'd2Close_dt2_ema']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    df[['dClose_dt','d2Close_dt2','dClose_dt_ema','d2Close_dt2_ema']] = \
        df[['dClose_dt','d2Close_dt2','dClose_dt_ema','d2Close_dt2_ema']].fillna(method='bfill').fillna(method='ffill')
    
    return df

def compute_indicators_safe(df, period=PRED_PERIOD):
    df = df.copy()
    span1 = max(7, period)
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
    df['EMA_1'] = df['Close'].ewm(span=span1, adjust=False).mean()
    df['EMA_2'] = df['Close'].ewm(span=span1*2, adjust=False).mean()
    df['BBM'] = df['Close'].rolling(span1).mean()
    
    tr = pd.concat([df['High']-df['Low'],
                    (df['High']-df['Close'].shift(1)).abs(),
                    (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(span1).mean()
    df['BBU'] = df['BBM'] + 2*df['ATR']
    df['BBL'] = df['BBM'] - 2*df['ATR']
    
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(span1).mean()
    avg_loss = loss.rolling(span1).mean()
    
    rs = safe_div(avg_gain, avg_loss)
    df['RSI'] = 100 - (100/(1+rs))
    
    low_min = df['Low'].rolling(span1).min()
    high_max = df['High'].rolling(span1).max()
    df['StochK'] = safe_div(100*(df['Close']-low_min), (high_max-low_min))
    df['StochD'] = df['StochK'].rolling(3).mean()
    df['ROC'] = df['Close'].pct_change(period)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e6, upper=1e6)
    
    return df

def add_lag_features(df, period=PRED_PERIOD):
    df = df.copy()
    df = add_price_derivatives(df, use_log=True, ema_span=max(5, period))

    for lag in range(1, period+1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    for win in [period, period*2, period*3]:
        df[f'rolling_mean_{win}'] = df['Close'].rolling(win).mean()
        df[f'rolling_std_{win}'] = df['Close'].rolling(win).std()
    df = compute_indicators_safe(df, period)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def trend_from_indicators(df):
    df = df.copy()
    ema_short = df['Close'].ewm(span=12, adjust=False).mean()
    ema_long  = df['Close'].ewm(span=26, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    roc = df['Close'].pct_change(5)
    
    trend=[]
    for i in range(len(df)):
        if ema_short[i] > ema_long[i] and rsi[i]<70 and roc[i]>0:
            trend.append(1)
        elif ema_short[i]<ema_long[i] and rsi[i]>30 and roc[i]<0:
            trend.append(2)
        else:
            trend.append(0)
    df['Trend']=trend
    return df

def prepare_data(df):
    df = add_lag_features(df, PRED_PERIOD)
    df = trend_from_indicators(df)
    X = df.drop(columns=['Symbol','DateTime','Trend'], errors='ignore')
    y = df['Trend']
    return X, y, df


def train_xgboost(X_train, y_train, X_test):
    model = XGBClassifier(objective='multi:softmax', num_class=3,
                          n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def train_random_forest(X_train, y_train, X_test):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    return y_pred, rf_model
def train_lstm(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    timesteps = PRED_PERIOD
    
    # Create sequences
    def create_sequences(X, y):
        Xs, ys = [], []
        for i in range(timesteps, len(X)):
            Xs.append(X[i-timesteps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values)
    
    # LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(timesteps, X_train_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_seq, y_train_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              validation_split=0.1, verbose=0, callbacks=[EarlyStopping(patience=5)])
    
    # Predict
    y_pred_train = np.argmax(model.predict(X_train_seq), axis=1)
    y_pred_test = np.argmax(model.predict(X_test_seq), axis=1)
    
    # Pad at the beginning to match original X lengths
    y_pred_train_full = np.concatenate([np.zeros(timesteps, dtype=int), y_pred_train])
    y_pred_test_full = np.concatenate([np.zeros(timesteps, dtype=int), y_pred_test])
    
    # Truncate to exactly match X lengths
    y_pred_train_full = y_pred_train_full[:len(X_train)]
    y_pred_test_full = y_pred_test_full[:len(X_test)]
    
    return y_pred_train_full, y_pred_test_full, model


def hybrid_prediction(y_xgb, y_lstm, y_rf, weights=(0.4,0.3,0.3)):
    hybrid_pred = []
    for x,l,r in zip(y_xgb,y_lstm,y_rf):
        scores = np.zeros(3)
        for pred, w in zip([x,l,r], weights):
            scores[pred] += w
        hybrid_pred.append(np.argmax(scores))
    return np.array(hybrid_pred)

def backtest_and_plot(df_feat, y_true, y_pred, name='Model', global_trades=None, return_trades=False):
    """
    Backtest trading signals:
    - Signals are generated at current close.
    - Trades executed at next bar open.
    - Can return tradebook if return_trades=True.
    """
    df_trades = df_feat.copy()  # use the df_feat passed, no indexing needed
    df_trades['Pred'] = y_pred
    df_trades['Actual'] = y_true

    # Shift predictions by 1 bar for realistic next-open execution
    df_trades['Pred_shifted'] = df_trades['Pred'].shift(1, fill_value=0)

    trades = []
    position = 0
    entry_price = 0
    entry_time = None

    for i, row in df_trades.iterrows():
        signal = row['Pred_shifted']

        if position == 0:
            if signal == 1:
                position = 1
                entry_price = row['Open']
                entry_time = row['DateTime']
            elif signal == 2:
                position = -1
                entry_price = row['Open']
                entry_time = row['DateTime']

        else:
            if (position == 1 and signal == 2) or (position == -1 and signal == 1):
                exit_price = row['Open']
                pnl = (exit_price - entry_price) * position
                trades.append({
                    'EntryDateTime': entry_time,
                    'ExitDateTime': row['DateTime'],
                    'Type': 'Long' if position == 1 else 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Profit': pnl
                })
                # flip position
                position = 1 if signal == 1 else -1
                entry_price = row['Open']
                entry_time = row['DateTime']

    # Close any open position at last bar's close
    if position != 0:
        exit_price = df_trades.iloc[-1]['Close']
        pnl = (exit_price - entry_price) * position
        trades.append({
            'EntryDateTime': entry_time,
            'ExitDateTime': df_trades.iloc[-1]['DateTime'],
            'Type': 'Long' if position == 1 else 'Short',
            'EntryPrice': entry_price,
            'ExitPrice': exit_price,
            'Profit': pnl
        })

    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['Profit'].sum()
    print(f"{name} Total Profit: {total_profit}")
    print(f"{name} Overall Accuracy: {accuracy_score(y_true, y_pred)}")
    
    if not trades_df.empty:
        print(f"{name} Tradebook (first 5 trades):")
        print(trades_df.head())

    if global_trades is not None:
        global_trades.extend(trades)

    if return_trades:
        return trades_df, total_profit
    return total_profit

def create_tradebook(model_tradebooks):
    """
    Combine all trades from different models and datasets into a single DataFrame
    for review.
    """
    tradebook_list = []

    for model_name, sets in model_tradebooks.items():
        for dataset_type, df_trades in sets.items():
            if df_trades is not None and not df_trades.empty:
                df = df_trades.copy()
                df['Model'] = model_name
                df['Dataset'] = dataset_type
                tradebook_list.append(df)

    if tradebook_list:
        tradebook = pd.concat(tradebook_list, ignore_index=True)
        # Sort by EntryDateTime for easier review
        tradebook.sort_values(by='EntryDateTime', inplace=True)
        return tradebook
    else:
        return pd.DataFrame()  # empty if no trades

def run_pipeline():
    global_trades = []  # all trades combined
    model_tradebooks = defaultdict(dict)  # save train/test tradebooks

    for tf in TIMEFRAMES:
        files = glob(f"./data/*_{tf}.csv")
        for file in files:
            print(f"\nProcessing {file}...")
            df = pd.read_csv(file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            X, y, df_feat = prepare_data(df)

            X.replace([np.inf,-np.inf], np.nan, inplace=True)
            X.fillna(method='bfill', inplace=True)
            X.fillna(method='ffill', inplace=True)
            X = X.clip(-1e10,1e10)

            # Split train/test
            split_idx = int(len(X)*0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # --- XGBoost
            y_pred_xgb, model_xgb = train_xgboost(X_train, y_train, X)
            # Profit & trades for training set
            trades_train, profit_train = backtest_and_plot(
                df_feat.iloc[:split_idx], y_train, y_pred_xgb[:split_idx], 
                'XGBoost Train', return_trades=True
            )

            trades_test, profit_test = backtest_and_plot(
                df_feat.iloc[split_idx:], y_test, y_pred_xgb[split_idx:], 
                'XGBoost Test', return_trades=True
            )

            model_tradebooks['XGBoost']['Train'] = trades_train
            model_tradebooks['XGBoost']['Test'] = trades_test

            # Combine all trades
            global_trades.extend(trades_train.to_dict('records'))
            global_trades.extend(trades_test.to_dict('records'))

            print(f"XGBoost — Train Profit: {profit_train}, Test Profit: {profit_test}\n")

            # --- LSTM
            y_pred_train_lstm, y_pred_test_lstm, model_lstm = train_lstm(X_train, y_train, X_test, y_test)

            # Backtest
            trades_train_lstm, profit_train_lstm = backtest_and_plot(
                df_feat.iloc[:split_idx], y_train, y_pred_train_lstm, 'LSTM Train', return_trades=True
            )
            trades_test_lstm, profit_test_lstm = backtest_and_plot(
                df_feat.iloc[split_idx:], y_test, y_pred_test_lstm, 'LSTM Test', return_trades=True
            )

            model_tradebooks['LSTM']['Train'] = trades_train_lstm
            model_tradebooks['LSTM']['Test'] = trades_test_lstm
            global_trades.extend(trades_train_lstm.to_dict('records'))
            global_trades.extend(trades_test_lstm.to_dict('records'))
            print(f"LSTM — Train Profit: {profit_train_lstm}, Test Profit: {profit_test_lstm}\n")

            # --- Hybrid
            y_pred_rf, model_rf = train_random_forest(X_train, y_train, X)
            # For train hybrid
            y_pred_train_hybrid = hybrid_prediction(
                y_pred_xgb[:split_idx], y_pred_train_lstm, y_pred_rf[:split_idx]
            )
            
            # For test hybrid
            y_pred_test_hybrid = hybrid_prediction(
                y_pred_xgb[split_idx:], y_pred_test_lstm, y_pred_rf[split_idx:]
            )
            trades_train_h, profit_train_h = backtest_and_plot(df_feat.iloc[:split_idx], y_train, y_pred_train_hybrid,
                                                               'Hybrid Train', return_trades=True)
            trades_test_h, profit_test_h = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_test_hybrid,
                                                             'Hybrid Test', return_trades=True)
            model_tradebooks['Hybrid']['Train'] = trades_train_h
            model_tradebooks['Hybrid']['Test'] = trades_test_h
            global_trades.extend(trades_train_h.to_dict('records'))
            global_trades.extend(trades_test_h.to_dict('records'))
            print(f"Hybrid — Train Profit: {profit_train_h}, Test Profit: {profit_test_h}\n")

    # Combined profit for all trades
    combined_profit = sum([t['Profit'] for t in global_trades])
    print(f"\nCombined Total Profit (Train+Test, all models): {combined_profit}")

    # Create a full tradebook
    full_tradebook = create_tradebook(model_tradebooks)

    # Save to CSV for review
    full_tradebook.to_csv("full_tradebook_review.csv", index=False)
    print("\nFull tradebook saved to 'full_tradebook_review.csv'")
    print(full_tradebook.head(20))  # preview first 20 trades

    return model_tradebooks, global_trades



if __name__=="__main__":
    run_pipeline()

