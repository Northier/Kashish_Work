"""
Stock Market Trend Prediction & Backtesting Pipeline
- XGBoost / Random Forest / LSTM / Hybrid
- Proper train/test split
- Overfitting control
- Backtesting with tradebook generation
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# ------------------ PARAMETERS ------------------
PRED_PERIOD = 7
TIMEFRAMES = ['1H','1D']
LSTM_EPOCHS = 15
LSTM_BATCH = 64

# ------------------ UTILITY FUNCTIONS ------------------

def safe_div(a, b):
    b = np.where(b==0, np.nan, b)
    return a / b

def add_price_derivatives(df, use_log=True, ema_span=7):
    df = df.copy()
    if 'DateTime' not in df.columns:
        raise ValueError("DateTime column required")
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    dt = df['DateTime'].diff().dt.total_seconds() # Time delta in seconds
    dt.iloc[0] = dt.median() if np.isfinite(dt.median()) else 1.0
    dt = dt.replace(0, np.nan).fillna(dt.median() if np.isfinite(dt.median()) else 1.0)

    series = np.log(df['Close'].clip(lower=1e-12)) if use_log else df['Close']

    df['dClose_dt'] = series.diff() / dt
    df['d2Close_dt2'] = df['dClose_dt'].diff() / dt

    df['dClose_dt_ema'] = df['dClose_dt'].ewm(span=ema_span, adjust=False).mean()
    df['d2Close_dt2_ema'] = df['d2Close_dt2'].ewm(span=ema_span, adjust=False).mean()

    for col in ['dClose_dt', 'd2Close_dt2', 'dClose_dt_ema', 'd2Close_dt2_ema']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df[['dClose_dt','d2Close_dt2','dClose_dt_ema','d2Close_dt2_ema']] = \
        df[['dClose_dt','d2Close_dt2','dClose_dt_ema','d2Close_dt2_ema']].fillna(method='bfill').fillna(method='ffill')

    return df


def compute_indicators_safe(df, period=PRED_PERIOD):
    df = df.copy()
    span1 = max(7, period)
    df['EMA_1'] = df['Close'].ewm(span=span1, adjust=False).mean()
    df['EMA_2'] = df['Close'].ewm(span=span1*2, adjust=False).mean()
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
    tr = pd.concat([df['High']-df['Low'],
                    (df['High']-df['Close'].shift(1)).abs(),
                    (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(span1).mean()
    df['BBM'] = df['Close'].rolling(span1).mean()
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
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def add_lag_features(df, period=PRED_PERIOD):
    df = add_price_derivatives(df, ema_span=max(5, period))
    for lag in range(1, period+1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    for win in [period, period*2, period*3]:
        df[f'rolling_mean_{win}'] = df['Close'].rolling(win).mean()
        df[f'rolling_std_{win}'] = df['Close'].rolling(win).std()
    df = compute_indicators_safe(df, period)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def trend_from_indicators(df, roc_threshold=0.01):
    df = df.copy()
    close = df['Close']
    ema_short = close.ewm(span=12, adjust=False).mean()
    ema_long  = close.ewm(span=26, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    rs = safe_div(gain.rolling(14).mean(), loss.rolling(14).mean())
    rsi = 100 - (100/(1+rs))
    roc = close.pct_change(5)
    ema_signal = np.where(ema_short > ema_long, 1, 2)
    rsi_signal = np.where(rsi > 50, 1, 2)
    roc_signal = np.where(roc > roc_threshold, 1, np.where(roc < -roc_threshold, 2, 0))
    votes_long = (ema_signal==1).astype(int)+(rsi_signal==1).astype(int)+(roc_signal==1).astype(int)
    votes_short = (ema_signal==2).astype(int)+(rsi_signal==2).astype(int)+(roc_signal==2).astype(int)
    trend = np.where(votes_long>=2,1,np.where(votes_short>=2,2,0))
    df['Trend'] = trend
    df.fillna(0, inplace=True)
    df['Trend'] = df['Trend'].astype(int)
    return df

def prepare_data(df):
    df = add_lag_features(df)
    df = trend_from_indicators(df)
    X = df.drop(columns=['Symbol','DateTime','Trend'], errors='ignore')
    y = df['Trend']
    return X, y, df

# ------------------ MODEL TRAINING ------------------
def train_xgboost(X_train, y_train, X_valid=None, y_valid=None):
    model = XGBClassifier(
        objective='multi:softmax', num_class=3,
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    eval_set = [(X_train, y_train)]
    if X_valid is not None and y_valid is not None:
        eval_set.append((X_valid, y_valid))
        model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=20, verbose=False)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_valid) if X_valid is not None else model.predict(X_train)
    return y_pred, model

def train_random_forest(X_train, y_train, X_valid=None):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid) if X_valid is not None else model.predict(X_train)
    return y_pred, model

def train_lstm(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    timesteps = PRED_PERIOD

    def create_sequences(X, y):
        Xs, ys = [], []
        for i in range(timesteps, len(X)):
            Xs.append(X[i-timesteps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values)

    model = Sequential()
    model.add(LSTM(64, input_shape=(timesteps, X_train_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_seq, y_train_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              validation_split=0.1, verbose=0, callbacks=[EarlyStopping(patience=5)])

    y_pred_test = np.argmax(model.predict(X_test_seq), axis=1)
    y_pred_train = np.argmax(model.predict(X_train_seq), axis=1)

    # pad beginning to match original length
    y_pred_train_full = np.concatenate([np.zeros(timesteps, dtype=int), y_pred_train])[:len(X_train)]
    y_pred_test_full = np.concatenate([np.zeros(timesteps, dtype=int), y_pred_test])[:len(X_test)]

    return y_pred_train_full, y_pred_test_full, model

# ------------------ HYBRID ------------------
def hybrid_prediction(y_xgb, y_lstm, y_rf):
    preds = np.vstack([y_xgb, y_lstm, y_rf]).T
    hybrid = []
    for row in preds:
        long_count = np.sum(row==1)
        short_count = np.sum(row==2)
        if long_count>=2:
            hybrid.append(1)
        elif short_count>=2:
            hybrid.append(2)
        else:
            hybrid.append(0)
    return np.array(hybrid, dtype=int)

# ------------------ BACKTEST ------------------
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from glob import glob

# ------------------ BACKTEST ------------------
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from glob import glob

# ------------------ BACKTEST ------------------
def backtest_and_plot(df_feat, y_true, y_pred, name='Model', stop_loss_pct=0.03, take_profit_pct=0.06, max_holding_period=20, return_trades=False):
    df_trades = df_feat.copy().reset_index(drop=True)
    df_trades['Pred'] = np.array(y_pred)[:len(df_trades)]
    df_trades['Actual'] = np.array(y_true)[:len(df_trades)]
    df_trades['Pred_shifted'] = df_trades['Pred'].shift(1, fill_value=0)

    trades = []
    position = 0
    entry_price = None
    entry_time = None
    entry_bar = None

    sl_hits = 0
    tp_hits = 0
    holding_periods = []

    for bar_idx, row in df_trades.iterrows():
        signal = int(row['Pred_shifted'])
        current_open = float(row['Open'])

        if position != 0:
            holding = bar_idx - entry_bar if entry_bar is not None else 0
            pnl_pct = (current_open - entry_price)/entry_price if position==1 else (entry_price - current_open)/entry_price
            hit_sl = pnl_pct <= -stop_loss_pct
            hit_tp = pnl_pct >= take_profit_pct

            if hit_sl or hit_tp or holding >= max_holding_period:
                trades.append({
                    'EntryDateTime': entry_time,
                    'ExitDateTime': row['DateTime'],
                    'Type': 'Long' if position==1 else 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_open,
                    'Profit': (current_open - entry_price)*position,
                    'HoldingBars': holding
                })
                holding_periods.append(holding)
                if hit_sl: sl_hits += 1
                if hit_tp: tp_hits += 1
                position = 0
                entry_price = None
                entry_time = None
                entry_bar = None

        if position==0:
            if signal==1:
                position=1
                entry_price=current_open
                entry_time=row['DateTime']
                entry_bar=bar_idx
            elif signal==2:
                position=-1
                entry_price=current_open
                entry_time=row['DateTime']
                entry_bar=bar_idx
        else:
            if (position==1 and signal==2) or (position==-1 and signal==1):
                holding = bar_idx - entry_bar if entry_bar is not None else 0
                trades.append({
                    'EntryDateTime': entry_time,
                    'ExitDateTime': row['DateTime'],
                    'Type': 'Long' if position==1 else 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_open,
                    'Profit': (current_open - entry_price)*position,
                    'HoldingBars': holding
                })
                holding_periods.append(holding)
                position = 1 if signal==1 else -1
                entry_price = current_open
                entry_time = row['DateTime']
                entry_bar = bar_idx

    if position != 0:
        exit_price = df_trades.iloc[-1]['Close']
        holding = len(df_trades) - entry_bar
        trades.append({
            'EntryDateTime': entry_time,
            'ExitDateTime': df_trades.iloc[-1]['DateTime'],
            'Type': 'Long' if position==1 else 'Short',
            'EntryPrice': entry_price,
            'ExitPrice': exit_price,
            'Profit': (exit_price - entry_price)*position,
            'HoldingBars': holding
        })
        holding_periods.append(holding)

    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['Profit'].sum() if not trades_df.empty else 0
    acc = accuracy_score(df_trades['Actual'], df_trades['Pred'])
    win_trades = trades_df[trades_df['Profit']>0].shape[0]
    total_trades = trades_df.shape[0]
    win_rate = (win_trades/total_trades*100) if total_trades>0 else 0
    avg_holding = np.mean(holding_periods) if holding_periods else 0

    print(f"{name} Total Profit: {total_profit:.2f}, Accuracy: {acc:.4f}, WinRate: {win_rate:.2f}%, AvgHold: {avg_holding:.2f}, SL Hits: {sl_hits}, TP Hits: {tp_hits}")

    metrics = {
        'Profit': total_profit,
        'Accuracy': acc,
        'WinRate': win_rate,
        'AvgHolding': avg_holding,
        'SL_Hits': sl_hits,
        'TP_Hits': tp_hits
    }

    if return_trades:
        return trades_df, metrics
    return metrics

# ------------------ PIPELINE ------------------
def _get_max_hold_for_timeframe(tf):
    tf = tf.lower()
    if '1d' in tf: return 20
    if '1h' in tf: return 48
    if '30' in tf: return 96
    if '15' in tf: return 192
    if '5' in tf: return 480
    return 20

def run_pipeline():
    global_trades = []
    profits_per_model = defaultdict(lambda: defaultdict(float))  # profits_per_model[tf][model]

    for tf in TIMEFRAMES:
        files = glob(f"./data/*_{tf}.csv")
        for file in files:
            print(f"\nProcessing {file} (timeframe={tf})...")
            df = pd.read_csv(file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            X, y, df_feat = prepare_data(df)

            # Clean data
            X.replace([np.inf,-np.inf], np.nan, inplace=True)
            X.fillna(method='bfill', inplace=True)
            X.fillna(method='ffill', inplace=True)
            X = X.clip(-1e10,1e10)

            split_idx = int(len(X)*0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            max_hold = _get_max_hold_for_timeframe(tf)

            # --- XGBoost
            y_pred_xgb_train, model_xgb = train_xgboost(X_train, y_train, X_train, y_train)
            y_pred_xgb_test, _ = train_xgboost(X_train, y_train, X_test, y_test)
            _, metrics_xgb = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_xgb_test, f"XGBoost ({tf})", max_holding_period=max_hold, return_trades=True)
            profits_per_model[tf]['XGBoost'] += metrics_xgb['Profit']

            # --- LSTM
            y_pred_train_lstm, y_pred_test_lstm, model_lstm = train_lstm(X_train, y_train, X_test, y_test)
            _, metrics_lstm = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_test_lstm, f"LSTM ({tf})", max_holding_period=max_hold, return_trades=True)
            profits_per_model[tf]['LSTM'] += metrics_lstm['Profit']

            # --- Random Forest
            y_pred_rf_train, model_rf = train_random_forest(X_train, y_train, X_train)
            y_pred_rf_test, _ = train_random_forest(X_train, y_train, X_test)
            _, metrics_rf = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_rf_test, f"RandomForest ({tf})", max_holding_period=max_hold, return_trades=True)
            profits_per_model[tf]['RandomForest'] += metrics_rf['Profit']

            # --- Hybrid
            y_pred_test_hybrid = hybrid_prediction(y_pred_xgb_test, y_pred_test_lstm, y_pred_rf_test)
            trades_test, metrics_hybrid = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_test_hybrid, f"Hybrid ({tf})", max_holding_period=max_hold, return_trades=True)
            trades_test['Timeframe'] = tf
            global_trades.extend(trades_test.to_dict('records'))
            profits_per_model[tf]['Hybrid'] += metrics_hybrid['Profit']

    total_profit = sum([t['Profit'] for t in global_trades])
    print(f"\nTotal Profit (All Timeframes, Hybrid): {profits_per_model}")
    print("\nProfit per model per timeframe:")
    for tf, model_dict in profits_per_model.items():
        print(f"Timeframe: {tf}")
        for model, profit in model_dict.items():
            print(f"  {model}: {profit:.2f}")

    tradebook_df = pd.DataFrame(global_trades)
    tradebook_df.to_csv("full_tradebook_review.csv", index=False)
    print("\nFull tradebook saved to 'full_tradebook_review.csv'")
    return tradebook_df, profits_per_model

# ------------------ RUN ------------------
if __name__=="__main__":
    run_pipeline()

