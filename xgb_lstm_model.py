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

PARAMS_MAP = {
    '5Min': {'pred_period': 15, 'lstm_timesteps': 30, 'max_holding': 48, 'stop_loss_multi': 1.5, 'take_profit_multi': 3.0},
    '15Min': {'pred_period': 7, 'lstm_timesteps': 15, 'max_holding': 96, 'stop_loss_multi': 1.5, 'take_profit_multi': 3.0},
    '30Min': {'pred_period': 7, 'lstm_timesteps': 10, 'max_holding': 96, 'stop_loss_multi': 2.0, 'take_profit_multi': 4.0},
    '1H': {'pred_period': 7, 'lstm_timesteps': 7, 'max_holding': 48, 'stop_loss_multi': 2.0, 'take_profit_multi': 4.0},
    '1D': {'pred_period': 20, 'lstm_timesteps': 20, 'max_holding': 20, 'stop_loss_multi': 2.5, 'take_profit_multi': 5.0}
}
LSTM_EPOCHS = 15
LSTM_BATCH = 64
TIMEFRAMES = ['1H','1D'] #5Min, 15Min, 30Min

def safe_div(a, b):
    b = np.where(b==0, np.nan, b)
    return a / b

def add_price_derivatives(df, use_log=True, ema_span=7):
    df = df.copy()
    if 'DateTime' not in df.columns:
        raise ValueError("DateTime column required")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    dt = df['DateTime'].diff().dt.total_seconds()

    # ✅ Fix SettingWithCopyWarning
    median_val = dt.median() if np.isfinite(dt.median()) else 1.0
    dt = dt.copy()
    dt.loc[dt.index[0]] = median_val

    dt = dt.replace(0, np.nan).fillna(median_val)
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

def compute_indicators_safe(df, period):
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

def add_lag_features(df, period):
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

def trend_from_indicators(df, period, roc_threshold=0.01):
    df = df.copy()
    close = df['Close']
    ema_short = close.ewm(span=max(5, period), adjust=False).mean()
    ema_long  = close.ewm(span=max(10, period*2), adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    rs = safe_div(gain.rolling(max(10,period)).mean(), loss.rolling(max(10,period)).mean())
    rsi = 100 - (100/(1+rs))
    roc = close.pct_change(max(5, period//2))
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

def prepare_data(df, tf):
    params = PARAMS_MAP.get(tf, PARAMS_MAP['1D'])
    pred_period = params['pred_period']
    df = add_lag_features(df, period=pred_period)
    df = trend_from_indicators(df, period=pred_period)
    X = df.drop(columns=['Symbol','DateTime','Trend'], errors='ignore')
    y = df['Trend']
    return X, y, df

# MODEL TRAINING
def train_xgboost(X_train, y_train, X_valid=None, y_valid=None):
    model = XGBClassifier(
        objective='multi:softmax', num_class=3,
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss',
        early_stopping_rounds=20   # ✅ moved here
    )
    eval_set = [(X_train, y_train)]
    if X_valid is not None and y_valid is not None:
        eval_set.append((X_valid, y_valid))
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_valid) if X_valid is not None else model.predict(X_train)
    return y_pred, model

def train_random_forest(X_train, y_train, X_valid=None):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid) if X_valid is not None else model.predict(X_train)
    return y_pred, model

def train_lstm(X_train, y_train, X_test, y_test, tf):
    params = PARAMS_MAP.get(tf, PARAMS_MAP['1D'])
    timesteps = params['lstm_timesteps']
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
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
    y_pred_train_full = np.concatenate([np.zeros(timesteps, dtype=int), y_pred_train])[:len(X_train)]
    y_pred_test_full = np.concatenate([np.zeros(timesteps, dtype=int), y_pred_test])[:len(X_test)]
    return y_pred_train_full, y_pred_test_full, model

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

def backtest_and_plot(df_feat, y_true, y_pred, name='Model', tf='1D', return_trades=False):
    params = PARAMS_MAP.get(tf, PARAMS_MAP['1D'])
    max_holding_period = params['max_holding']
    stop_loss_multi = params['stop_loss_multi']
    take_profit_multi = params['take_profit_multi']
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
    profitable_trades = []
    losing_trades = []
    for bar_idx, row in df_trades.iterrows():
        signal = int(row['Pred_shifted'])
        current_open = float(row['Open'])
        current_high = float(row['High'])
        current_low = float(row['Low'])
        
        if position != 0:
            holding = bar_idx - entry_bar if entry_bar is not None else 0
            
            atr = float(row['ATR'])
            stop_loss_price = entry_price - atr * stop_loss_multi if position == 1 else entry_price + atr * stop_loss_multi
            take_profit_price = entry_price + atr * take_profit_multi if position == 1 else entry_price - atr * take_profit_multi

            hit_sl = (position == 1 and current_low <= stop_loss_price) or \
                     (position == -1 and current_high >= stop_loss_price)
            
            hit_tp = (position == 1 and current_high >= take_profit_price) or \
                     (position == -1 and current_low <= take_profit_price)

            if hit_sl or hit_tp or holding >= max_holding_period:
                exit_price = take_profit_price if hit_tp else stop_loss_price if hit_sl else current_open
                profit = (exit_price - entry_price)*position
                
                trades.append({
                    'EntryDateTime': entry_time,
                    'ExitDateTime': row['DateTime'],
                    'Type': 'Long' if position==1 else 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Profit': profit,
                    'HoldingBars': holding
                })
                if profit > 0:
                    profitable_trades.append(profit)
                else:
                    losing_trades.append(profit)
                holding_periods.append(holding)
                if hit_sl: sl_hits += 1
                if hit_tp: tp_hits += 1
                position = 0
                entry_price = None
                entry_time = None
                entry_bar = None

        # Check for entry conditions
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
            # Check for reversal signal to close and re-enter
            if (position==1 and signal==2) or (position==-1 and signal==1):
                holding = bar_idx - entry_bar if entry_bar is not None else 0
                profit = (current_open - entry_price)*position
                trades.append({
                    'EntryDateTime': entry_time,
                    'ExitDateTime': row['DateTime'],
                    'Type': 'Long' if position==1 else 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_open,
                    'Profit': profit,
                    'HoldingBars': holding
                })
                if profit > 0:
                    profitable_trades.append(profit)
                else:
                    losing_trades.append(profit)
                holding_periods.append(holding)
                position = 1 if signal==1 else -1
                entry_price = current_open
                entry_time = row['DateTime']
                entry_bar = bar_idx
    
    # Close final open position
    if position != 0 and entry_bar is not None:
        exit_price = df_trades.iloc[-1]['Close']
        holding = len(df_trades) - entry_bar
        profit = (exit_price - entry_price)*position
        trades.append({
            'EntryDateTime': entry_time,
            'ExitDateTime': df_trades.iloc[-1]['DateTime'],
            'Type': 'Long' if position==1 else 'Short',
            'EntryPrice': entry_price,
            'ExitPrice': exit_price,
            'Profit': profit,
            'HoldingBars': holding
        })
        if profit > 0:
            profitable_trades.append(profit)
        else:
            losing_trades.append(profit)
        holding_periods.append(holding)
    
    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['Profit'].sum() if not trades_df.empty else 0
    acc = accuracy_score(df_trades['Actual'], df_trades['Pred'])
    win_trades = len(profitable_trades)
    total_trades = trades_df.shape[0]
    win_rate = (win_trades/total_trades*100) if total_trades>0 else 0
    avg_holding = np.mean(holding_periods) if holding_periods else 0
    avg_win = np.mean(profitable_trades) if profitable_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    print(f"{name} Total Profit: {total_profit:.2f}, Accuracy: {acc:.4f}, WinRate: {win_rate:.2f}%, AvgHold: {avg_holding:.2f}, SL Hits: {sl_hits}, TP Hits: {tp_hits}, Risk/Reward: {risk_reward:.2f}")
    metrics = {
        'Profit': total_profit,
        'Accuracy': acc,
        'WinRate': win_rate,
        'AvgHolding': avg_holding,
        'SL_Hits': sl_hits,
        'TP_Hits': tp_hits,
        'RiskReward': risk_reward,
        'TotalTrades': total_trades,
        'TotalCorrectPredictions': accuracy_score(df_trades['Actual'], df_trades['Pred'], normalize=False)
    }
    if return_trades:
        return trades_df, metrics
    return metrics

# ------------------ PIPELINE ------------------
def run_pipeline():
    global_trades = []
    


    # dictionary to store summaries 
    all_stock_summaries = defaultdict(lambda: defaultdict(dict))

    for tf in TIMEFRAMES:
        files = glob(f"./data/*_{tf}.csv")
        for file in files:
            file_name = file.split("\\")[-1].replace(f"_{tf}.csv", "")
            print(f"\nProcessing {file} (timeframe={tf})...")
            df = pd.read_csv(file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            X, y, df_feat = prepare_data(df, tf=tf)
            X.replace([np.inf,-np.inf], np.nan, inplace=True)
            X.fillna(method='bfill', inplace=True)
            X.fillna(method='ffill', inplace=True)
            X = X.clip(-1e10,1e10)
            split_idx = int(len(X)*0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # --- XGBoost
            y_pred_xgb_train, model_xgb = train_xgboost(X_train, y_train, X_train, y_train)
            y_pred_xgb_test, _ = train_xgboost(X_train, y_train, X_test, y_test)
            _, metrics_xgb = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_xgb_test, f"XGBoost ({tf})", tf=tf, return_trades=True)
            all_stock_summaries[file_name][tf]['XGBoost'] = metrics_xgb
            
            # --- LSTM
            y_pred_train_lstm, y_pred_test_lstm, model_lstm = train_lstm(X_train, y_train, X_test, y_test, tf=tf)
            _, metrics_lstm = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_test_lstm, f"LSTM ({tf})", tf=tf, return_trades=True)
            all_stock_summaries[file_name][tf]['LSTM'] = metrics_lstm
            
            # --- Random Forest
            y_pred_rf_train, model_rf = train_random_forest(X_train, y_train, X_train)
            y_pred_rf_test, _ = train_random_forest(X_train, y_train, X_test)
            _, metrics_rf = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_rf_test, f"RandomForest ({tf})", tf=tf, return_trades=True)
            all_stock_summaries[file_name][tf]['RandomForest'] = metrics_rf
            
            # --- Hybrid
            y_pred_test_hybrid = hybrid_prediction(y_pred_xgb_test, y_pred_test_lstm, y_pred_rf_test)
            trades_test, metrics_hybrid = backtest_and_plot(df_feat.iloc[split_idx:], y_test, y_pred_test_hybrid, f"Hybrid ({tf})", tf=tf, return_trades=True)
            all_stock_summaries[file_name][tf]['Hybrid'] = metrics_hybrid
            trades_test['Timeframe'] = tf
            global_trades.extend(trades_test.to_dict('records'))
            
    print("\n--- Summary of Results for Each Stock ---")
    for stock, timeframe_dict in all_stock_summaries.items():
        print(f"\nStock: {stock}")
        for tf, model_dict in timeframe_dict.items():
            print(f"  Timeframe: {tf}")
            for model, metrics in model_dict.items():
                print(f"    {model}: Total Profit: {metrics['Profit']:.2f}, Accuracy: {metrics['Accuracy']:.4f}, WinRate: {metrics['WinRate']:.2f}%, AvgHold: {metrics['AvgHolding']:.2f}, SL Hits: {metrics['SL_Hits']}, TP Hits: {metrics['TP_Hits']}, Risk/Reward: {metrics['RiskReward']:.2f}")

    tradebook_df = pd.DataFrame(global_trades)
    tradebook_df.to_csv("full_tradebook_review.csv", index=False)
    print("\nFull tradebook saved to 'full_tradebook_review.csv'")
    return tradebook_df, all_stock_summaries

if __name__=="__main__":
    run_pipeline()