"""
STOCK MARKET TREND PREDICTION
XGBoost vs LSTM vs Hybrid
Includes: feature engineering, LSTM sequences, hybrid voting, backtest & plots
"""

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

PRED_PERIOD = 7
TIMEFRAMES = ['5min', '15min', '30min', '1hr', '1D']
CLASSES = ['Sideways', 'Up', 'Down']  # Trend classes
LSTM_EPOCHS = 15
LSTM_BATCH = 64


def safe_div(a,b): return a / b.replace(0,np.nan)

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

def train_lstm(X_train, y_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape sequences
    timesteps = PRED_PERIOD
    def create_sequences(X, y):
        Xs, ys = [], []
        for i in range(timesteps, len(X)):
            Xs.append(X[i-timesteps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_train.values[:len(X_test_scaled)])
    
    # LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(timesteps, X_train_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_seq, y_train_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              validation_split=0.1, verbose=0, callbacks=[EarlyStopping(patience=5)])
    
    y_pred_prob = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Pad to match length
    y_pred_full = np.concatenate([np.full(timesteps,0), y_pred])
    return y_pred_full, model

def hybrid_prediction(y_pred_xgb, y_pred_lstm):
    hybrid_pred = []
    for x,l in zip(y_pred_xgb, y_pred_lstm):
        counts = np.bincount([x,l])
        hybrid_pred.append(np.argmax(counts))
    return np.array(hybrid_pred)

def backtest_and_plot(df_feat, y_true, y_pred, name='Model', global_trades=None):
    
    df_trades = df_feat.iloc[y_true.index].copy()
    df_trades['Pred'] = y_pred
    df_trades['Actual'] = y_true

    trades = []
    position = 0
    entry_price = 0

    for i, row in df_trades.iterrows():
        if position == 0:
            if row['Pred'] == 1:
                position = 1
                entry_price = row['Close']
            elif row['Pred'] == 2:
                position = -1
                entry_price = row['Close']
            entry_time = row['DateTime']
        else:
            if (position==1 and row['Pred']==2) or (position==-1 and row['Pred']==1):
                exit_price = row['Close']
                pnl = (exit_price - entry_price) * position
                trades.append({'EntryDateTime': entry_time,
                               'ExitDateTime': row['DateTime'],
                               'Type': 'Long' if position==1 else 'Short',
                               'EntryPrice': entry_price,
                               'ExitPrice': exit_price,
                               'Profit': pnl})
                position = 1 if row['Pred']==1 else -1
                entry_price = row['Close']
            entry_time = row['DateTime']

    if position != 0:
        exit_price = df_trades.iloc[-1]['Close']
        pnl = (exit_price - entry_price) * position
        trades.append({'EntryDateTime': entry_time,
                       'ExitDateTime': df_trades.iloc[-1]['DateTime'],
                       'Type': 'Long' if position==1 else 'Short',
                       'EntryPrice': entry_price,
                       'ExitPrice': exit_price,
                       'Profit': pnl})

    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['Profit'].sum()
    print(f"{name} Total Profit/Loss (this timeframe): {total_profit}")

    if global_trades is not None:
        global_trades.extend(trades)

    # Plot cumulative PnL
    if not trades_df.empty:
        cum_profit = trades_df['Profit'].cumsum()
        plt.figure(figsize=(12,4))
        plt.plot(trades_df['ExitDateTime'], cum_profit, label='Cumulative PnL')
        plt.xlabel('DateTime'); plt.ylabel('Profit')
        plt.title(f'{name} — Cumulative PnL')
        plt.legend(); plt.grid(True)
        plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_row = cm.astype(float)/cm.sum(axis=1)[:,None]
    cm_row = np.nan_to_num(cm_row)

    plt.figure(figsize=(6,5))
    plt.imshow(cm_row, aspect='auto', cmap='Blues')
    plt.title(f'{name} — Confusion Matrix (row %)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.xticks(range(len(CLASSES)), CLASSES)
    plt.yticks(range(len(CLASSES)), CLASSES)
    thresh = cm_row.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,f"{cm[i,j]}\n({cm_row[i,j]*100:.1f}%)", ha='center',
                     color='white' if cm_row[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.show()

    # PRF
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)))
    x = np.arange(len(CLASSES))
    width = 0.22
    plt.figure(figsize=(8,4))
    plt.bar(x - width, precisions, width=width, label='Precision')
    plt.bar(x, recalls, width=width, label='Recall')
    plt.bar(x + width, f1s, width=width, label='F1-score')
    plt.xticks(x, CLASSES)
    plt.ylim(0,1.02)
    plt.title(f'{name} — Precision/Recall/F1')
    plt.legend()
    for i in x:
        plt.text(i - width, precisions[i]+0.02, f"{precisions[i]:.2f}", ha='center')
        plt.text(i, recalls[i]+0.02, f"{recalls[i]:.2f}", ha='center')
        plt.text(i + width, f1s[i]+0.02, f"{f1s[i]:.2f}", ha='center')
    plt.tight_layout()
    plt.show()

    # Actual vs Predicted trend
    df_plot = df_feat.iloc[y_true.index]
    plt.figure(figsize=(15,4))
    plt.plot(df_plot['DateTime'], y_true, label='Actual', alpha=0.7)
    plt.plot(df_plot['DateTime'], y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('DateTime'); plt.ylabel('Trend')
    plt.title(f'{name} — Actual vs Predicted Trend')
    plt.legend()
    plt.show()

    print(f"{name} Overall accuracy: {accuracy_score(y_true, y_pred)}\n")
    print(f"{name} Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))


def run_pipeline():
    global_trades = []  

    for tf in TIMEFRAMES:
        files = glob(f"resampled_{tf}_NSE*.csv")
        for file in files:
            print(f"\nProcessing {file}...")
            df = pd.read_csv(file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            X, y, df_feat = prepare_data(df)
            
            X.replace([np.inf,-np.inf], np.nan, inplace=True)
            X.fillna(method='bfill', inplace=True)
            X.fillna(method='ffill', inplace=True)
            X = X.clip(-1e10,1e10)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # --- XGBoost
            y_pred_xgb, model_xgb = train_xgboost(X_train, y_train, X_test)
            backtest_and_plot(df_feat, y_test, y_pred_xgb, 'XGBoost', global_trades)
            plot_importance(model_xgb, max_num_features=20); plt.show()
            
            # --- LSTM
            y_pred_lstm, model_lstm = train_lstm(X_train, y_train, X_test)
            backtest_and_plot(df_feat, y_test, y_pred_lstm, 'LSTM', global_trades)
            
            # --- Hybrid
            y_pred_hybrid = hybrid_prediction(y_pred_xgb, y_pred_lstm)
            backtest_and_plot(df_feat, y_test, y_pred_hybrid, 'Hybrid', global_trades)
    
    # Combined PnL across all timeframes
    if global_trades:
        combined_profit = sum([t['Profit'] for t in global_trades])
        print(f"\nCombined Total Profit/Loss across all timeframes: {combined_profit}")

if __name__=="__main__":
    run_pipeline()

