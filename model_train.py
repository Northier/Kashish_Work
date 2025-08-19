
import os
from glob import glob
from xml.parsers.expat import model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

TIMEFRAMES = ['5min', '15min', '30min', '1hr', '1D']
BASE_FEATURES = [
    'Open','High','Low','Close','Volume',
    'RSI','MACD','StochK','StochD','CCI',
    'EMA','ADX','ATR','rolling_volatility_10',
    'BBL_20_2.0','BBM_20_2.0','BBU_20_2.0',
    'log_return'
]
MODEL_PATH = "xgb_stock_trend.json"
SCALER_PATH = "xgb_scaler.gz"
TEST_SIZE = 0.10

def load_all_timeframes(timeframes=TIMEFRAMES):
    data_list = []
    for tf in timeframes:
        print(f"Processing {tf} Data")
        files = glob(f"resampled_{tf}_NSE_*.csv")
        if not files:
            print("  No files found for", tf)
            continue
        for file in files:
            df = pd.read_csv(file, parse_dates=['DateTime'])
            df = df.sort_values('DateTime').reset_index(drop=True)
            data_list.append(df)
    if not data_list:
        raise RuntimeError("No data files found. Make sure files match pattern.")
    data = pd.concat(data_list, ignore_index=True)
    data = data.sort_values('DateTime').reset_index(drop=True)
    print("Final combined shape:", data.shape)
    return data

# --------------------------
# 2) Feature Engineering
# --------------------------
def add_features(df):
    df = df.copy()

    # Lags
    for lag in [1, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    # Rolling statistics
    for win in [5, 10, 20]:
        df[f'rolling_mean_{win}'] = df['Close'].rolling(win).mean()
        df[f'rolling_std_{win}'] = df['Close'].rolling(win).std()

    df.fillna(method="bfill", inplace=True)
    return df

from imblearn.over_sampling import SMOTE

def preprocess(data, up_th=0.003, down_th=-0.003):
    data = add_features(data)

    # Target labels
    data['Return'] = data['Close'].pct_change()
    data['Trend'] = data['Return'].apply(lambda x: 1 if x > up_th else (2 if x < down_th else 0))
    data.dropna(inplace=True)

    all_features = BASE_FEATURES + \
                   [f'Close_lag_{lag}' for lag in [1, 3, 5, 10]] + \
                   [f'rolling_mean_{win}' for win in [5, 10, 20]] + \
                   [f'rolling_std_{win}' for win in [5, 10, 20]]

    missing_cols = [c for c in all_features if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X = data[all_features].copy()
    y = data['Trend'].astype(int).copy()

    return data, X, y, all_features

def signal_accuracy_percentage(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    return (correct / total) * 100

def train_pipeline():
    data = load_all_timeframes()
    data, X, y, features = preprocess(data)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, shuffle=False)

    sm = SMOTE(sampling_strategy={0: max(y_train.value_counts()[1:])}, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    classes = np.unique(y_train_res)
    cw = class_weight.compute_class_weight('balanced', classes=classes, y=y_train_res)
    class_weights = {int(c): w for c, w in zip(classes, cw)}
    sample_weights = np.array([class_weights[yy] for yy in y_train_res])

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_res, y_train_res,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        eval_metric="mlogloss",
        verbose=50,
        early_stopping_rounds=30
    )

    model.save_model(MODEL_PATH)
    print("✅ Model saved to:", MODEL_PATH)

    y_pred = model.predict(X_test)

    sig_acc = signal_accuracy_percentage(y_test, y_pred)
    print(f"Signal Accuracy: {sig_acc:.2f}%")

    acc = accuracy_score(y_test, y_pred)
    print("\n📊 Accuracy:", acc)
    print("\n📋 Classification Report:\n", classification_report(
        y_test, y_pred, target_names=["Hold(0)", "Buy(1)", "Sell(2)"]
    ))

    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Hold(0)", "Buy(1)", "Sell(2)"],
                yticklabels=["Hold(0)", "Buy(1)", "Sell(2)"])
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model, scaler, features, data


if __name__ == "__main__":
    model, scaler, features, combined_data = train_pipeline()  


