
####################################### ACCURACY 0.58151616 and Signal accuracy: 29.84% ##########################################################################

# import pandas as pd
# from glob import glob
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from xgboost import XGBClassifier
# import joblib

# timeframe = ['5min', '15min', '30min', '1hr', '1D']

# data_list = []
# for tf in timeframe:
#     print(f"\nProcessing {tf} Data")
#     files = glob(f"resampled_{tf}_NSE_*.csv")
#     if not files:
#         print("No Files Found")
#         continue
#     for file in files:
#         temp_data = pd.read_csv(file, parse_dates=['DateTime'])
#         temp_data.sort_values('DateTime', inplace=True)
#         data_list.append(temp_data)

# data = pd.concat(data_list).reset_index(drop=True)
# print("Sample data:\n", data.head())

# data['future_pct_change'] = data['Close'].shift(-1) / data['Close'] - 1

# up_th = data['future_pct_change'].quantile(0.66)
# down_th = data['future_pct_change'].quantile(0.33)

# rolling_window = 50
# data['Volatility'] = data['future_pct_change'].rolling(rolling_window).std()

# # Trend labeling
# def label_trend(x, vol_th, up_th, down_th):
#     thr = max(vol_th, (up_th - down_th)/2)
#     if x > thr:
#         return "Buy"
#     elif x < -thr:
#         return "Sell"
#     else:
#         return "Hold"

# data['Trend'] = data.apply(lambda row: label_trend(row['future_pct_change'], row['Volatility'], up_th, down_th), axis=1)

# # Feature selection
# selected_features = [
#     'Close', 'Close_lag_1', 'Close_lag_3', 'Price_diff', 'log_return', 'ROC',
#     'Volume', 'Volume_diff', 'OBV',
#     'RSI', 'RSI_diff',
#     'EMA', 'EMA_diff',
#     'ADX_14', 'DMP_14', 'DMN_14',
#     'ATR', 'ATR_diff', 'rolling_volatility_10',
#     'BBP_20_2.0', 'bb_width_pct',
#     'MACD', 'MACD_signal',
#     'StochK', 'StochD',
#     'CCI'
# ]

# # Drop rows with NaNs
# features = data[selected_features].dropna()
# target = data['Trend'].loc[features.index] 

# # Encode target labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(target)

# split = int(len(features) * 0.9)
# X_train, X_test = features[:split], features[split:]
# y_train, y_test = y[:split], y[split:]

# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# joblib.dump(scaler, 'scaler.pkl')

# # Train XGBoost
# xgb_model = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='mlogloss'
# )
# from sklearn.utils import class_weight
# import numpy as np

# weights = class_weight.compute_sample_weight(
#     class_weight='balanced',
#     y=y_train
# )

# xgb_model.fit(X_train_scaled, y_train, sample_weight=weights)

# # Predict
# y_pred = xgb_model.predict(X_test_scaled)
# y_pred_labels = label_encoder.inverse_transform(y_pred)

# print("\nModel Training Completed using XGBoost")
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Signal accuracy
# def calculate_signal_accuracy(pred_labels, actual_data, threshold=0.001):
#     correct = 0
#     future_change = actual_data['Close'].shift(-1).values / actual_data['Close'].values - 1
#     for i, pred in enumerate(pred_labels):
#         if i >= len(future_change)-1:
#             continue
#         change = future_change[i]
#         if (pred == "Buy" and change > 0) or \
#            (pred == "Sell" and change < 0) or \
#            (pred == "Hold" and abs(change) <= threshold):
#             correct += 1
#     return correct / len(pred_labels)

# signal_acc = calculate_signal_accuracy(y_pred_labels, data.iloc[split:])
# print("XGBoost Signal Accuracy:", round(signal_acc*100, 2), "%")

# joblib.dump(xgb_model, 'xgb_model.pkl')


# import matplotlib.pyplot as plt
# import joblib

# # Reload 1D dataset
# data_1d = pd.read_csv('resampled_1D_NSE_ABB.csv', parse_dates=['DateTime'])
# data_1d.sort_values("DateTime", inplace=True)

# # Recreate labels (same as training)
# data_1d['future_pct_change'] = data_1d['Close'].shift(-1) / data_1d['Close'] - 1

# # Thresholds (same as training step)
# up_th = data_1d['future_pct_change'].quantile(0.66)
# down_th = data_1d['future_pct_change'].quantile(0.33)
# rolling_window = 50
# data_1d['Volatility'] = data_1d['future_pct_change'].rolling(rolling_window).std()

# def label_trend(x, vol_th, up_th, down_th):
#     thr = max(vol_th, (up_th - down_th)/2)
#     if x > thr:
#         return "Buy"
#     elif x < -thr:
#         return "Sell"
#     else:
#         return "Hold"

# data_1d['Actual'] = data_1d.apply(lambda row: label_trend(row['future_pct_change'], row['Volatility'], up_th, down_th), axis=1)

# # Prepare features
# features_1d = data_1d[selected_features].dropna()

# # Scale features
# scaler = joblib.load("scaler.pkl")
# X_1d_scaled = scaler.transform(features_1d)

# # Predict using trained model
# xgb_model = joblib.load("xgb_model.pkl")
# y_pred_1d = xgb_model.predict(X_1d_scaled)
# y_pred_labels_1d = label_encoder.inverse_transform(y_pred_1d)

# # Align lengths
# data_1d = data_1d.iloc[-len(y_pred_labels_1d):].copy()
# data_1d["Predicted"] = y_pred_labels_1d

# # ---------------------- PLOT ---------------------- #
# plt.figure(figsize=(15,7))

# # Plot Actual signals
# plt.plot(data_1d["DateTime"], data_1d["Actual"], label="Actual Trend", color="blue", linewidth=2)

# # Plot Predicted signals
# plt.plot(data_1d["DateTime"], data_1d["Predicted"], label="Predicted Trend", color="red", linestyle="--", linewidth=2)

# plt.title("Comparison of Actual vs Predicted Signals (Buy/Sell/Hold)")
# plt.xlabel("Date")
# plt.ylabel("Signal")
# plt.legend()
# plt.grid(True)
# plt.show()

# --------------------------
# Enhanced XGBoost Pipeline with SMOTE and Adaptive Thresholds
# --------------------------


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



    