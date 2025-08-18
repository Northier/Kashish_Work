
####################################### ACCURACY 0.773726 and Signal accuracy: 29.84% ##########################################################################

import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

timeframe = ['5min', '15min', '30min', '1hr', '1D']

df_list = []
for tf in timeframe:
    print(f"\nProcessing {tf} Data")
    files = glob(f"resampled_{tf}_NSE_*.csv")
    if not files:
        print("No Files Found")
        continue
    for file in files:
        temp_data = pd.read_csv(file, parse_dates=['DateTime'])
        temp_data.sort_values('DateTime', inplace=True)
        df_list.append(temp_data)

data = pd.concat(df_list).reset_index(drop=True)
print("Sample data:\n", data.head())

data['future_pct_change'] = data['Close'].shift(-1) / data['Close'] - 1

up_th = data['future_pct_change'].quantile(0.66)
down_th = data['future_pct_change'].quantile(0.33)

rolling_window = 50
data['Volatility'] = data['future_pct_change'].rolling(rolling_window).std()

# Trend labeling
def label_trend(x, vol_th, up_th, down_th):
    thr = max(vol_th, (up_th - down_th)/2)
    if x > thr:
        return "Buy"
    elif x < -thr:
        return "Sell"
    else:
        return "Hold"

data['Trend'] = data.apply(lambda row: label_trend(row['future_pct_change'], row['Volatility'], up_th, down_th), axis=1)

# Feature selection
selected_features = [
    'Close', 'Close_lag_1', 'Close_lag_3', 'Price_diff', 'log_return', 'ROC',
    'Volume', 'Volume_diff', 'OBV',
    'RSI', 'RSI_diff',
    'EMA', 'EMA_diff',
    'ADX_14', 'DMP_14', 'DMN_14',
    'ATR', 'ATR_diff', 'rolling_volatility_10',
    'BBP_20_2.0', 'bb_width_pct',
    'MACD', 'MACD_signal',
    'StochK', 'StochD',
    'CCI'
]

# Drop rows with NaNs
features = data[selected_features].dropna()
target = data['Trend'].loc[features.index] 

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target)

split = int(len(features) * 0.9)
X_train, X_test = features[:split], features[split:]
y_train, y_test = y[:split], y[split:]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, 'scaler.pkl')

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
from sklearn.utils import class_weight
import numpy as np

weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

xgb_model.fit(X_train_scaled, y_train, sample_weight=weights)

# Predict
y_pred = xgb_model.predict(X_test_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\nModel Training Completed using XGBoost")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Signal accuracy
def calculate_signal_accuracy(pred_labels, actual_data, threshold=0.001):
    correct = 0
    future_change = actual_data['Close'].shift(-1).values / actual_data['Close'].values - 1
    for i, pred in enumerate(pred_labels):
        if i >= len(future_change)-1:
            continue
        change = future_change[i]
        if (pred == "Buy" and change > 0) or \
           (pred == "Sell" and change < 0) or \
           (pred == "Hold" and abs(change) <= threshold):
            correct += 1
    return correct / len(pred_labels)

signal_acc = calculate_signal_accuracy(y_pred_labels, data.iloc[split:])
print("XGBoost Signal Accuracy:", round(signal_acc*100, 2), "%")

joblib.dump(xgb_model, 'xgb_model.pkl')
