import pandas as pd
from sklearn.preprocessing import MinMaxScalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from datetime import datetime
from glob import glob

timeframe = ['5min', '15min', '30min', '1hr', '1D']

for tf in timeframe:
    print(f"\n Processing {tf} Data")
    files = glob(f"resampled_{tf}_NSE_*.csv")

    if not files:
        print("No Files Found")
        continue

    df_list = []

    for file in files:
        temp_data = pd.read_csv('file', parse_dates=['DateTime'])
        temp_data.sort_values('DateTime', inplace=True)
        df_list.append(temp_data)

    data = pd.concat(df_list).reset_index(drop=True)

    print(data.head(5))

data['future_pct_change'] = data['Close'].shift(-1) / data['Close'] - 1

up_th =  data['future_pct_change'].quantile(0.66)
down_th = data['future_pct_change'].quantile(0.33)

rolling_window = 50
data['Volatility']= data['future_pct_change'].rolling(rolling_window).std()

def label_trend(x,up_th, down_th, vol_th):
    thr = max(vol_th,(up_th - down_th)/2)

    if x > thr:
        return "Buy"
    elif x < -thr:
        return "Sell"
    else:
        return "Hold"
    
data['Trend'] = data.apply(
    lambda row: label_trend(row['future_pct_change'], row['Volatility'], up_th, down_th),
    axis = 1
)

