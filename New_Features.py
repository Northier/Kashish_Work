import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('resampled_1D_NSE_ABB.csv')

# Adding Indicators 
data["RSI"] = data.ta.rsi(close='Close', length=14)

BollingerBands = data.ta.bbands(close='Close', length=20,std=2)
data = data.join(BollingerBands)

data["EMA"] = data.ta.ema(close='Close', length=14)

Adx = data.ta.adx(close='Close',high='High',low='Low',length=14)
data = data.join(Adx)

data['ATR'] = data.ta.atr(high='High', low='Low', close='Close', length=14)

# # View the data
print(data.tail(10)) 


# DIFFERENTIAL FEATURES
data['RSI_diff'] = data['RSI'].diff()
data['EMA_diff'] = data['EMA'].diff()
data['ATR_diff'] = data['ATR'].diff()
data['Price_diff'] = data['Close'].diff()
data['Volume_diff'] = data['Volume'].diff()

# % CHANGES
data['vol_pct_change'] = data['Volume'].pct_change() * 100
data['close_pct_change'] = data['Close'].pct_change() * 100
data['rsi_pct_change'] = data['RSI'].pct_change() * 100

data['ROC'] = ta.roc(data['Close'], length=5)
data["bb_width_pct"] = (data["BBU_20_2.0"] - data["BBL_20_2.0"]) / data["BBM_20_2.0"]

macd = ta.macd(data['Close'])
data['MACD'] = macd['MACD_12_26_9']
data['MACD_signal'] = macd['MACDs_12_26_9']

stoch = ta.stoch(data['High'], data['Low'], data['Close'])
data['StochK'] = stoch['STOCHk_14_3_3']
data['StochD'] = stoch['STOCHd_14_3_3']

data['CCI'] = ta.cci(data['High'], data['Low'], data['Close'], length=20)
data['OBV'] = ta.obv(data['Close'], data['Volume'])

# LAGGED VALUES (1 and 3 periods back)
data['Close_lag_1'] = data['Close'].shift(1)
data['Close_lag_3'] = data['Close'].shift(3)
data['RSI_lag_1'] = data['RSI'].shift(1)
data['RSI_lag_3'] = data['RSI'].shift(3)

# ROLLING FEATURES
data['roll_mean_close_5'] = data['Close'].rolling(window=5).mean()
data['roll_std_close_5'] = data['Close'].rolling(window=5).std()
data['roll_max_close_5'] = data['Close'].rolling(window=5).max()
data['roll_min_close_5'] = data['Close'].rolling(window=5).min()

# RSI × VOLUME (momentum with participation)
data['RSI_x_Volume'] = data['RSI'] * data['Volume']

# ADX × RANGE (trend strength × candle size)
adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
data['ADX'] = adx['ADX_14']
data['True_Range'] = ta.true_range(data['High'], data['Low'], data['Close'])
data['ADX_x_Range'] = data['ADX'] * data['True_Range']

# ROLLING VOLATILITY (% based)
data['rolling_volatility_10'] = data['close_pct_change'].rolling(window=10).std()

# LOG RETURNS
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# NORMALIZATION
def Normalize(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())

data['atr_norm'] = Normalize(data["ATR"])
data['vol_norm'] = Normalize(data['Volume_diff'])
data['roc_norm'] = Normalize(data['ROC'])

# MARKET HEAT INDEX (MHI)
data['MHI'] = (
    (data['atr_norm'] * 0.5) +
    (data['vol_norm'] * 0.3) +
    (data['roc_norm'] * 0.2)
) * 100

def market_state(score):
    if score <= 30:
        return "Cold"
    elif score <= 70:
        return "Normal"
    else:
        return "Hot"

data['Market_state'] = data['MHI'].apply(market_state)

print(data[["DateTime", "MHI", "Market_state"]].head(10))
data.to_csv("resampled_1D_NSE_ABB.csv", index=False)
