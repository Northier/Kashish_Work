import pandas as pd
import os


timeframes = [ "5Min", "15Min", "30Min", "1H", "1D"]
stocks = ["ABB", "INDHOTEL"]
path = "./data/"

for stock in stocks:
    data = pd.read_csv(f"{path}{stock}.csv")
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.set_index("DateTime")

    start_time = pd.to_datetime("09:15:00").time()
    end_time = pd.to_datetime("15:29:00").time()
    data = data[(data.index.time >= start_time) & (data.index.time <= end_time)]

    for tf in timeframes:
        freq = tf
        offset = '15min' if tf in ['1H', '30Min', '15Min', '5Min'] else '0min'
        # Apply the resampling

        resample_data = data.groupby(pd.Grouper(freq=freq, offset=offset)).agg({"Symbol":"first",
                                                                                  "Open": "first",
                                                                                  "High": "max",
                                                                                  "Low": "min",
                                                                                  "Close": "last", 
                                                                                  "Volume": "sum"}).reset_index()
        
        resample_data.columns = ["DateTime", "Symbol", "Open", "High", "Low", "Close", "Volume"]
        resample_data = resample_data[["Symbol", "DateTime", "Open", "High", "Low", "Close", "Volume"]]
        resample_data = resample_data.dropna()
        resample_data.to_csv(f"{path}{stock}_{tf}.csv", index=False)

        print(resample_data)
