import pandas as pd

data = pd.read_csv("questdb-query-1754912225193.csv")


data['DateTime'] = pd.to_datetime(data['DateTime'])

#make the time the index. 
data = data.set_index("DateTime")

# Extract time range
start_time = pd.to_datetime("09:15:00").time()
end_time = pd.to_datetime("15:29:00").time()

data = data[(data.index.time >= start_time) & (data.index.time <= end_time)]


resample_data_30min = data.groupby(pd.Grouper(freq='1H',offset = '15min')).agg({"Symbol":"first",
                                                                "Open": "first",
                                                                "High": "max",
                                                                "Low": "min",
                                                                "Close": "last", 
                                                                "Volume": "sum"}).reset_index()

resample_data_30min.columns = ["DateTime", "Symbol", "Open", "High", "Low", "Close", "Volume"]

# Now reorder to match your desired format: Symbol first
resample_data_30min = resample_data_30min[["Symbol", "DateTime", "Open", "High", "Low", "Close", "Volume"]]

resample_data_30min = resample_data_30min.dropna()

print(resample_data_30min)
   

resample_data_30min.to_csv('resampled_1hr_NSE_ABB.csv',index = False)


