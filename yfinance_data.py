import yfinance as yf
import pandas as pd
import csv
from datetime import datetime
import os
import time

symbols = {
    "TATASTEEL.NS": "NSE:TATASTEEL",
    "JSWSTEEL.NS": "NSE:JSWSTEEL"
}
start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
interval = "1d"

os.makedirs("data", exist_ok=True)

for yf_symbol, export_symbol in symbols.items():
    print(f"Fetching data for {export_symbol}...")
    attempts = 3
    for attempt in range(attempts):
        try:
            df = yf.download(yf_symbol, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
            if df.empty:
                raise ValueError("No data returned")
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        print(f"Skipping {export_symbol} due to repeated download failure.")
        continue

    df.reset_index(inplace=True)

    open_col = df['Open'].squeeze()
    high_col = df['High'].squeeze()
    low_col = df['Low'].squeeze()
    close_col = df['Close'].squeeze()
    volume_col = df['Volume'].astype(int).squeeze()

    df["DateTime"] = df["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")

    formatted = pd.DataFrame({
        "Symbol": [export_symbol] * len(df),
        "DateTime": df["DateTime"],
        "Open": open_col.round(2),
        "High": high_col.round(2),
        "Low": low_col.round(2),
        "Close": close_col.round(2),
        "Volume": volume_col
    })

    filename = os.path.join("data", f"{export_symbol.replace(':','_')}_{interval}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Symbol","DateTime","Open","High","Low","Close","Volume"])
        for _, row in formatted.iterrows():
            writer.writerow([
                row["Symbol"],
                row["DateTime"],
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"]
            ])

    print(f"Saved: {filename}")
