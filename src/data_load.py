# src/data_loader.py
import pandas as pd
import os

DATA_PATH = './data'  # adjust if needed

def load_data(stock: str, timeframe: str) -> pd.DataFrame:
    """
    Loads CSV with index column 'DateTime' (parse as datetime).
    Expects files named like: ABB_5Min.csv, INDHOTEL_1D.csv etc.
    """
    filename = f"{stock}_{timeframe}.csv"
    fp = os.path.join(DATA_PATH, filename)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing data file: {fp}")
    df = pd.read_csv(fp, parse_dates=["DateTime"], index_col="DateTime")
    df = df.sort_index()
    return df
