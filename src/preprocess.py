# src/preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.fetch_data import fetch_stock_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_data(ticker="AAPL", start="2020-01-01", end="2025-01-01"):
    """
    Load stock data from CSV if exists, otherwise fetch fresh data.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{ticker}_stock.csv")

    if os.path.exists(file_path):
        print(f"📂 Loading cached data from {file_path}")
        df = pd.read_csv(file_path)
    else:
        df = fetch_stock_data(ticker, start, end)

    # Ensure 'Close' column exists
    close_candidates = [col for col in df.columns if "Close" in col]
    if not close_candidates:
        raise ValueError(f"❌ No 'Close' column found in {file_path}")
    
    close_col = close_candidates[0]
    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["Close"])

    return df

def scale_data(df):
    """
    Scale Close prices between 0 and 1.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["Close"]].values)
    return scaled_data, scaler

def create_sequences(data, time_step=60):
    """
    Turn a time series into (X, y) sequences for LSTM.
    """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
