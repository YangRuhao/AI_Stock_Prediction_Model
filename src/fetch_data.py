# src/fetch_data.py
import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2025-01-01"):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{ticker}_stock.csv")

    print(f"⬇️  Fetching stock data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, group_by="ticker")

    # Reset index so Date is a column
    df.reset_index(inplace=True)

    # 🔑 Handle multi-index (columns like ('Close','AAPL'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # ✅ If columns are still just ticker names (like "AAPL"), rename them manually
    if set(df.columns[1:]) == {ticker}:
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # ✅ If Adj Close exists, rename it to Close
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # Keep only the expected columns if they exist
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected_cols if c in df.columns]]

    df.to_csv(file_path, index=False)
    print(f"✅ Data saved to {file_path}")
    print(df.head())

    return df


if __name__ == "__main__":
    fetch_stock_data()
