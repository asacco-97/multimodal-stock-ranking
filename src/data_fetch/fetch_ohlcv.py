# src/data_fetch/fetch_ohlcv.py
import yfinance as yf
import pandas as pd
import os

def fetch_ohlcv(tickers, start_date, end_date, save_dir="data/raw/ohlcv/"):
    os.makedirs(save_dir, exist_ok=True)
    
    for ticker in tickers:
        print(f"Fetching OHLCV for {ticker}")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df.reset_index(inplace=True)
        save_path = os.path.join(save_dir, f"{ticker}_ohlcv.csv")
        df.to_csv(save_path, index=False)
    print("Finished fetching OHLCV data.")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT"]
    start_date = "2024-01-01"
    end_date = "2024-03-31"
    fetch_ohlcv(tickers, start_date, end_date)
