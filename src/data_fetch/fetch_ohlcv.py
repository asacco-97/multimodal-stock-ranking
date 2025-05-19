# src/data_fetch/fetch_ohlcv.py
import yfinance as yf
import pandas as pd
import os
import time

def fetch_ohlcv(tickers, start_date, end_date, save_dir="data/raw/ohlcv/"):
    os.makedirs(save_dir, exist_ok=True)
    
    for ticker in tickers:
        print(f"Fetching OHLCV for {ticker}")
        ticker_object = yf.Ticker("AAPL")
        df = ticker_object.history(start=start_date, end=end_date, auto_adjust=False)
        df.reset_index(inplace=True)
        
        # Convert all OHLCV columns to float
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # Save cleaned file
        save_path = os.path.join(save_dir, f"{ticker}_ohlcv.csv")
        df.to_csv(save_path, index=False)

        # Add delay to avoid rate limite
        time.sleep(3)  
        
    print("Finished fetching OHLCV data.")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "JPM", "UNH", "XOM", "HD", "NVDA", "KO"]
    start_date = "2024-05-01"
    end_date = "2025-05-01"
    fetch_ohlcv(tickers, start_date, end_date)
