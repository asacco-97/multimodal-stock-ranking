# src/data_fetch/fetch_ohlcv.py
"""
Enhanced module to fetch OHLCV (Open, High, Low, Close, Volume) data for equities.
Supports batch processing, error handling, and resume capability for large datasets.
"""
import yfinance as yf
import pandas as pd
import os
import json
from typing import List, Optional
from time import sleep
from datetime import datetime

def fetch_ohlcv_single(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

        if df.empty:
            print(f"No data returned for {ticker}")
            return None

        df.reset_index(inplace=True)

        # If the columns are a MultiIndex, flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Drop rows where "Close" column contains non-numeric values
        df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]

        if df.empty:
            print(f"No valid data after cleaning for {ticker}")
            return None

        # Ensure we have all required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns for {ticker}")
            return None

        # Convert all OHLCV columns to float
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # Add ticker column
        df['ticker'] = ticker

        # Rename Date to date for consistency
        df.rename(columns={'Date': 'date'}, inplace=True)

        # Reorder columns
        df = df[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        return df

    except Exception as e:
        print(f"Error fetching OHLCV for {ticker}: {e}")
        return None

def fetch_ohlcv_batch(tickers: List[str], start_date: str, end_date: str,
                     save_dir: str = "data/raw/ohlcv/",
                     batch_size: int = 100,
                     resume: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV data for a batch of tickers with progress tracking and resume capability.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_dir: Directory to save individual ticker files
        batch_size: Save progress every N tickers
        resume: If True, skip already downloaded tickers

    Returns:
        Combined DataFrame with all OHLCV data
    """
    os.makedirs(save_dir, exist_ok=True)

    # Track progress
    progress_file = os.path.join(save_dir, 'download_progress.json')
    failed_file = os.path.join(save_dir, 'failed_tickers.json')

    # Load previous progress
    completed_tickers = set()
    failed_tickers = []

    if resume and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            completed_tickers = set(json.load(f))
        print(f"Resuming: {len(completed_tickers)} tickers already downloaded")

    if resume and os.path.exists(failed_file):
        with open(failed_file, 'r') as f:
            failed_tickers = json.load(f)

    # Filter out completed tickers
    remaining_tickers = [t for t in tickers if t not in completed_tickers]

    print(f"Fetching OHLCV data for {len(remaining_tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")

    all_data = []
    newly_completed = []

    for i, ticker in enumerate(remaining_tickers):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(remaining_tickers)} ({len(completed_tickers) + i + 1}/{len(tickers)} total)")

        # Fetch data
        df = fetch_ohlcv_single(ticker, start_date, end_date)

        if df is not None:
            # Save individual file
            save_path = os.path.join(save_dir, f"{ticker}_ohlcv.csv")
            df.to_csv(save_path, index=False)

            all_data.append(df)
            newly_completed.append(ticker)
        else:
            if ticker not in failed_tickers:
                failed_tickers.append(ticker)

        # Save progress periodically
        if (i + 1) % batch_size == 0:
            completed_tickers.update(newly_completed)
            with open(progress_file, 'w') as f:
                json.dump(list(completed_tickers), f)
            with open(failed_file, 'w') as f:
                json.dump(failed_tickers, f)
            print(f"Progress saved: {len(completed_tickers)} completed, {len(failed_tickers)} failed")

        # Rate limiting
        sleep(0.1)

    # Final save
    completed_tickers.update(newly_completed)
    with open(progress_file, 'w') as f:
        json.dump(list(completed_tickers), f)
    with open(failed_file, 'w') as f:
        json.dump(failed_tickers, f)

    print(f"\n=== Summary ===")
    print(f"Successfully fetched: {len(completed_tickers)} tickers")
    print(f"Failed: {len(failed_tickers)} tickers")

    if failed_tickers:
        print(f"Failed tickers saved to: {failed_file}")

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(save_dir, 'all_ohlcv_combined.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined data saved to: {combined_path}")
        return combined_df
    else:
        return pd.DataFrame()

def load_ohlcv_combined(save_dir: str = "data/raw/ohlcv/") -> pd.DataFrame:
    """
    Load previously downloaded OHLCV data.

    Args:
        save_dir: Directory where OHLCV data is saved

    Returns:
        Combined DataFrame with all OHLCV data
    """
    combined_path = os.path.join(save_dir, 'all_ohlcv_combined.csv')

    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} rows for {df['ticker'].nunique()} tickers")
        return df
    else:
        print(f"No combined file found at {combined_path}")
        return pd.DataFrame()

# Keep the old function for backwards compatibility
def fetch_ohlcv(tickers, start_date, end_date, save_dir="data/raw/ohlcv/"):
    """Legacy function - calls the enhanced batch version"""
    return fetch_ohlcv_batch(tickers, start_date, end_date, save_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch OHLCV data for equities")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers OR path to JSON file")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--save_dir", type=str, default="data/raw/ohlcv/", help="Directory to save data")
    parser.add_argument("--batch_size", type=int, default=100, help="Save progress every N tickers")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh (ignore previous progress)")

    args = parser.parse_args()

    # Parse tickers
    if args.tickers.endswith('.json'):
        with open(args.tickers, 'r') as f:
            tickers = json.load(f)
    else:
        tickers = args.tickers.split(',')

    # Fetch data
    df = fetch_ohlcv_batch(
        tickers,
        args.start_date,
        args.end_date,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        resume=not args.no_resume
    )

    print("\n=== Sample Data ===")
    print(df.head())
