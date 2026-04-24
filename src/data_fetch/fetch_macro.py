# src/data_fetch/fetch_macro.py
"""
Module to fetch macroeconomic data from the Federal Reserve Economic Data (FRED) API.
These features can help with market-level forecasts and regime detection.
"""
import pandas as pd
import os
from typing import List, Dict, Optional
from datetime import datetime
import requests
from time import sleep
from dotenv import load_dotenv, find_dotenv

# Load environment variables - find_dotenv() searches up the directory tree
load_dotenv(find_dotenv())

# You'll need to get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Key macroeconomic indicators based on the research document
MACRO_INDICATORS = {
    # Interest rates & monetary policy
    'DFF': 'Federal Funds Rate',
    'DGS10': '10-Year Treasury Rate',
    'DGS2': '2-Year Treasury Rate',
    'T10Y2Y': '10Y-2Y Treasury Spread',

    # Inflation
    'CPIAUCSL': 'Consumer Price Index',
    'CPILFESL': 'Core CPI (ex food & energy)',
    'PCEPI': 'Personal Consumption Expenditures Price Index',

    # Economic growth
    'GDP': 'Gross Domestic Product',
    'GDPC1': 'Real GDP',
    'INDPRO': 'Industrial Production Index',

    # Employment
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'ICSA': 'Initial Unemployment Claims',

    # Consumer & business sentiment
    'UMCSENT': 'University of Michigan Consumer Sentiment',
    'RSXFS': 'Retail Sales',

    # Market indicators
    'VIXCLS': 'VIX (Volatility Index)',
    'DCOILWTICO': 'WTI Crude Oil Price',

    # Money supply & credit
    'M2SL': 'M2 Money Supply',
    'TOTALSL': 'Total Consumer Credit',
}

def fetch_fred_series(series_id: str, start_date: str, end_date: str, api_key: str = None) -> Optional[pd.DataFrame]:
    """
    Fetch a single FRED time series.

    Args:
        series_id: FRED series identifier (e.g., 'GDP', 'UNRATE')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_key: FRED API key

    Returns:
        DataFrame with date and value columns
    """
    if api_key is None:
        api_key = FRED_API_KEY
    if not api_key:
        raise ValueError("FRED API key not set. Please add FRED_API_KEY to your .env file. Get one at https://fred.stlouisfed.org/docs/api/api_key.html")

    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'observations' not in data:
            print(f"No data found for {series_id}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data['observations'])
        df = df[['date', 'value']]
        df.columns = ['date', series_id]

        # Convert date and value to proper types
        df['date'] = pd.to_datetime(df['date'])
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')

        return df

    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return None

def fetch_all_macro_indicators(start_date: str, end_date: str,
                               indicators: Dict[str, str] = MACRO_INDICATORS,
                               save_dir: str = "data/raw/macro/") -> pd.DataFrame:
    """
    Fetch all macroeconomic indicators and merge into a single DataFrame.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        indicators: Dictionary of {series_id: description}
        save_dir: Directory to save the data

    Returns:
        DataFrame with all macro indicators merged by date
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Fetching {len(indicators)} macroeconomic indicators from FRED...")

    all_series = []

    for series_id, description in indicators.items():
        print(f"Fetching {series_id}: {description}")
        df = fetch_fred_series(series_id, start_date, end_date)

        if df is not None:
            all_series.append(df)

        # Be polite to the API
        sleep(0.5)

    # Merge all series on date
    if not all_series:
        print("No data fetched!")
        return pd.DataFrame()

    macro_df = all_series[0]
    for df in all_series[1:]:
        macro_df = macro_df.merge(df, on='date', how='outer')

    # Sort by date
    macro_df = macro_df.sort_values('date').reset_index(drop=True)

    # Forward fill missing values (many series are monthly/quarterly, need to broadcast to daily)
    macro_df = macro_df.ffill()

    # Save results
    output_path = os.path.join(save_dir, 'macro_indicators.csv')
    macro_df.to_csv(output_path, index=False)

    print(f"\n=== Summary ===")
    print(f"Fetched {len(macro_df.columns) - 1} indicators")
    print(f"Date range: {macro_df['date'].min()} to {macro_df['date'].max()}")
    print(f"Total rows: {len(macro_df)}")
    print(f"Saved to: {output_path}")

    return macro_df

def merge_macro_with_stocks(stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macroeconomic indicators with stock data.

    Args:
        stock_df: DataFrame with stock data (must have 'date' column)
        macro_df: DataFrame with macro indicators (must have 'date' column)

    Returns:
        Merged DataFrame with macro features added
    """
    # Ensure date columns are datetime
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    macro_df['date'] = pd.to_datetime(macro_df['date'])

    # Merge on date (left join to keep all stock data)
    merged = stock_df.merge(macro_df, on='date', how='left')

    # Forward fill any missing macro values
    macro_cols = [col for col in macro_df.columns if col != 'date']
    merged[macro_cols] = merged.groupby('ticker')[macro_cols].ffill()

    print(f"Merged macro data: {len(merged)} rows with {len(macro_cols)} macro features")

    return merged

def create_macro_derived_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from macro indicators (e.g., changes, trends).

    Args:
        macro_df: DataFrame with macro indicators

    Returns:
        DataFrame with additional derived features
    """
    df = macro_df.copy()

    # Calculate changes and momentum for key indicators
    for col in df.columns:
        if col != 'date':
            # 1-month change
            df[f'{col}_change_1m'] = df[col].pct_change(periods=20)  # ~1 month

            # 3-month change
            df[f'{col}_change_3m'] = df[col].pct_change(periods=60)  # ~3 months

            # Moving average
            df[f'{col}_ma_3m'] = df[col].rolling(window=60).mean()

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch macroeconomic data from FRED")
    parser.add_argument("--universe_file", type=str, default=None,
                        help="Unused for macro, accepted for CLI consistency")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data_tag", type=str, default=None, help="Run tag for output folder (default: timestamp)")
    parser.add_argument("--raw_root", type=str, default="data/raw", help="Root raw data directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--derive_features", action="store_true", help="Create derived features")

    args = parser.parse_args()
    data_tag = args.data_tag or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = args.save_dir or os.path.join(args.raw_root, data_tag, "macro")

    # Fetch macro data
    macro_df = fetch_all_macro_indicators(args.start_date, args.end_date, save_dir=save_dir)

    # Optionally create derived features
    if args.derive_features:
        print("\nCreating derived macro features...")
        macro_df = create_macro_derived_features(macro_df)

        derived_path = os.path.join(save_dir, 'macro_indicators_derived.csv')
        macro_df.to_csv(derived_path, index=False)
        print(f"Saved derived features to: {derived_path}")

    print("\n=== Sample Data ===")
    print(macro_df.head())

    print("\n=== Data Completeness ===")
    completeness = (1 - macro_df.isnull().sum() / len(macro_df)) * 100
    print(completeness.sort_values(ascending=False))
