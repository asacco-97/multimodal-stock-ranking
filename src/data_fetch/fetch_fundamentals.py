# src/data_fetch/fetch_fundamentals.py
"""
Module to fetch fundamental financial data for equities.
Supports quarterly and annual financial statements, valuation metrics, and ratios.
"""
import yfinance as yf
import pandas as pd
import os
import json
from typing import List, Dict, Optional
from time import sleep
from datetime import datetime

def fetch_ticker_fundamentals(ticker: str) -> Dict:
    """
    Fetch comprehensive fundamental data for a single ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing financial statements and key metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key valuation metrics
        fundamentals = {
            'ticker': ticker,
            'fetch_date': datetime.now().strftime('%Y-%m-%d'),

            # Valuation metrics
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_to_revenue': info.get('enterpriseToRevenue'),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda'),

            # Profitability metrics
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'gross_margin': info.get('grossMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),

            # Growth metrics
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),

            # Financial health
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),

            # Per-share metrics
            'book_value': info.get('bookValue'),
            'revenue_per_share': info.get('revenuePerShare'),
            'earnings_per_share': info.get('trailingEps'),

            # Dividend metrics
            'dividend_rate': info.get('dividendRate'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),

            # Other
            'beta': info.get('beta'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'float_shares': info.get('floatShares'),
            'held_percent_insiders': info.get('heldPercentInsiders'),
            'held_percent_institutions': info.get('heldPercentInstitutions'),

            # Classification
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }

        return fundamentals

    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return None

def fetch_quarterly_financials(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly financial statements for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame with quarterly income statement, balance sheet, and cash flow data
    """
    try:
        stock = yf.Ticker(ticker)

        # Get quarterly statements
        quarterly_income = stock.quarterly_financials
        quarterly_balance = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow

        if quarterly_income is None or quarterly_income.empty:
            return None

        # Transpose so dates are rows
        quarterly_income = quarterly_income.T
        quarterly_balance = quarterly_balance.T if quarterly_balance is not None else pd.DataFrame()
        quarterly_cashflow = quarterly_cashflow.T if quarterly_cashflow is not None else pd.DataFrame()

        # Combine all statements
        combined = pd.concat([quarterly_income, quarterly_balance, quarterly_cashflow], axis=1)
        combined['ticker'] = ticker
        combined['date'] = combined.index
        combined.reset_index(drop=True, inplace=True)

        return combined

    except Exception as e:
        print(f"Error fetching quarterly financials for {ticker}: {e}")
        return None

def fetch_fundamentals_batch(tickers: List[str], save_dir: str = "data/raw/fundamentals/",
                             include_quarterly: bool = False) -> pd.DataFrame:
    """
    Fetch fundamental data for a batch of tickers.

    Args:
        tickers: List of ticker symbols
        save_dir: Directory to save the data
        include_quarterly: Whether to fetch quarterly financial statements (slower)

    Returns:
        DataFrame containing fundamental metrics for all tickers
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'quarterly'), exist_ok=True)

    all_fundamentals = []
    failed_tickers = []

    print(f"Fetching fundamentals for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(tickers)}")

        # Fetch current fundamentals
        fundamentals = fetch_ticker_fundamentals(ticker)

        if fundamentals:
            all_fundamentals.append(fundamentals)

            # Optionally fetch quarterly data
            if include_quarterly:
                quarterly = fetch_quarterly_financials(ticker)
                if quarterly is not None:
                    quarterly_path = os.path.join(save_dir, 'quarterly', f'{ticker}_quarterly.csv')
                    quarterly.to_csv(quarterly_path, index=False)

        else:
            failed_tickers.append(ticker)

        # Rate limiting
        sleep(0.5)

    # Create DataFrame
    fundamentals_df = pd.DataFrame(all_fundamentals)

    # Save results
    output_path = os.path.join(save_dir, 'fundamentals_current.csv')
    fundamentals_df.to_csv(output_path, index=False)

    # Save failed tickers
    if failed_tickers:
        failed_path = os.path.join(save_dir, 'failed_tickers.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_tickers, f, indent=2)
        print(f"\nWarning: Failed to fetch {len(failed_tickers)} tickers. See {failed_path}")

    print(f"\n=== Summary ===")
    print(f"Successfully fetched fundamentals for {len(fundamentals_df)} tickers")
    print(f"Failed: {len(failed_tickers)} tickers")
    print(f"Saved to: {output_path}")

    return fundamentals_df

def merge_fundamentals_with_prices(price_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fundamental data with price data using point-in-time logic.

    For each ticker/date in price_df, assigns the most recent quarterly
    fundamentals that were available (reported) BEFORE that date.
    This prevents look-ahead bias.

    Requires fundamentals_df to have a date column ('report_date',
    'quarter_end_date', or 'fetch_date') for temporal alignment.

    Args:
        price_df: DataFrame with columns ['ticker', 'date', ...price data...]
        fundamentals_df: DataFrame with fundamental metrics and a date column

    Returns:
        Merged DataFrame with fundamentals from the prior quarter
    """
    price_df = price_df.copy()
    fundamentals_df = fundamentals_df.copy()

    price_df['date'] = pd.to_datetime(price_df['date']).dt.as_unit('ns')

    # Determine which date column to use for point-in-time alignment
    if 'report_date' in fundamentals_df.columns:
        merge_col = 'report_date'
    elif 'quarter_end_date' in fundamentals_df.columns:
        merge_col = 'quarter_end_date'
    elif 'fetch_date' in fundamentals_df.columns:
        merge_col = 'fetch_date'
    else:
        raise ValueError(
            "fundamentals_df must have 'report_date', 'quarter_end_date', or 'fetch_date' "
            "column for point-in-time merge. A simple ticker-only merge would broadcast "
            "current fundamentals to all historical dates, causing look-ahead bias."
        )

    fundamentals_df[merge_col] = pd.to_datetime(fundamentals_df[merge_col]).dt.as_unit('ns')

    # merge_asof requires left_on and right_on to be globally sorted
    price_df = price_df.sort_values('date').reset_index(drop=True)
    fundamentals_df = fundamentals_df.sort_values(merge_col).reset_index(drop=True)

    # Point-in-time merge: for each date, get the most recent fundamentals
    # where the reporting/availability date is on or before the price date
    merged = pd.merge_asof(
        price_df,
        fundamentals_df,
        left_on='date',
        right_on=merge_col,
        by='ticker',
        direction='backward',
        suffixes=('', '_fundamental')
    )

    # Restore ticker+date sort order
    merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)

    coverage = merged[merge_col].notna().sum() / len(merged) * 100
    print(f"Merged fundamentals (point-in-time): {len(merged)} rows, {coverage:.1f}% coverage")

    return merged

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch fundamental data for equities")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers OR path to JSON file with ticker list")
    parser.add_argument("--save_dir", type=str, default="data/raw/fundamentals/", help="Directory to save data")
    parser.add_argument("--quarterly", action="store_true", help="Include quarterly financial statements")

    args = parser.parse_args()

    # Parse tickers
    if args.tickers.endswith('.json'):
        with open(args.tickers, 'r') as f:
            tickers = json.load(f)
    else:
        tickers = args.tickers.split(',')

    # Fetch data
    fundamentals_df = fetch_fundamentals_batch(tickers, save_dir=args.save_dir, include_quarterly=args.quarterly)

    print("\n=== Sample Data ===")
    print(fundamentals_df.head())

    print("\n=== Data Completeness ===")
    completeness = (1 - fundamentals_df.isnull().sum() / len(fundamentals_df)) * 100
    print(completeness.sort_values(ascending=False).head(20))
