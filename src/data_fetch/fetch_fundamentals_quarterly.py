# src/data_fetch/fetch_fundamentals_quarterly.py
"""
Fetch quarterly fundamentals with proper point-in-time handling to avoid look-ahead bias.

This module ensures that fundamentals are only available AFTER their reporting date,
simulating real-world conditions where quarterly reports have a lag.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Optional
from time import sleep
from datetime import datetime, timedelta


def fetch_quarterly_fundamentals(ticker: str, reporting_lag_days: int = 45) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly fundamentals with their reporting dates.

    Args:
        ticker: Stock ticker symbol
        reporting_lag_days: Days after quarter end before data is available (default 45 days for 10-Q filing deadline)

    Returns:
        DataFrame with columns: ticker, quarter_end_date, report_date, and all fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)

        # Get quarterly financials
        quarterly_income = stock.quarterly_financials
        quarterly_balance = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow

        if quarterly_income is None or quarterly_income.empty:
            print(f"  No quarterly data for {ticker}")
            return None

        # Transpose so dates are rows
        quarterly_income = quarterly_income.T
        quarterly_balance = quarterly_balance.T if quarterly_balance is not None else pd.DataFrame()
        quarterly_cashflow = quarterly_cashflow.T if quarterly_cashflow is not None else pd.DataFrame()

        # Get the quarter end dates (index)
        quarter_dates = quarterly_income.index

        # Calculate reporting dates (quarter end + lag)
        # In reality, companies report on different schedules, but we'll use a uniform lag as approximation
        report_dates = [qdate + pd.Timedelta(days=reporting_lag_days) for qdate in quarter_dates]

        # Extract key metrics from income statement
        fundamentals_list = []

        for i, (quarter_end, report_date) in enumerate(zip(quarter_dates, report_dates)):
            # Get income statement data
            income = quarterly_income.iloc[i] if i < len(quarterly_income) else pd.Series()
            balance = quarterly_balance.iloc[i] if i < len(quarterly_balance) else pd.Series()
            cashflow = quarterly_cashflow.iloc[i] if i < len(quarterly_cashflow) else pd.Series()

            # Calculate fundamental metrics from financial statements
            # Revenue and earnings
            revenue = income.get('Total Revenue', np.nan)
            net_income = income.get('Net Income', np.nan)
            operating_income = income.get('Operating Income', np.nan)
            gross_profit = income.get('Gross Profit', np.nan)
            ebitda = income.get('EBITDA', np.nan)

            # Balance sheet items
            total_assets = balance.get('Total Assets', np.nan)
            total_equity = balance.get('Stockholders Equity', balance.get('Total Equity Gross Minority Interest', np.nan))
            total_debt = balance.get('Total Debt', np.nan)
            current_assets = balance.get('Current Assets', np.nan)
            current_liabilities = balance.get('Current Liabilities', np.nan)
            cash = balance.get('Cash And Cash Equivalents', np.nan)
            inventory = balance.get('Inventory', np.nan)

            # Calculate ratios
            profit_margin = net_income / revenue if revenue and revenue != 0 else np.nan
            operating_margin = operating_income / revenue if revenue and revenue != 0 else np.nan
            gross_margin = gross_profit / revenue if revenue and revenue != 0 else np.nan

            roe = net_income / total_equity if total_equity and total_equity != 0 else np.nan
            roa = net_income / total_assets if total_assets and total_assets != 0 else np.nan

            debt_to_equity = total_debt / total_equity if total_equity and total_equity != 0 else np.nan
            current_ratio = current_assets / current_liabilities if current_liabilities and current_liabilities != 0 else np.nan
            quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities and current_liabilities != 0 else np.nan

            fundamentals = {
                'ticker': ticker,
                'quarter_end_date': quarter_end.strftime('%Y-%m-%d'),
                'report_date': report_date.strftime('%Y-%m-%d'),

                # Raw financial statement items
                'revenue': revenue,
                'net_income': net_income,
                'operating_income': operating_income,
                'gross_profit': gross_profit,
                'ebitda': ebitda,
                'total_assets': total_assets,
                'total_equity': total_equity,
                'total_debt': total_debt,
                'total_cash': cash,
                'current_assets': current_assets,
                'current_liabilities': current_liabilities,

                # Calculated ratios
                'profit_margin': profit_margin,
                'operating_margin': operating_margin,
                'gross_margin': gross_margin,
                'roe': roe,
                'roa': roa,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
            }

            fundamentals_list.append(fundamentals)

        df = pd.DataFrame(fundamentals_list)

        # Calculate quarter-over-quarter growth rates
        df = df.sort_values('quarter_end_date').reset_index(drop=True)
        df['revenue_growth_qoq'] = df['revenue'].pct_change()
        df['earnings_growth_qoq'] = df['net_income'].pct_change()

        # Calculate year-over-year growth (comparing to 4 quarters ago)
        df['revenue_growth_yoy'] = df['revenue'].pct_change(periods=4)
        df['earnings_growth_yoy'] = df['net_income'].pct_change(periods=4)

        return df

    except Exception as e:
        print(f"  Error fetching quarterly fundamentals for {ticker}: {e}")
        return None


def fetch_current_valuation_metrics(ticker: str) -> Dict:
    """
    Fetch current valuation metrics that change daily with stock price.
    These need to be calculated separately since P/E, P/B etc. depend on current price.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with current valuation metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'ticker': ticker,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'beta': info.get('beta'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'float_shares': info.get('floatShares'),
            'held_percent_insiders': info.get('heldPercentInsiders'),
            'held_percent_institutions': info.get('heldPercentInstitutions'),
        }

    except Exception as e:
        print(f"  Error fetching valuation for {ticker}: {e}")
        return None


def fetch_all_quarterly_fundamentals(tickers: List[str],
                                     save_dir: str = "data/raw/fundamentals_quarterly/",
                                     reporting_lag_days: int = 45) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals for a list of tickers.

    Args:
        tickers: List of stock ticker symbols
        save_dir: Directory to save data
        reporting_lag_days: Days after quarter end before data is available

    Returns:
        DataFrame with all quarterly fundamentals
    """
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    failed_tickers = []

    print(f"Fetching quarterly fundamentals for {len(tickers)} tickers...")
    print(f"Using reporting lag of {reporting_lag_days} days")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(tickers)}")

        df = fetch_quarterly_fundamentals(ticker, reporting_lag_days)

        if df is not None and not df.empty:
            all_data.append(df)
        else:
            failed_tickers.append(ticker)

        # Rate limiting
        sleep(0.5)

    if not all_data:
        print("No data fetched!")
        return pd.DataFrame()

    # Combine all data
    fundamentals_df = pd.concat(all_data, ignore_index=True)

    # Save
    output_path = os.path.join(save_dir, 'quarterly_fundamentals.parquet')
    fundamentals_df.to_parquet(output_path, index=False)

    # Save failed tickers
    if failed_tickers:
        failed_path = os.path.join(save_dir, 'failed_tickers.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_tickers, f, indent=2)
        print(f"\nWarning: Failed to fetch {len(failed_tickers)} tickers")

    print(f"\n=== Summary ===")
    print(f"Total quarterly records: {len(fundamentals_df)}")
    print(f"Unique tickers: {fundamentals_df['ticker'].nunique()}")
    print(f"Date range: {fundamentals_df['quarter_end_date'].min()} to {fundamentals_df['quarter_end_date'].max()}")
    print(f"Saved to: {output_path}")

    return fundamentals_df


def merge_fundamentals_point_in_time(price_df: pd.DataFrame,
                                     fundamentals_df: pd.DataFrame,
                                     valuation_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge quarterly fundamentals with price data using point-in-time logic.

    For each date in price_df, we use the most recent fundamentals that were
    REPORTED (available) before that date. This prevents look-ahead bias.

    Args:
        price_df: DataFrame with columns ['ticker', 'date', ...]
        fundamentals_df: DataFrame with quarterly fundamentals including 'report_date'
        valuation_df: Optional DataFrame with sector/industry info (merged on ticker)

    Returns:
        Merged DataFrame with fundamentals correctly aligned
    """
    print("Performing point-in-time merge of fundamentals...")

    # Ensure date columns are datetime
    price_df = price_df.copy()
    fundamentals_df = fundamentals_df.copy()

    price_df['date'] = pd.to_datetime(price_df['date']).dt.as_unit('ns')
    fundamentals_df['report_date'] = pd.to_datetime(fundamentals_df['report_date']).dt.as_unit('ns')
    fundamentals_df['quarter_end_date'] = pd.to_datetime(fundamentals_df['quarter_end_date']).dt.as_unit('ns')

    # merge_asof requires left_on and right_on to be globally sorted
    price_df = price_df.sort_values('date').reset_index(drop=True)
    fundamentals_df = fundamentals_df.sort_values('report_date').reset_index(drop=True)

    # Perform as-of merge (merge_asof)
    # For each ticker/date in price_df, get the most recent fundamentals where report_date <= date
    merged = pd.merge_asof(
        price_df,
        fundamentals_df,
        left_on='date',
        right_on='report_date',
        by='ticker',
        direction='backward',
        suffixes=('', '_fundamental')
    )

    # Restore ticker+date sort order
    merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Merge valuation metrics (sector, industry) if provided
    if valuation_df is not None:
        val_cols = [c for c in ['ticker', 'sector', 'industry'] if c in valuation_df.columns]
        merged = merged.merge(valuation_df[val_cols], on='ticker', how='left')

    print(f"Merged {len(merged)} rows")
    print(f"Fundamentals coverage: {merged['quarter_end_date'].notna().sum() / len(merged) * 100:.1f}%")

    # Report any missing fundamentals
    missing_pct = merged['quarter_end_date'].isna().sum() / len(merged) * 100
    if missing_pct > 0:
        print(f"Warning: {missing_pct:.1f}% of rows have no fundamentals (early dates or delisted stocks)")

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch quarterly fundamentals with point-in-time handling")
    parser.add_argument("--tickers", type=str, required=True,
                       help="Comma-separated list of tickers OR path to JSON file")
    parser.add_argument("--save_dir", type=str, default="data/raw/fundamentals_quarterly/",
                       help="Directory to save data")
    parser.add_argument("--reporting_lag", type=int, default=45,
                       help="Days after quarter end before data is available (default: 45)")

    args = parser.parse_args()

    # Parse tickers
    if args.tickers.endswith('.json'):
        with open(args.tickers, 'r') as f:
            data = json.load(f)
            tickers = data if isinstance(data, list) else data.get('tickers', [])
    else:
        tickers = args.tickers.split(',')

    print(f"Fetching quarterly fundamentals for {len(tickers)} tickers...")

    # Fetch quarterly fundamentals
    fundamentals_df = fetch_all_quarterly_fundamentals(
        tickers,
        save_dir=args.save_dir,
        reporting_lag_days=args.reporting_lag
    )

    print("\n=== Sample Data ===")
    print(fundamentals_df.head(10))

    print("\n=== Data Completeness ===")
    completeness = (1 - fundamentals_df.isnull().sum() / len(fundamentals_df)) * 100
    print(completeness.sort_values(ascending=False))
