"""
Enhanced pipeline for comprehensive equity data collection and processing.
Supports top 1000 US equities with price, fundamental, macro, and news data.
"""
import os
from datetime import datetime
import pandas as pd
import json
from typing import Optional

# Import data fetching modules
from src.data_fetch.get_universe import get_top_n_equities_by_liquidity, load_universe
from src.data_fetch.fetch_ohlcv import fetch_ohlcv_batch, load_ohlcv_combined
from src.data_fetch.fetch_fundamentals import fetch_fundamentals_batch, merge_fundamentals_with_prices
from src.data_fetch.fetch_macro import fetch_all_macro_indicators, merge_macro_with_stocks
from src.data_fetch.fetch_news import fetch_news
from src.data_fetch.build_daily_dataset import load_ohlcv_data, load_news_data, build_dataset
from src.embeddings.embed_news import embed_news
from src.utils.add_trading_metrics import add_trading_metrics
from src.modeling.add_cross_sectional_features import add_all_cross_sectional_features

def run_full_pipeline(
    n_equities: int = 1000,
    start_date: str = "1995-01-01",
    end_date: str = "2026-02-09",
    steps: Optional[list] = None
):
    """
    Run the complete data pipeline for medium-term equity forecasting.

    Args:
        n_equities: Number of top equities to include (by liquidity)
        start_date: Start date for historical data
        end_date: End date for historical data
        steps: List of steps to run (None = run all)
               Options: ['universe', 'ohlcv', 'fundamentals', 'macro', 'news', 'embed',
                        'features', 'merge', 'cross_sectional']
    """

    if steps is None:
        steps = ['universe', 'ohlcv', 'fundamentals', 'macro', 'news', 'embed',
                'features', 'merge', 'cross_sectional']

    print(f"\n{'='*80}")
    print(f"EQUITY DATA PIPELINE - Top {n_equities} US Equities by Liquidity")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Steps: {', '.join(steps)}")
    print(f"{'='*80}\n")

    # ================================================================================
    # STEP 1: Build Universe - Get top N most liquid equities
    # ================================================================================
    if 'universe' in steps:
        print("\n" + "="*80)
        print("STEP 1: Building Universe")
        print("="*80)

        universe_df = get_top_n_equities_by_liquidity(n=n_equities)
        tickers = universe_df['ticker'].tolist()

    else:
        print("\nSkipping universe creation - loading existing universe...")
        tickers = load_universe("data/universe/top_1000_tickers.json")

    print(f"\nUniverse contains {len(tickers)} tickers")

    # ================================================================================
    # STEP 2: Fetch OHLCV Data
    # ================================================================================
    if 'ohlcv' in steps:
        print("\n" + "="*80)
        print("STEP 2: Fetching OHLCV Data")
        print("="*80)

        ohlcv_df = fetch_ohlcv_batch(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            save_dir="data/raw/ohlcv/"
        )
    else:
        print("\nSkipping OHLCV fetch - loading existing data...")
        ohlcv_df = load_ohlcv_combined("data/raw/ohlcv/")

    # ================================================================================
    # STEP 3: Fetch Fundamental Data
    # ================================================================================
    if 'fundamentals' in steps:
        print("\n" + "="*80)
        print("STEP 3: Fetching Fundamental Data")
        print("="*80)

        fundamentals_df = fetch_fundamentals_batch(
            tickers=tickers,
            save_dir="data/raw/fundamentals/"
        )
    else:
        print("\nSkipping fundamentals fetch - loading existing data...")
        fundamentals_df = pd.read_csv("data/raw/fundamentals/fundamentals_current.csv")

    # ================================================================================
    # STEP 4: Fetch Macroeconomic Data
    # ================================================================================
    if 'macro' in steps:
        print("\n" + "="*80)
        print("STEP 4: Fetching Macroeconomic Data")
        print("="*80)

        macro_df = fetch_all_macro_indicators(
            start_date=start_date,
            end_date=end_date,
            save_dir="data/raw/macro/"
        )
    else:
        print("\nSkipping macro fetch - loading existing data...")
        macro_df = pd.read_csv("data/raw/macro/macro_indicators.csv")
        macro_df['date'] = pd.to_datetime(macro_df['date'])

    # ================================================================================
    # STEP 5: Fetch News Data (Optional - can be slow for 1000 tickers)
    # ================================================================================
    if 'news' in steps:
        print("\n" + "="*80)
        print("STEP 5: Fetching News Data")
        print("="*80)
        print("WARNING: Fetching news for 1000 tickers may take a very long time!")
        print("Consider running this step separately for smaller batches.\n")

        # Optionally, only fetch news for a subset
        user_input = input("Fetch news for all tickers? (y/n): ")
        if user_input.lower() == 'y':
            fetch_news(tickers, start_date, end_date)

    # ================================================================================
    # STEP 6: Embed News Headlines (if news data exists)
    # ================================================================================
    if 'embed' in steps:
        print("\n" + "="*80)
        print("STEP 6: Embedding News Headlines")
        print("="*80)

        if os.path.exists("data/processed/daily_dataset.csv"):
            embed_news(
                input_path="data/processed/daily_dataset.csv",
                output_path="data/processed/daily_with_finbert.parquet"
            )
        else:
            print("Skipping embedding - no daily_dataset.csv found")

    # ================================================================================
    # STEP 7: Add Technical Indicators / Features
    # ================================================================================
    if 'features' in steps:
        print("\n" + "="*80)
        print("STEP 7: Adding Technical Indicators")
        print("="*80)

        # Make sure we have OHLCV data
        if ohlcv_df.empty:
            ohlcv_df = load_ohlcv_combined("data/raw/ohlcv/")

        # Calculate returns
        ohlcv_df = ohlcv_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        ohlcv_df['return_t+1'] = ohlcv_df.groupby('ticker')['Close'].transform(lambda x: x.pct_change().shift(-1))

        # Rename columns to lowercase for consistency
        ohlcv_df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        # Add trading metrics
        ohlcv_df = add_trading_metrics(ohlcv_df, price_col="close", return_col="return_t+1")

        # Save intermediate result
        os.makedirs("data/processed", exist_ok=True)
        ohlcv_df.to_parquet("data/processed/ohlcv_with_features.parquet", index=False)
        print("Saved: data/processed/ohlcv_with_features.parquet")

    # ================================================================================
    # STEP 8: Merge All Data Sources
    # ================================================================================
    if 'merge' in steps:
        print("\n" + "="*80)
        print("STEP 8: Merging All Data Sources")
        print("="*80)

        # Load OHLCV with features
        if os.path.exists("data/processed/ohlcv_with_features.parquet"):
            merged_df = pd.read_parquet("data/processed/ohlcv_with_features.parquet")
        else:
            merged_df = ohlcv_df

        # Merge fundamentals
        if fundamentals_df is not None and not fundamentals_df.empty:
            print("Merging fundamentals...")
            merged_df = merge_fundamentals_with_prices(merged_df, fundamentals_df)

        # Merge macro
        if macro_df is not None and not macro_df.empty:
            print("Merging macro indicators...")
            merged_df = merge_macro_with_stocks(merged_df, macro_df)

        # Save merged dataset
        final_path = "data/processed/final_dataset.parquet"
        merged_df.to_parquet(final_path, index=False)

        print(f"\nMerged dataset saved to: {final_path}")
        print(f"Shape: {merged_df.shape}")
    else:
        # Load existing merged data if skipping merge step
        if os.path.exists("data/processed/final_dataset.parquet"):
            merged_df = pd.read_parquet("data/processed/final_dataset.parquet")
        else:
            print("Warning: Merged dataset not found, skipping remaining steps")
            return None

    # ================================================================================
    # STEP 9: Add Cross-Sectional Features for Modeling
    # ================================================================================
    if 'cross_sectional' in steps:
        print("\n" + "="*80)
        print("STEP 9: Adding Cross-Sectional Features for Modeling")
        print("="*80)

        # Add cross-sectional features
        modeling_df = add_all_cross_sectional_features(
            merged_df,
            sector_col=None,  # Set to column name if you have sector data
            target_type='quintile'  # quintile, decile, binary_top_bottom, continuous_rank
        )

        # Save modeling dataset
        modeling_path = "data/processed/modeling_dataset.parquet"
        modeling_df.to_parquet(modeling_path, index=False)

        print(f"\nModeling dataset saved to: {modeling_path}")
        print(f"Shape: {modeling_df.shape}")
        print(f"Features for modeling: {len([c for c in modeling_df.columns if c not in ['ticker', 'date', 'target']])}")

        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"\nFinal modeling dataset ready at: {modeling_path}")
        print(f"Tickers: {modeling_df['ticker'].nunique()}")
        print(f"Date range: {modeling_df['date'].min()} to {modeling_df['date'].max()}")

        if 'target' in modeling_df.columns:
            print(f"\nTarget variable distribution:")
            print(modeling_df['target'].value_counts().sort_index())

        return modeling_df
    else:
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"\nMerged dataset saved to: {final_path}")
        print(f"Tickers: {merged_df['ticker'].nunique()}")
        print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

        return merged_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run equity data pipeline")
    parser.add_argument("--n_equities", type=int, default=1000, help="Number of top equities to include")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-02-09", help="End date (YYYY-MM-DD)")
    parser.add_argument("--steps", type=str, help="Comma-separated steps to run (default: all)")

    args = parser.parse_args()

    # Parse steps
    if args.steps:
        steps = [s.strip() for s in args.steps.split(',')]
    else:
        steps = None  # Run all

    # Run pipeline
    final_df = run_full_pipeline(
        n_equities=args.n_equities,
        start_date=args.start_date,
        end_date=args.end_date,
        steps=steps
    )
