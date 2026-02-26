"""
Enhanced pipeline for comprehensive equity data collection and processing.
Supports top 1000 US equities with price, fundamental, macro, and news data.
"""
import os
from datetime import datetime
import pandas as pd
import json
from typing import Optional, List

# Import data fetching modules
from src.data_fetch.get_universe import get_top_n_equities_by_liquidity, load_universe
from src.data_fetch.fetch_ohlcv import fetch_ohlcv_batch, load_ohlcv_combined
from src.data_fetch.fetch_fundamentals_quarterly import (
    fetch_all_quarterly_fundamentals,
    fetch_current_valuation_metrics,
    merge_fundamentals_point_in_time
)
from src.data_fetch.fetch_macro import fetch_all_macro_indicators, merge_macro_with_stocks
from src.data_fetch.fetch_news import fetch_news
from src.data_fetch.build_daily_dataset import load_ohlcv_data, load_news_data, build_dataset
from src.embeddings.embed_news import embed_news
from src.utils.add_trading_metrics import add_trading_metrics
from src.modeling.add_cross_sectional_features import add_all_cross_sectional_features
from src.features.build_gkx_characteristics import build_and_attach_gkx
from src.constants import validate_gkx_schema

def run_full_pipeline(
    n_equities: int = 1000,
    start_date: str = "1995-01-01",
    end_date: str = "2026-02-09",
    steps: Optional[list] = None,
    data_tag: Optional[str] = None,
    exclude_etfs: bool = False,
    benchmark_tickers: Optional[List[str]] = None,
):
    """
    Run the complete data pipeline for medium-term equity forecasting.

    Args:
        n_equities: Number of top equities to include (by liquidity)
        start_date: Start date for historical data
        end_date: End date for historical data
        steps: List of steps to run (None = run all)
               Options: ['universe', 'ohlcv', 'fundamentals', 'macro', 'news', 'embed',
                        'features', 'merge', 'gkx_features', 'cross_sectional']
        data_tag: Unique tag for this run (default: auto-generated timestamp)
        exclude_etfs: If True, build universe from non-ETF equities only.
        benchmark_tickers: Optional tickers to force-include in the universe
                           (e.g., SPY/QQQ/IWM) for comparison benchmarks.
    """

    if steps is None:
        steps = ['universe', 'ohlcv', 'fundamentals', 'macro', 'news', 'embed',
                'features', 'merge', 'gkx_features', 'cross_sectional']

    if data_tag is None:
        data_tag = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create output directories for this run
    raw_dir = f"data/raw/{data_tag}"
    processed_dir = f"data/processed/{data_tag}"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EQUITY DATA PIPELINE - Top {n_equities} US Equities by Liquidity")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Data Tag: {data_tag}")
    print(f"Raw Dir: {raw_dir}")
    print(f"Processed Dir: {processed_dir}")
    print(f"Exclude ETFs: {exclude_etfs}")
    if benchmark_tickers:
        print(f"Forced Benchmark Tickers: {', '.join(benchmark_tickers)}")
    print(f"Steps: {', '.join(steps)}")
    print(f"{'='*80}\n")

    # ================================================================================
    # STEP 1: Build Universe - Get top N most liquid equities
    # ================================================================================
    if 'universe' in steps:
        print("\n" + "="*80)
        print("STEP 1: Building Universe")
        print("="*80)

        universe_df = get_top_n_equities_by_liquidity(
            n=n_equities,
            exclude_etfs=exclude_etfs,
            force_include_tickers=benchmark_tickers,
        )
        tickers = universe_df['ticker'].tolist()

    else:
        print("\nSkipping universe creation - loading existing universe...")
        default_path = (
            f"data/universe/top_{n_equities}_tickers_non_etf.json"
            if exclude_etfs
            else f"data/universe/top_{n_equities}_tickers.json"
        )
        if not os.path.exists(default_path):
            default_path = "data/universe/top_1000_tickers.json"
        tickers = load_universe(default_path)

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
            save_dir=f"{raw_dir}/ohlcv/"
        )
    else:
        print("\nSkipping OHLCV fetch - loading existing data...")
        ohlcv_df = load_ohlcv_combined(f"{raw_dir}/ohlcv/")

    # ================================================================================
    # STEP 3: Fetch Fundamental Data (Quarterly with Point-in-Time)
    # ================================================================================
    if 'fundamentals' in steps:
        print("\n" + "="*80)
        print("STEP 3: Fetching Quarterly Fundamental Data (Point-in-Time)")
        print("="*80)
        print("Using actual SEC filing dates from EDGAR (no estimated lag)")

        # Fetch quarterly fundamentals
        fundamentals_df = fetch_all_quarterly_fundamentals(
            tickers=tickers,
            save_dir=f"{raw_dir}/fundamentals_quarterly/",
        )

        # Fetch sector/industry info (doesn't change much, can use current)
        print("\nFetching sector/industry classifications...")
        valuation_data = []
        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(tickers)}")
            val = fetch_current_valuation_metrics(ticker)
            if val:
                valuation_data.append(val)

        valuation_df = pd.DataFrame(valuation_data)
        valuation_df.to_csv(f"{raw_dir}/fundamentals_quarterly/valuation_current.csv", index=False)
    else:
        print("\nSkipping fundamentals fetch - loading existing quarterly data...")
        if os.path.exists(f"{raw_dir}/fundamentals_quarterly/quarterly_fundamentals.parquet"):
            fundamentals_df = pd.read_parquet(f"{raw_dir}/fundamentals_quarterly/quarterly_fundamentals.parquet")
            valuation_df = pd.read_csv(f"{raw_dir}/fundamentals_quarterly/valuation_current.csv") if os.path.exists(f"{raw_dir}/fundamentals_quarterly/valuation_current.csv") else None
        else:
            print("Warning: No quarterly fundamentals found. Run with --steps fundamentals first.")
            fundamentals_df = None
            valuation_df = None

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
            save_dir=f"{raw_dir}/macro/"
        )
    else:
        print("\nSkipping macro fetch - loading existing data...")
        macro_df = pd.read_csv(f"{raw_dir}/macro/macro_indicators.csv")
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

        if os.path.exists(f"{processed_dir}/daily_dataset.csv"):
            embed_news(
                input_path=f"{processed_dir}/daily_dataset.csv",
                output_path=f"{processed_dir}/daily_with_finbert.parquet"
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
            ohlcv_df = load_ohlcv_combined(f"{raw_dir}/ohlcv/")

        ohlcv_df = ohlcv_df.sort_values(['ticker', 'date']).reset_index(drop=True)

        # Rename columns to lowercase for consistency
        ohlcv_df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        # Add trading metrics (uses backward-looking returns only, no future data)
        ohlcv_df = add_trading_metrics(ohlcv_df, price_col="close")

        # Save intermediate result
        ohlcv_path = f"{processed_dir}/ohlcv_with_features.parquet"
        ohlcv_df.to_parquet(ohlcv_path, index=False)
        print(f"Saved: {ohlcv_path}")

    # ================================================================================
    # STEP 8: Merge All Data Sources
    # ================================================================================
    if 'merge' in steps:
        print("\n" + "="*80)
        print("STEP 8: Merging All Data Sources")
        print("="*80)

        # Load OHLCV with features
        ohlcv_path = f"{processed_dir}/ohlcv_with_features.parquet"
        if os.path.exists(ohlcv_path):
            merged_df = pd.read_parquet(ohlcv_path)
        else:
            merged_df = ohlcv_df

        # Merge fundamentals (point-in-time to prevent look-ahead bias)
        if fundamentals_df is not None and not fundamentals_df.empty:
            print("Merging quarterly fundamentals (point-in-time)...")
            merged_df = merge_fundamentals_point_in_time(
                merged_df,
                fundamentals_df,
                valuation_df=valuation_df if 'valuation_df' in locals() else None
            )

        # Merge macro
        if macro_df is not None and not macro_df.empty:
            print("Merging macro indicators...")
            merged_df = merge_macro_with_stocks(merged_df, macro_df)

        # Save merged dataset
        final_path = f"{processed_dir}/final_dataset.parquet"
        merged_df.to_parquet(final_path, index=False)

        print(f"\nMerged dataset saved to: {final_path}")
        print(f"Shape: {merged_df.shape}")
    else:
        # Load existing merged data if skipping merge step
        final_path = f"{processed_dir}/final_dataset.parquet"
        if os.path.exists(final_path):
            merged_df = pd.read_parquet(final_path)
        else:
            print("Warning: Merged dataset not found, skipping remaining steps")
            return None

    # ================================================================================
    # STEP 9: Add GKX Features (Monthly proxies attached to daily panel)
    # ================================================================================
    if 'gkx_features' in steps:
        print("\n" + "="*80)
        print("STEP 9: Building GKX-94 Proxy Characteristics")
        print("="*80)
        merged_df, gkx_monthly_df, coverage_df = build_and_attach_gkx(merged_df)
        gkx_monthly_path = f"{processed_dir}/gkx_monthly.parquet"
        coverage_csv = f"{processed_dir}/gkx_coverage_report.csv"
        coverage_json = f"{processed_dir}/gkx_coverage_report.json"
        gkx_daily_path = f"{processed_dir}/final_dataset_with_gkx.parquet"
        gkx_monthly_df.to_parquet(gkx_monthly_path, index=False)
        coverage_df.to_csv(coverage_csv, index=False)
        coverage_df.to_json(coverage_json, orient="records", indent=2)
        merged_df.to_parquet(gkx_daily_path, index=False)
        schema = validate_gkx_schema(merged_df.columns.tolist())
        print(f"GKX schema: {len(schema['present'])}/{len(schema['required'])} present")
        if schema["missing"]:
            print(f"Warning: Missing GKX columns (still expected as NaN fallback in some cases): {schema['missing']}")
        print(f"Saved GKX monthly characteristics: {gkx_monthly_path}")
        print(f"Saved coverage report: {coverage_csv}")
        print(f"Saved GKX-attached daily dataset: {gkx_daily_path}")

    # ================================================================================
    # STEP 10: Add Cross-Sectional Features for Modeling
    # ================================================================================
    if 'cross_sectional' in steps:
        print("\n" + "="*80)
        print("STEP 10: Adding Cross-Sectional Features for Modeling")
        print("="*80)

        # Add cross-sectional features (no target creation — use target_builder separately)
        modeling_df = add_all_cross_sectional_features(
            merged_df,
            sector_col=None,  # Set to column name if you have sector data
        )

        # Save modeling dataset
        modeling_path = f"{processed_dir}/modeling_dataset.parquet"
        modeling_df.to_parquet(modeling_path, index=False)

        print(f"\nModeling dataset saved to: {modeling_path}")
        print(f"Shape: {modeling_df.shape}")
        print(f"Features: {len([c for c in modeling_df.columns if c not in ['ticker', 'date']])}")

        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"\nData tag: {data_tag}")
        print(f"Modeling dataset ready at: {modeling_path}")
        print(f"Tickers: {modeling_df['ticker'].nunique()}")
        print(f"Date range: {modeling_df['date'].min()} to {modeling_df['date'].max()}")
        print(f"\nTo add targets, use:")
        print(f"  from src.utils.target_builder import build_targets")
        print(f"  df = pd.read_parquet('{modeling_path}')")
        print(f"  df = build_targets(df)")

        return modeling_df
    else:
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"\nData tag: {data_tag}")
        print(f"Merged dataset saved to: {final_path}")
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
    parser.add_argument("--data_tag", type=str, default=None, help="Unique tag for this run (default: auto-generated timestamp)")
    parser.add_argument("--exclude_etfs", action="store_true", help="Exclude ETFs from ranked universe")
    parser.add_argument(
        "--benchmark_tickers",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated tickers to force-include in universe for benchmarks",
    )

    args = parser.parse_args()

    # Parse steps
    if args.steps:
        steps = [s.strip() for s in args.steps.split(',')]
    else:
        steps = None  # Run all

    benchmark_tickers = [
        t.strip().upper().replace(".", "-")
        for t in (args.benchmark_tickers or "").split(",")
        if t.strip()
    ]

    # Run pipeline
    final_df = run_full_pipeline(
        n_equities=args.n_equities,
        start_date=args.start_date,
        end_date=args.end_date,
        steps=steps,
        data_tag=args.data_tag,
        exclude_etfs=args.exclude_etfs,
        benchmark_tickers=benchmark_tickers,
    )
