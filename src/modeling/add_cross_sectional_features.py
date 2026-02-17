# src/modeling/add_cross_sectional_features.py
"""
Module to add cross-sectional features for stock ranking models.
Transforms raw features into relative metrics that are comparable across stocks and time periods.
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def add_percentile_ranks(df: pd.DataFrame,
                         features: List[str],
                         groupby_col: str = 'date',
                         suffix: str = '_rank') -> pd.DataFrame:
    """
    Add percentile rank features for each feature within each time period.

    Args:
        df: Input DataFrame
        features: List of feature column names to rank
        groupby_col: Column to group by (typically 'date')
        suffix: Suffix to add to rank column names

    Returns:
        DataFrame with added rank columns
    """
    df_copy = df.copy()

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue

        rank_col = f"{feature}{suffix}"
        df_copy[rank_col] = df_copy.groupby(groupby_col)[feature].rank(pct=True, method='average')

    return df_copy


def add_z_scores(df: pd.DataFrame,
                 features: List[str],
                 groupby_col: str = 'date',
                 suffix: str = '_zscore') -> pd.DataFrame:
    """
    Add z-score normalized features within each time period.

    Args:
        df: Input DataFrame
        features: List of feature column names to normalize
        groupby_col: Column to group by (typically 'date')
        suffix: Suffix to add to z-score column names

    Returns:
        DataFrame with added z-score columns
    """
    df_copy = df.copy()

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue

        zscore_col = f"{feature}{suffix}"
        df_copy[zscore_col] = df_copy.groupby(groupby_col)[feature].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)  # Add small epsilon to avoid division by zero
        )

    return df_copy


def add_sector_relative_features(df: pd.DataFrame,
                                  features: List[str],
                                  sector_col: str = 'sector',
                                  groupby_col: str = 'date',
                                  suffix: str = '_sector_rel') -> pd.DataFrame:
    """
    Add sector-relative features (value minus sector mean).

    Args:
        df: Input DataFrame
        features: List of feature column names
        sector_col: Column containing sector information
        groupby_col: Column to group by (typically 'date')
        suffix: Suffix to add to sector-relative column names

    Returns:
        DataFrame with added sector-relative columns
    """
    df_copy = df.copy()

    if sector_col not in df_copy.columns:
        print(f"Warning: {sector_col} not in DataFrame, skipping sector-relative features...")
        return df_copy

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue

        rel_col = f"{feature}{suffix}"
        # Calculate sector mean for each date
        sector_mean = df_copy.groupby([groupby_col, sector_col])[feature].transform('mean')
        df_copy[rel_col] = df_copy[feature] - sector_mean

    return df_copy


def add_sector_ranks(df: pd.DataFrame,
                     features: List[str],
                     sector_col: str = 'sector',
                     groupby_col: str = 'date',
                     suffix: str = '_sector_rank') -> pd.DataFrame:
    """
    Add within-sector percentile ranks.

    Args:
        df: Input DataFrame
        features: List of feature column names to rank
        sector_col: Column containing sector information
        groupby_col: Column to group by (typically 'date')
        suffix: Suffix to add to sector rank column names

    Returns:
        DataFrame with added sector rank columns
    """
    df_copy = df.copy()

    if sector_col not in df_copy.columns:
        print(f"Warning: {sector_col} not in DataFrame, skipping sector ranks...")
        return df_copy

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue

        rank_col = f"{feature}{suffix}"
        df_copy[rank_col] = df_copy.groupby([groupby_col, sector_col])[feature].rank(
            pct=True, method='average'
        )

    return df_copy


def add_quintile_buckets(df: pd.DataFrame,
                         features: List[str],
                         groupby_col: str = 'date',
                         n_buckets: int = 5,
                         suffix: str = '_quintile') -> pd.DataFrame:
    """
    Add quintile/decile buckets for features.

    Args:
        df: Input DataFrame
        features: List of feature column names to bucket
        groupby_col: Column to group by (typically 'date')
        n_buckets: Number of buckets (5 for quintiles, 10 for deciles)
        suffix: Suffix to add to bucket column names

    Returns:
        DataFrame with added bucket columns
    """
    df_copy = df.copy()

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue

        bucket_col = f"{feature}{suffix}"
        df_copy[bucket_col] = df_copy.groupby(groupby_col)[feature].transform(
            lambda x: pd.qcut(x, q=n_buckets, labels=False, duplicates='drop')
        )

    return df_copy


def add_rank_changes(df: pd.DataFrame,
                     rank_features: List[str],
                     periods: List[int] = [1, 5, 20],
                     suffix: str = '_rank_change') -> pd.DataFrame:
    """
    Add changes in rank over time (rank momentum).

    Args:
        df: Input DataFrame (must be sorted by ticker and date)
        rank_features: List of rank column names
        periods: List of periods to calculate rank changes
        suffix: Suffix to add to rank change column names

    Returns:
        DataFrame with added rank change columns
    """
    df_copy = df.copy()

    # Ensure sorted
    df_copy = df_copy.sort_values(['ticker', 'date']).reset_index(drop=True)

    for rank_feature in rank_features:
        if rank_feature not in df_copy.columns:
            print(f"Warning: {rank_feature} not in DataFrame, skipping...")
            continue

        for period in periods:
            change_col = f"{rank_feature}{suffix}_{period}d"
            df_copy[change_col] = df_copy.groupby('ticker')[rank_feature].diff(periods=period)

    return df_copy


def create_target_variable(df: pd.DataFrame,
                           forward_days: int = 1,
                           target_type: str = 'quintile',
                           groupby_col: str = 'date',
                           price_col: str = 'close') -> pd.DataFrame:
    """
    Create target variable for modeling by calculating forward returns.

    Args:
        df: Input DataFrame (must have 'ticker', 'date', and price_col)
        forward_days: Number of days to look ahead for return calculation
        target_type: Type of target - 'quintile', 'decile', 'binary_top_bottom', 'continuous_rank'
        groupby_col: Column to group by for ranking (typically 'date')
        price_col: Column to use for price (default 'close')

    Returns:
        DataFrame with added target column and forward return column
    """
    df_copy = df.copy()

    # Ensure sorted by ticker and date
    df_copy = df_copy.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Calculate forward return
    return_col = f'return_t+{forward_days}'
    df_copy[return_col] = df_copy.groupby('ticker')[price_col].transform(
        lambda x: x.shift(-forward_days) / x - 1
    )

    # Check if return column has valid data
    if df_copy[return_col].isna().all():
        print(f"Warning: All forward returns are NaN. Check if data has enough future periods.")
        return df_copy

    # Create target based on cross-sectional ranking
    target_col = f'target_{forward_days}d'

    if target_type == 'quintile':
        # 5 buckets: 0 (worst), 1, 2, 3, 4 (best)
        # Use labels=False to get integer codes, then they'll be 0-4 (or fewer if duplicates)
        df_copy[target_col] = df_copy.groupby(groupby_col)[return_col].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        )

    elif target_type == 'decile':
        # 10 buckets: 0 (worst) to 9 (best)
        df_copy[target_col] = df_copy.groupby(groupby_col)[return_col].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
        )

    elif target_type == 'binary_top_bottom':
        # 0 = bottom 15%, 1 = middle 70%, 2 = top 15%
        df_copy[target_col] = df_copy.groupby(groupby_col)[return_col].transform(
            lambda x: pd.cut(x,
                           bins=[-np.inf, x.quantile(0.15), x.quantile(0.85), np.inf],
                           labels=False)
        )

    elif target_type == 'continuous_rank':
        # Percentile rank of returns (0 to 1)
        df_copy[target_col] = df_copy.groupby(groupby_col)[return_col].rank(pct=True)

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    print(f"Created target variable '{target_col}' based on {forward_days}-day forward returns")
    print(f"  Non-null targets: {df_copy[target_col].notna().sum()} / {len(df_copy)}")

    return df_copy


def add_all_cross_sectional_features(df: pd.DataFrame,
                                      sector_col: Optional[str] = 'sector') -> pd.DataFrame:
    """
    Main function to add all cross-sectional features at once.

    This function applies:
    1. Percentile ranks for key features
    2. Z-scores for key features
    3. Sector-relative features (if sector column exists)
    4. Sector ranks (if sector column exists)
    5. Quintile buckets for key features
    6. Rank momentum (changes in ranks over time)

    Target variables are NOT created here. Use src.utils.target_builder
    to add forward returns and binned targets separately.

    Args:
        df: Input DataFrame with merged data
        sector_col: Column containing sector information (None to skip sector features)

    Returns:
        DataFrame with all cross-sectional features added
    """
    print("Adding cross-sectional features for stock ranking model...")

    # Ensure sorted by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Define features to transform
    # These are common features that should exist after the pipeline
    price_features = ['open', 'high', 'low', 'close', 'volume']

    technical_features = [
        'momentum_20d', 'momentum_60d', 'momentum_120d',
        'volatility_20d', 'volatility_60d',
        'rsi_14d', 'max_drawdown_60d'
    ]

    fundamental_features = [
       'market_cap', 'enterprise_value', 'trailing_pe',
       'forward_pe', 'peg_ratio', 'price_to_book', 'price_to_sales',
       'enterprise_to_revenue', 'enterprise_to_ebitda', 'profit_margin',
       'operating_margin', 'gross_margin', 'roe', 'roa', 'revenue_growth',
       'earnings_growth', 'total_cash', 'total_debt', 'debt_to_equity',
       'current_ratio', 'quick_ratio', 'book_value', 'revenue_per_share',
       'earnings_per_share', 'dividend_rate', 'dividend_yield', 'payout_ratio',
       'beta', 'shares_outstanding', 'float_shares', 'held_percent_insiders',
       'held_percent_institutions',
    ]

    # Filter to only features that exist in the DataFrame
    all_features = price_features + technical_features + fundamental_features
    existing_features = [f for f in all_features if f in df.columns]

    if not existing_features:
        print("Warning: No standard features found in DataFrame!")
        print(f"Available columns: {df.columns.tolist()}")
        return df

    print(f"Transforming {len(existing_features)} features...")

    # 1. Add percentile ranks
    print("  - Adding percentile ranks...")
    df = add_percentile_ranks(df, existing_features)

    # 2. Add z-scores
    print("  - Adding z-scores...")
    df = add_z_scores(df, existing_features)

    # 3. Add sector-relative features (if sector column exists)
    if sector_col and sector_col in df.columns:
        print("  - Adding sector-relative features...")
        df = add_sector_relative_features(df, existing_features, sector_col=sector_col)

        print("  - Adding sector ranks...")
        df = add_sector_ranks(df, existing_features, sector_col=sector_col)
    else:
        print(f"  - Skipping sector features ('{sector_col}' column not found)")

    # 4. Add quintile buckets for key features
    print("  - Adding quintile buckets...")
    key_features = [f for f in ['momentum_60d', 'rsi_14d', 'pe_ratio', 'roe'] if f in df.columns]
    if key_features:
        df = add_quintile_buckets(df, key_features)

    # 5. Add rank momentum (changes in ranks over time)
    print("  - Adding rank momentum...")
    rank_features = [f"{feat}_rank" for feat in existing_features if f"{feat}_rank" in df.columns]
    if rank_features:
        df = add_rank_changes(df, rank_features) # Limit to 5 features to avoid too many columns

    print(f"Added cross-sectional features. New shape: {df.shape}")
    print(f"  Total features: {len([c for c in df.columns if c not in ['ticker', 'date']])}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add cross-sectional features to merged dataset")
    parser.add_argument("--input", type=str, default="data/processed/final_dataset.parquet",
                       help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/processed/modeling_dataset.parquet",
                       help="Output parquet file")
    parser.add_argument("--sector_col", type=str, default=None,
                       help="Column containing sector information (optional)")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)

    print(f"Input shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {df['ticker'].nunique()}")

    # Add cross-sectional features
    df_modeling = add_all_cross_sectional_features(
        df,
        sector_col=args.sector_col,
    )

    # Save
    print(f"\nSaving to {args.output}...")
    df_modeling.to_parquet(args.output, index=False)

    print("\n=== Summary ===")
    print(f"Output shape: {df_modeling.shape}")
    print(f"Features added: {len(df_modeling.columns) - len(df.columns)}")
    print(f"\nSample of new features:")
    new_cols = [c for c in df_modeling.columns if c not in df.columns]
    print(new_cols[:20])
