# src/modeling/add_cross_sectional_features.py
"""
Module to add cross-sectional features for stock ranking models.
Transforms raw features into relative metrics that are comparable across stocks and time periods.
"""
import gc
import pandas as pd
import numpy as np
from typing import List, Optional
from src.features.gkx_registry import get_gkx_feature_names


def _as_float32(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def _work_df(df: pd.DataFrame, inplace: bool) -> pd.DataFrame:
    return df if inplace else df.copy()


def optimize_numeric_memory(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Downcast wide numeric panels to reduce RAM pressure.
    """
    exclude = set(exclude or [])
    out = df
    float64_cols = [c for c in out.select_dtypes(include=["float64"]).columns if c not in exclude]
    for c in float64_cols:
        out[c] = out[c].astype(np.float32)
    return out


def add_percentile_ranks(df: pd.DataFrame,
                         features: List[str],
                         groupby_col: str = 'date',
                         suffix: str = '_rank',
                         inplace: bool = False) -> pd.DataFrame:
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
    df_copy = _work_df(df, inplace)
    grouped = df_copy.groupby(groupby_col, sort=False)

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue
        if not pd.api.types.is_numeric_dtype(df_copy[feature]):
            continue

        rank_col = f"{feature}{suffix}"
        df_copy[rank_col] = _as_float32(grouped[feature].rank(pct=True, method='average'))

    return df_copy


def add_z_scores(df: pd.DataFrame,
                 features: List[str],
                 groupby_col: str = 'date',
                 suffix: str = '_zscore',
                 inplace: bool = False) -> pd.DataFrame:
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
    df_copy = _work_df(df, inplace)
    grouped = df_copy.groupby(groupby_col, sort=False)

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue
        if not pd.api.types.is_numeric_dtype(df_copy[feature]):
            continue

        zscore_col = f"{feature}{suffix}"
        mean = grouped[feature].transform("mean")
        std = grouped[feature].transform("std")
        z = (df_copy[feature] - mean) / (std + 1e-8)
        df_copy[zscore_col] = _as_float32(z)

    return df_copy


def add_sector_relative_features(df: pd.DataFrame,
                                  features: List[str],
                                  sector_col: str = 'sector',
                                  groupby_col: str = 'date',
                                  suffix: str = '_sector_rel',
                                  inplace: bool = False) -> pd.DataFrame:
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
    df_copy = _work_df(df, inplace)

    if sector_col not in df_copy.columns:
        print(f"Warning: {sector_col} not in DataFrame, skipping sector-relative features...")
        return df_copy

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue
        if not pd.api.types.is_numeric_dtype(df_copy[feature]):
            continue

        rel_col = f"{feature}{suffix}"
        # Calculate sector mean for each date
        sector_mean = df_copy.groupby([groupby_col, sector_col])[feature].transform('mean')
        df_copy[rel_col] = _as_float32(df_copy[feature] - sector_mean)

    return df_copy


def add_sector_ranks(df: pd.DataFrame,
                     features: List[str],
                     sector_col: str = 'sector',
                     groupby_col: str = 'date',
                     suffix: str = '_sector_rank',
                     inplace: bool = False) -> pd.DataFrame:
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
    df_copy = _work_df(df, inplace)

    if sector_col not in df_copy.columns:
        print(f"Warning: {sector_col} not in DataFrame, skipping sector ranks...")
        return df_copy

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue
        if not pd.api.types.is_numeric_dtype(df_copy[feature]):
            continue

        rank_col = f"{feature}{suffix}"
        df_copy[rank_col] = _as_float32(df_copy.groupby([groupby_col, sector_col], sort=False)[feature].rank(
            pct=True, method='average'
        ))

    return df_copy


def add_quintile_buckets(df: pd.DataFrame,
                         features: List[str],
                         groupby_col: str = 'date',
                         n_buckets: int = 5,
                         suffix: str = '_quintile',
                         rank_suffix: str = '_rank',
                         inplace: bool = False) -> pd.DataFrame:
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
    df_copy = _work_df(df, inplace)

    for feature in features:
        if feature not in df_copy.columns:
            print(f"Warning: {feature} not in DataFrame, skipping...")
            continue
        if not pd.api.types.is_numeric_dtype(df_copy[feature]):
            continue

        bucket_col = f"{feature}{suffix}"
        rank_col = f"{feature}{rank_suffix}"

        if rank_col in df_copy.columns:
            # Prefer precomputed percentile ranks to avoid expensive qcut/groupby transforms.
            pct = pd.to_numeric(df_copy[rank_col], errors="coerce")
        else:
            pct = df_copy.groupby(groupby_col, sort=False)[feature].rank(pct=True, method='average')

        # Convert percentile rank [0,1] into integer buckets [0, n_buckets-1].
        bucket = np.floor((pct - 1e-12) * n_buckets).clip(0, n_buckets - 1)
        bucket = bucket.where(pct.notna(), np.nan)
        df_copy[bucket_col] = bucket.astype(np.float32)

    return df_copy


def add_rank_changes(df: pd.DataFrame,
                     rank_features: List[str],
                     periods: List[int] = [1, 5, 20],
                     suffix: str = '_rank_change',
                     inplace: bool = False,
                     assume_sorted: bool = False) -> pd.DataFrame:
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
    df_copy = _work_df(df, inplace)

    # Ensure sorted
    if not assume_sorted:
        df_copy = df_copy.sort_values(['ticker', 'date']).reset_index(drop=True)

    for rank_feature in rank_features:
        if rank_feature not in df_copy.columns:
            print(f"Warning: {rank_feature} not in DataFrame, skipping...")
            continue

        for period in periods:
            change_col = f"{rank_feature}{suffix}_{period}d"
            df_copy[change_col] = _as_float32(df_copy.groupby('ticker', sort=False)[rank_feature].diff(periods=period))

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
    df = optimize_numeric_memory(df, exclude=["date"])

    # Registry-driven GKX base features plus optional controls for backward compatibility.
    gkx_features = get_gkx_feature_names()
    optional_controls = [
        'open', 'high', 'low', 'close', 'volume',
        'profit_margin', 'operating_margin', 'gross_margin',
        'roe', 'roa', 'total_cash', 'total_debt',
        'debt_to_equity', 'current_ratio', 'quick_ratio'
    ]

    # Filter to only features that exist in the DataFrame
    all_features = gkx_features + optional_controls
    existing_features = [
        f for f in all_features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    if not existing_features:
        print("Warning: No standard features found in DataFrame!")
        print(f"Available columns: {df.columns.tolist()}")
        return df

    print(f"Transforming {len(existing_features)} features...")

    # 1. Add percentile ranks
    print("  - Adding percentile ranks...")
    df = add_percentile_ranks(df, existing_features, inplace=True)
    gc.collect()

    # 2. Add quintile buckets for key features (from rank columns to reduce memory).
    print("  - Adding quintile buckets...")
    key_features = [f for f in ['mom12m', 'mom6m', 'bm', 'ep', 'roeq', 'roaq', 'operprof'] if f in df.columns]
    if key_features:
        df = add_quintile_buckets(df, key_features, inplace=True)
    gc.collect()

    # 3. Add z-scores
    print("  - Adding z-scores...")
    df = add_z_scores(df, existing_features, inplace=True)
    gc.collect()

    # 4. Add sector-relative features (if sector column exists)
    if sector_col and sector_col in df.columns:
        print("  - Adding sector-relative features...")
        df = add_sector_relative_features(df, existing_features, sector_col=sector_col, inplace=True)

        print("  - Adding sector ranks...")
        df = add_sector_ranks(df, existing_features, sector_col=sector_col, inplace=True)
    else:
        print(f"  - Skipping sector features ('{sector_col}' column not found)")
    gc.collect()

    # 5. Add rank momentum (changes in ranks over time)
    print("  - Adding rank momentum...")
    rank_momentum_base = [
        f for f in ["mom12m", "mom6m", "bm", "ep", "roeq"]
        if f in existing_features
    ]
    rank_features = [f"{feat}_rank" for feat in rank_momentum_base if f"{feat}_rank" in df.columns]
    if rank_features:
        df = add_rank_changes(df, rank_features, inplace=True, assume_sorted=True)
    gc.collect()

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
