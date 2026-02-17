"""
Target variable builder functions for stock ranking models.
Provides flexible creation of forward returns and binned targets
across multiple horizons and bin configurations.
"""
import pandas as pd
import numpy as np
from typing import List


def add_forward_returns(
    df: pd.DataFrame,
    periods: List[int] = [1, 10, 20],
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Add raw forward return columns for specified periods.

    Args:
        df: DataFrame with 'ticker', 'date', and price_col columns.
        periods: List of forward periods in trading days (e.g., [1, 10, 20]).
        price_col: Column to use for price.

    Returns:
        DataFrame with added columns: return_t+1, return_t+10, return_t+20, etc.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for period in periods:
        col = f"return_t+{period}"
        df[col] = df.groupby("ticker")[price_col].transform(
            lambda x: x.shift(-period) / x - 1
        )

    return df


def add_return_bins(
    df: pd.DataFrame,
    return_col: str,
    n_bins: int = 5,
    groupby_col: str = "date",
) -> pd.DataFrame:
    """
    Bin a forward return column cross-sectionally into quantile buckets.

    Args:
        df: DataFrame containing return_col and groupby_col.
        return_col: Name of the return column to bin (e.g., 'return_t+10').
        n_bins: Number of quantile bins (e.g., 4, 5, 10).
        groupby_col: Column to group by for cross-sectional binning.

    Returns:
        DataFrame with added column named '{return_col}_{n_bins}bin'
        (e.g., 'return_t+10_5bin'). Values are integers 0 to n_bins-1.
    """
    df = df.copy()
    target_col = f"{return_col}_{n_bins}bin"

    df[target_col] = df.groupby(groupby_col)[return_col].transform(
        lambda x: pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    )

    return df


def build_targets(
    df: pd.DataFrame,
    periods: List[int] = [1, 10, 20],
    bin_configs: List[int] = [4, 5, 10],
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Build all target variable combinations: raw forward returns + binned versions.

    Creates columns like:
        return_t+1, return_t+10, return_t+20          (raw returns)
        return_t+1_4bin, return_t+1_5bin, return_t+1_10bin
        return_t+10_4bin, return_t+10_5bin, return_t+10_10bin
        return_t+20_4bin, return_t+20_5bin, return_t+20_10bin

    Args:
        df: DataFrame with 'ticker', 'date', and price_col columns.
        periods: List of forward periods in trading days.
        bin_configs: List of bin counts to create for each period.
        price_col: Column to use for price.

    Returns:
        DataFrame with all target columns added.
    """
    df = add_forward_returns(df, periods=periods, price_col=price_col)

    for period in periods:
        return_col = f"return_t+{period}"
        for n_bins in bin_configs:
            df = add_return_bins(df, return_col=return_col, n_bins=n_bins)

    target_cols = [c for c in df.columns if c.startswith("return_t+")]
    print(f"Built {len(target_cols)} target columns: {sorted(target_cols)}")

    return df
