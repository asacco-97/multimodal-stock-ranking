import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from typing import Optional

def add_trading_metrics(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Adds trading metrics like momentum, volatility, moving average, RSI, and drawdown.
    Uses only backward-looking calculations to avoid forward-looking bias.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'ticker', 'date', and price_col
        price_col (str): Column name for price (e.g., 'close')

    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    df = df.copy()
    df.sort_values(by=["ticker", "date"], inplace=True)

    # Compute backward-looking daily return from prices (no future data)
    df["_daily_return"] = df.groupby("ticker")[price_col].pct_change()

    def compute_rsi(series, window=14):
        return RSIIndicator(close=series, window=window).rsi()

    for window in [5, 10, 20]:
        df[f"momentum_{window}"] = df.groupby("ticker")[price_col].transform(lambda x: x / x.shift(window) - 1)
        df[f"volatility_{window}"] = df.groupby("ticker")["_daily_return"].transform(lambda x: x.rolling(window).std())
        df[f"sma_{window}"] = df.groupby("ticker")[price_col].transform(lambda x: x.rolling(window).mean())

    df["rsi_14"] = df.groupby("ticker")[price_col].transform(compute_rsi)

    df["cumulative_return"] = df.groupby("ticker")["_daily_return"].cumsum()
    df["running_max"] = df.groupby("ticker")["cumulative_return"].cummax()
    df["drawdown"] = df["cumulative_return"] - df["running_max"]

    df.drop(columns=["_daily_return"], inplace=True)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add trading indicators to stock dataframe.")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet file path")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file path")
    parser.add_argument("--price_col", type=str, default="close", help="Price column name")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = add_trading_metrics(df, price_col=args.price_col)
    df.to_parquet(args.output, index=False)
    print(f"Saved dataset with trading metrics to {args.output}")