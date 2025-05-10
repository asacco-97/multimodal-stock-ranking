import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from typing import Optional

def add_trading_metrics(
    df: pd.DataFrame,
    price_col: str = "close",
    return_col: str = "return_t+1"
) -> pd.DataFrame:
    """
    Adds trading metrics like momentum, volatility, moving average, RSI, and drawdown.
    Assumes input DataFrame has 'ticker', 'date', price_col, and return_col.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        price_col (str): Column name for price (e.g., 'close')
        return_col (str): Column name for target return (e.g., 'return_t+1')

    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    df = df.copy()
    df.sort_values(by=["ticker", "date"], inplace=True)

    def compute_rsi(series, window=14):
        return RSIIndicator(close=series, window=window).rsi()

    for window in [5, 10, 20]:
        df[f"momentum_{window}"] = df.groupby("ticker")[price_col].transform(lambda x: x / x.shift(window) - 1)
        df[f"volatility_{window}"] = df.groupby("ticker")[return_col].transform(lambda x: x.rolling(window).std())
        df[f"sma_{window}"] = df.groupby("ticker")[price_col].transform(lambda x: x.rolling(window).mean())

    df["rsi_14"] = df.groupby("ticker")[price_col].transform(compute_rsi)

    df["cumulative_return"] = df.groupby("ticker")[return_col].cumsum()
    df["running_max"] = df.groupby("ticker")["cumulative_return"].cummax()
    df["drawdown"] = df["cumulative_return"] - df["running_max"]

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add trading indicators to stock dataframe.")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet file path")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file path")
    parser.add_argument("--price_col", type=str, default="close", help="Price column name")
    parser.add_argument("--return_col", type=str, default="return_t+1", help="Return column name")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = add_trading_metrics(df, price_col=args.price_col, return_col=args.return_col)
    df.to_parquet(args.output, index=False)
    print(f"Saved dataset with trading metrics to {args.output}")