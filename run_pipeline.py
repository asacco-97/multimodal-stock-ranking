import os
from datetime import datetime
from src.data_fetch.fetch_ohlcv import fetch_ohlcv
from src.data_fetch.fetch_news import fetch_news
from src.data_fetch.build_daily_dataset import load_ohlcv_data, load_news_data, build_dataset
from src.embeddings.embed_news import embed_news
from src.utils.add_trading_metrics import add_trading_metrics
import pandas as pd

def run_pipeline(tickers, start_date, end_date):
    print("\n Step 1: Fetching OHLCV data...")
    fetch_ohlcv(tickers, start_date, end_date)

    print("\n Step 2: Fetching News data...")
    fetch_news(tickers, start_date, end_date)

    print("\n Step 3: Building combined OHLCV + News dataset...")
    ohlcv_data = load_ohlcv_data("data/raw/ohlcv")
    news_data = load_news_data("data/raw/news")
    build_dataset(ohlcv_data, news_data, output_file="data/processed/daily_dataset.csv")

    print("\n🤖 Step 4: Embedding news headlines with BERT...")
    embed_news(
        input_path="data/processed/daily_dataset.csv",
        output_path="data/processed/daily_with_embeddings.parquet"
    )

    print("\n Step 5: Adding trading indicators...")
    df = pd.read_parquet("data/processed/daily_with_embeddings.parquet")
    df = add_trading_metrics(df, price_col="close", return_col="return_t+1")
    df.to_parquet("data/processed/daily_with_indicators.parquet", index=False)
    print("\n Pipeline complete. Enriched dataset saved to data/processed/daily_with_indicators.parquet")

if __name__ == "__main__":
    # Customize your tickers and date range here
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "JPM", "UNH", "XOM", "HD", "NVDA", "KO"]
    start_date = "2024-05-01"
    end_date = "2025-05-01"

    run_pipeline(tickers, start_date, end_date)
