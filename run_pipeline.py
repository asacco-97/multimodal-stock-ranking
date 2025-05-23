import os
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from src.data_fetch.fetch_ohlcv import fetch_ohlcv
from src.data_fetch.fetch_news import fetch_news
from src.data_fetch.build_daily_dataset import load_ohlcv_data, load_news_data, build_dataset
from src.embeddings.embed_news import embed_news
from src.utils.feature_engineering import add_trading_metrics, apply_exponential_smoothing
import pandas as pd

def run_pipeline(tickers, start_date, end_date):
    # print("\n Step 1: Fetching OHLCV data...")
    # fetch_ohlcv(tickers, start_date, end_date)

    print("\n Step 2: Fetching News data...")
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    # Compute one year ago from today since news headlines on FinnHub only go back until then
    # one_year_ago = date.today() - relativedelta(years=1)
    # news_start_date = max(start_date_dt, one_year_ago).strftime("%Y-%m-%d")
    # fetch_news(tickers, news_start_date, end_date)

    print("\n Step 3: Building combined OHLCV + News dataset...")
    ohlcv_data = load_ohlcv_data("data/raw/ohlcv")
    news_data = load_news_data("data/raw/news")
    build_dataset(ohlcv_data, news_data, output_file="data/processed/daily_dataset.csv")

    print("\n Step 4: Embedding news headlines with FinBERT...")
    embed_news(
        input_path="data/processed/daily_dataset.csv",
        output_path="data/processed/daily_with_finbert.parquet"
    )

    print("\n Step 5: Adding trading indicators...")
    df = pd.read_parquet("data/processed/daily_with_finbert.parquet")
    df = add_trading_metrics(df, price_col="close", return_col="return_t+1")
    df = apply_exponential_smoothing(df)
    df.to_parquet("data/processed/daily_with_finbert_and_indicators.parquet", index=False)
    print("\n Pipeline complete. Enriched dataset saved to data/processed/daily_with_finbert_and_indicators.parquet")

if __name__ == "__main__":
    # Customize your tickers and date range here
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "JPM", "UNH", "XOM", "HD", "NVDA", "KO"]
    start_date = "2010-05-01"
    end_date = "2025-05-28"

    run_pipeline(tickers, start_date, end_date)
