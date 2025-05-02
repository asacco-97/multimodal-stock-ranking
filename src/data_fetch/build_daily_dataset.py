import os
import json
import pandas as pd
from datetime import datetime, timedelta

def load_ohlcv_data(ohlcv_dir):
    data = {}
    for file in os.listdir(ohlcv_dir):
        if file.endswith(".csv"):
            ticker = file.split("_")[0]
            df = pd.read_csv(os.path.join(ohlcv_dir, file), parse_dates=["Date"])
            data[ticker] = df
    return data

def load_news_data(news_dir):
    news_data = {}
    for file in os.listdir(news_dir):
        if file.endswith(".json"):
            parts = file.replace(".json", "").split("_")
            ticker, date_str = parts[0], parts[1]
            with open(os.path.join(news_dir, file), "r") as f:
                headlines = json.load(f)
            if ticker not in news_data:
                news_data[ticker] = {}
            news_data[ticker][date_str] = [item["headline"] for item in headlines if item["headline"]]
    return news_data

def build_dataset(ohlcv_data, news_data, output_file="data/processed/daily_dataset.csv"):
    rows = []
    for ticker, df in ohlcv_data.items():
        df = df.sort_values("Date").reset_index(drop=True)
        for i in range(len(df) - 1):  # Use i+1 for next-day return
            date = df.loc[i, "Date"]
            next_day_close = float(df.loc[i + 1, "Close"])
            today_close = float(df.loc[i, "Close"])
            date_str = date.strftime("%Y-%m-%d")
            next_return = (next_day_close - today_close) / today_close

            row = {
                "ticker": ticker,
                "date": date_str,
                "open": df.loc[i, "Open"],
                "high": df.loc[i, "High"],
                "low": df.loc[i, "Low"],
                "close": df.loc[i, "Close"],
                "volume": df.loc[i, "Volume"],
                "return_t+1": next_return
            }

            # Add headlines if available
            headlines = news_data.get(ticker, {}).get(date_str, [])
            row["headlines"] = headlines
            rows.append(row)

    dataset = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset.to_csv(output_file, index=False)
    print(f"Saved dataset with {len(dataset)} rows to {output_file}")

if __name__ == "__main__":
    ohlcv_dir = "data/raw/ohlcv"
    news_dir = "data/raw/news"
    output_path = "data/processed/daily_dataset.csv"

    ohlcv_data = load_ohlcv_data(ohlcv_dir)
    news_data = load_news_data(news_dir)
    build_dataset(ohlcv_data, news_data, output_path)