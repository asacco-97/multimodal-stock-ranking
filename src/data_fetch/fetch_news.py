# src/data_fetch/fetch_news.py
import requests
import os
import json
from datetime import datetime, timedelta
from time import sleep
from pytz import timezone

FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"

def fetch_news_for_date(ticker, date, api_key=FINNHUB_API_KEY):
    url = (
        f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
        f"&from={date}&to={date}&token={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching news for {ticker} on {date}")
        return []

def filter_headlines_before_close(news_list, date):
    eastern = timezone("US/Eastern")
    market_close = eastern.localize(datetime.strptime(f"{date} 16:00:00", "%Y-%m-%d %H:%M:%S"))
    
    filtered = []
    for item in news_list:
        try:
            utc_time = datetime.utcfromtimestamp(item["datetime"])
            local_time = utc_time.replace(tzinfo=timezone("UTC")).astimezone(eastern)
            if local_time <= market_close:
                filtered.append({
                    "datetime": local_time.isoformat(),
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "")
                })
        except Exception:
            continue
    return filtered

def fetch_news(tickers, start_date, end_date, save_dir="data/raw/news/"):
    os.makedirs(save_dir, exist_ok=True)
    
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        for ticker in tickers:
            print(f"Fetching news for {ticker} on {date_str}")
            news = fetch_news_for_date(ticker, date_str)
            filtered_news = filter_headlines_before_close(news, date_str)
            
            if filtered_news:
                save_path = os.path.join(save_dir, f"{ticker}_{date_str}.json")
                with open(save_path, "w") as f:
                    json.dump(filtered_news, f, indent=2)
            sleep(1.2)  # To avoid API rate limits
        current_date += timedelta(days=1)
    
    print("Finished fetching news.")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT"]
    start_date = "2024-01-01"
    end_date = "2024-03-31"
    fetch_news(tickers, start_date, end_date)