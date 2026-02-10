# Data Collection Guide for Top 1000 US Equities

This guide explains how to use the enhanced data collection system for medium-term equity forecasting with the top 1000 US equities by liquidity.

## Overview

The system has been expanded to support comprehensive data collection across multiple dimensions:

1. **Universe Selection** - Identify top 1000 most liquid US equities
2. **Price Data (OHLCV)** - Historical price and volume data
3. **Fundamental Data** - Financial metrics, ratios, and company information
4. **Macroeconomic Data** - Economic indicators from FRED API
5. **News Data** - Company news with FinBERT embeddings
6. **Technical Features** - Momentum, volatility, RSI, and other indicators

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required for news data
FINNHUB_API_KEY=your_finnhub_api_key_here

# Required for macroeconomic data
FRED_API_KEY=your_fred_api_key_here
```

**Get API Keys:**
- Finnhub: https://finnhub.io/register (free tier available)
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html (free)

## Quick Start

### Option 1: Run Full Pipeline

```bash
python run_pipeline.py --n_equities 1000 --start_date 2020-01-01 --end_date 2025-02-09
```

### Option 2: Run Specific Steps

```bash
# Only build universe and fetch OHLCV data
python run_pipeline.py --steps universe,ohlcv --start_date 2020-01-01 --end_date 2025-02-09
```

Available steps:
- `universe` - Build top N equities list
- `ohlcv` - Fetch price/volume data
- `fundamentals` - Fetch financial metrics
- `macro` - Fetch economic indicators
- `news` - Fetch news articles (slow for 1000 tickers)
- `embed` - Embed news with FinBERT
- `features` - Add technical indicators
- `merge` - Merge all data sources

## Detailed Usage

### 1. Build Universe (Top 1000 by Liquidity)

```bash
python src/data_fetch/get_universe.py
```

This will:
- Fetch S&P 500 and NASDAQ 100 constituents
- Calculate average trading volume (90-day lookback)
- Rank by dollar volume (liquidity)
- Save top 1000 to `data/universe/top_1000_tickers.json`

**Output:**
- `data/universe/top_1000_equities_by_liquidity.csv` - Full metadata
- `data/universe/top_1000_tickers.json` - Just ticker symbols

### 2. Fetch OHLCV Data

```bash
python src/data_fetch/fetch_ohlcv.py \
  --tickers data/universe/top_1000_tickers.json \
  --start_date 2020-01-01 \
  --end_date 2025-02-09
```

**Features:**
- Batch processing with progress tracking
- Resume capability (skips already downloaded tickers)
- Individual files: `data/raw/ohlcv/{ticker}_ohlcv.csv`
- Combined file: `data/raw/ohlcv/all_ohlcv_combined.csv`
- Failed tickers logged to `data/raw/ohlcv/failed_tickers.json`

**Resume after interruption:**
```bash
# Automatically resumes from where it left off
python src/data_fetch/fetch_ohlcv.py \
  --tickers data/universe/top_1000_tickers.json \
  --start_date 2020-01-01 \
  --end_date 2025-02-09
```

### 3. Fetch Fundamental Data

```bash
python src/data_fetch/fetch_fundamentals.py \
  --tickers data/universe/top_1000_tickers.json
```

**Fetches:**
- Valuation metrics (P/E, P/B, EV/EBITDA, etc.)
- Profitability (margins, ROE, ROA)
- Growth metrics (revenue growth, earnings growth)
- Financial health (debt ratios, current ratio)
- Dividend metrics
- Sector/industry classification

**Optional: Include quarterly financial statements:**
```bash
python src/data_fetch/fetch_fundamentals.py \
  --tickers data/universe/top_1000_tickers.json \
  --quarterly
```

**Output:**
- `data/raw/fundamentals/fundamentals_current.csv` - Current metrics
- `data/raw/fundamentals/quarterly/{ticker}_quarterly.csv` - Quarterly statements (if requested)

### 4. Fetch Macroeconomic Data

```bash
python src/data_fetch/fetch_macro.py \
  --start_date 2020-01-01 \
  --end_date 2025-02-09
```

**Indicators fetched:**
- Interest rates (Federal Funds, 10Y/2Y Treasury)
- Inflation (CPI, Core CPI, PCE)
- Economic growth (GDP, Industrial Production)
- Employment (Unemployment Rate, Nonfarm Payrolls)
- Sentiment (Consumer Sentiment, Retail Sales)
- Market indicators (VIX, Oil Prices)
- Money supply (M2, Consumer Credit)

**With derived features:**
```bash
python src/data_fetch/fetch_macro.py \
  --start_date 2020-01-01 \
  --end_date 2025-02-09 \
  --derive_features
```

Creates 1-month and 3-month changes, moving averages for all indicators.

**Output:**
- `data/raw/macro/macro_indicators.csv` - Raw indicators
- `data/raw/macro/macro_indicators_derived.csv` - With derived features

### 5. Fetch News Data (Optional)

**Warning:** Fetching news for 1000 tickers is very time-consuming (days). Consider:
- Running for smaller batches
- Using fewer date ranges
- Running in background

```bash
python src/data_fetch/fetch_news.py
```

Modify the script to customize tickers and date range.

## Output Dataset Structure

The final merged dataset (`data/processed/final_dataset.parquet`) contains:

### Core Columns
- `ticker` - Stock symbol
- `date` - Trading date
- `open`, `high`, `low`, `close`, `volume` - OHLCV data

### Technical Features
- `return_t+1` - Next-day return (target variable)
- `momentum_{5,10,20}` - Price momentum over different windows
- `volatility_{5,10,20}` - Rolling volatility
- `sma_{5,10,20}` - Simple moving averages
- `rsi_14` - Relative Strength Index
- `drawdown` - Current drawdown from peak

### Fundamental Features
- `market_cap`, `enterprise_value`
- `trailing_pe`, `forward_pe`, `peg_ratio`
- `price_to_book`, `price_to_sales`
- `profit_margin`, `operating_margin`, `gross_margin`
- `roe`, `roa`
- `revenue_growth`, `earnings_growth`
- `debt_to_equity`, `current_ratio`
- And many more... (see [fetch_fundamentals.py](src/data_fetch/fetch_fundamentals.py:42-67))

### Macroeconomic Features
- `DFF` - Federal Funds Rate
- `DGS10`, `DGS2` - Treasury rates
- `CPIAUCSL` - Consumer Price Index
- `UNRATE` - Unemployment Rate
- `VIXCLS` - VIX
- And more... (see [fetch_macro.py](src/data_fetch/fetch_macro.py:19-44))

## Performance Considerations

### Time Estimates (for 1000 tickers)

- **Universe building**: ~15-30 minutes
- **OHLCV (5 years)**: ~2-4 hours
- **Fundamentals**: ~1-2 hours
- **Macro**: ~2-5 minutes
- **News (1 year)**: ~24-48 hours (rate limited)

### Recommendations

1. **Start small:** Test with 50-100 tickers first
2. **Run overnight:** OHLCV and fundamentals take time
3. **Use resume:** If interrupted, the system will continue from where it left off
4. **Skip news initially:** Focus on price/fundamentals first, add news later
5. **Batch processing:** The system saves progress every 100 tickers

## Data Quality

### Handling Missing Data

The system logs failed tickers:
- `data/raw/ohlcv/failed_tickers.json`
- `data/raw/fundamentals/failed_tickers.json`

Review these files and decide whether to:
1. Retry failed tickers
2. Remove from universe
3. Use forward-fill for missing values

### Data Validation

After collection, check:

```python
import pandas as pd

df = pd.read_parquet('data/processed/final_dataset.parquet')

# Check completeness
print(f"Shape: {df.shape}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Missing data analysis
print("\nMissing data percentage:")
print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(20))

# Data per ticker
print("\nRows per ticker (should be similar):")
print(df.groupby('ticker').size().describe())
```

## Example: Custom Pipeline

```python
from src.data_fetch.get_universe import get_top_n_equities_by_liquidity
from src.data_fetch.fetch_ohlcv import fetch_ohlcv_batch
from src.data_fetch.fetch_fundamentals import fetch_fundamentals_batch

# 1. Get top 500 by liquidity
universe_df = get_top_n_equities_by_liquidity(n=500)
tickers = universe_df['ticker'].tolist()

# 2. Fetch 2 years of price data
ohlcv_df = fetch_ohlcv_batch(
    tickers=tickers,
    start_date='2023-01-01',
    end_date='2025-02-09'
)

# 3. Fetch fundamentals
fundamentals_df = fetch_fundamentals_batch(tickers=tickers)

# 4. Merge
from src.data_fetch.fetch_fundamentals import merge_fundamentals_with_prices
final_df = merge_fundamentals_with_prices(ohlcv_df, fundamentals_df)

# 5. Save
final_df.to_parquet('data/processed/custom_dataset.parquet')
```

## Next Steps: Modeling

Once you have the data, refer to the ChatGPT research document for modeling approaches:

1. **Feature engineering:** Create cross-sectional features (z-scores, ranks)
2. **Train-test split:** Use time-series cross-validation (purged walk-forward)
3. **Model selection:** Start with tree-based models (XGBoost, LightGBM, Random Forest)
4. **Evaluation:** Use Sharpe ratio, IC, ranking metrics (not just R²)
5. **Portfolio construction:** Long-short or ranking-based strategies
6. **Transaction costs:** Always include realistic costs in backtests

## Troubleshooting

### Rate Limiting
- Add `sleep()` delays between requests
- Use batch processing with pauses

### Memory Issues
- Process tickers in smaller batches
- Use parquet format (more efficient than CSV)
- Clear intermediate DataFrames

### Missing API Keys
```
ValueError: FRED API key not set
```
Solution: Add API key to `.env` file

### yfinance Errors
```
No data returned for {ticker}
```
- Ticker may be delisted or invalid
- Check ticker symbol (use `-` instead of `.`)
- Skip and continue with other tickers

## Support

For issues or questions:
1. Check this guide first
2. Review the ChatGPT research document
3. Examine the source code in `src/data_fetch/`
4. Test with small samples before scaling to 1000 tickers

## Key Files

- [run_pipeline.py](run_pipeline.py) - Main orchestrator
- [get_universe.py](src/data_fetch/get_universe.py) - Universe selection
- [fetch_ohlcv.py](src/data_fetch/fetch_ohlcv.py) - Price data
- [fetch_fundamentals.py](src/data_fetch/fetch_fundamentals.py) - Financial metrics
- [fetch_macro.py](src/data_fetch/fetch_macro.py) - Economic indicators
- [fetch_news.py](src/data_fetch/fetch_news.py) - News articles
- [add_trading_metrics.py](src/utils/add_trading_metrics.py) - Technical features
