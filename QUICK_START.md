# Quick Start - Top 1000 Equities Data Collection

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** in `.env`:
   ```
   FINNHUB_API_KEY=your_key_here
   FRED_API_KEY=your_key_here
   ```
   - Finnhub (free): https://finnhub.io/register
   - FRED (free): https://fred.stlouisfed.org/docs/api/api_key.html

## Three Ways to Run

### 1. Full Pipeline (Recommended for Production)

```bash
python run_pipeline.py --n_equities 1000 --start_date 2020-01-01 --end_date 2025-02-09
```

**What it does:**
- ✅ Identifies top 1000 most liquid US stocks
- ✅ Downloads 5 years of price data
- ✅ Fetches fundamental metrics (P/E, margins, etc.)
- ✅ Gets macroeconomic indicators (GDP, inflation, etc.)
- ✅ Adds technical features (momentum, RSI, etc.)
- ✅ Merges everything into one dataset
- ❌ Skips news (too slow for 1000 stocks)

**Output:** `data/processed/final_dataset.parquet`

**Time:** ~4-6 hours for 1000 stocks

### 2. Test with Small Sample (Recommended First)

```bash
python run_pipeline.py --n_equities 50 --start_date 2023-01-01 --end_date 2025-02-09
```

**What it does:** Same as above but with only 50 stocks and 2 years of data

**Time:** ~20-30 minutes

### 3. Step-by-Step (Advanced)

Run individual components:

```bash
# Step 1: Build universe
python src/data_fetch/get_universe.py

# Step 2: Fetch price data
python src/data_fetch/fetch_ohlcv.py \
  --tickers data/universe/top_1000_tickers.json \
  --start_date 2020-01-01 \
  --end_date 2025-02-09

# Step 3: Fetch fundamentals
python src/data_fetch/fetch_fundamentals.py \
  --tickers data/universe/top_1000_tickers.json

# Step 4: Fetch macro data
python src/data_fetch/fetch_macro.py \
  --start_date 2020-01-01 \
  --end_date 2025-02-09

# Step 5: Merge and add features
python run_pipeline.py --steps features,merge
```

## What You'll Get

Final dataset with **~250 features** per stock-date:

| Category | Examples | Count |
|----------|----------|-------|
| **Price** | open, high, low, close, volume | 5 |
| **Technical** | momentum, RSI, volatility, SMA | ~15 |
| **Fundamental** | P/E, profit margin, debt/equity | ~40 |
| **Macro** | interest rates, inflation, VIX | ~20 |
| **News** (optional) | FinBERT embeddings | 768 |

## Resume After Interruption

The system automatically saves progress. If interrupted, just run the same command again:

```bash
# This will skip already downloaded stocks
python run_pipeline.py --n_equities 1000 --start_date 2020-01-01 --end_date 2025-02-09
```

Progress files:
- `data/raw/ohlcv/download_progress.json`
- `data/raw/ohlcv/failed_tickers.json`

## Validate Data

```python
import pandas as pd

# Load final dataset
df = pd.read_parquet('data/processed/final_dataset.parquet')

# Quick checks
print(f"Shape: {df.shape}")
print(f"Stocks: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSample:\n{df.head()}")

# Check missing data
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(f"\nTop 10 features with missing data:\n{missing_pct.head(10)}")
```

## Common Issues

### 1. "FRED API key not set"
→ Add `FRED_API_KEY=your_key` to `.env` file

### 2. Some tickers failed to download
→ Normal! Check `data/raw/ohlcv/failed_tickers.json`. Delisted stocks or data issues are common.

### 3. Out of memory
→ Reduce `--n_equities` or process in batches

### 4. Too slow
→ Start with `--n_equities 100` and `--start_date 2023-01-01`

## Next Steps

1. **Explore the data:**
   ```python
   import pandas as pd
   df = pd.read_parquet('data/processed/final_dataset.parquet')
   df.info()
   df.describe()
   ```

2. **Create features:**
   - Cross-sectional ranks
   - Industry-adjusted metrics
   - Factor scores

3. **Build models:**
   - Start with XGBoost/LightGBM
   - Use time-series cross-validation
   - Focus on ranking/classification, not regression

4. **Backtest strategies:**
   - Long-short portfolios
   - Sector rotation
   - Risk-adjusted returns

## Full Documentation

See [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) for detailed information.

## File Structure

```
data/
├── universe/
│   ├── top_1000_equities_by_liquidity.csv
│   └── top_1000_tickers.json
├── raw/
│   ├── ohlcv/
│   │   ├── {ticker}_ohlcv.csv (individual files)
│   │   └── all_ohlcv_combined.csv
│   ├── fundamentals/
│   │   └── fundamentals_current.csv
│   ├── macro/
│   │   └── macro_indicators.csv
│   └── news/
│       └── {ticker}_{date}.json
└── processed/
    └── final_dataset.parquet  ← Your modeling dataset
```
