# Point-in-Time Fundamentals Guide

## The Look-Ahead Bias Problem

**Problem**: Using current fundamentals to predict historical returns creates severe look-ahead bias.

### Example of the Problem:
```python
# WRONG: This creates look-ahead bias
fundamentals = yf.Ticker("AAPL").info  # Gets 2025 data
merge_to_all_dates(fundamentals)        # Uses 2025 P/E to predict 2020 returns!
```

**Why this is wrong:**
- You're using information from the future to predict the past
- P/E ratios, margins, growth rates change every quarter
- In real trading, you only have access to the most recently **reported** quarterly data

## The Solution: Point-in-Time Fundamentals

Our implementation ensures that for any date `t`, we only use fundamentals that were **available** (reported) before that date.

### Key Components:

#### 1. **Quarterly Fundamentals**
- Fetch actual quarterly financial statements (income statement, balance sheet, cash flow)
- Each quarter has a `quarter_end_date` and a `report_date`
- Example: Q3 2024 ends Sept 30, but is reported Nov 15 (45-day lag for 10-Q filing)

#### 2. **Reporting Lag**
- Companies have 45 days after quarter end to file 10-Q (quarterly report)
- Companies have 90 days after fiscal year end to file 10-K (annual report)
- We use a conservative 45-day lag: `report_date = quarter_end_date + 45 days`

#### 3. **As-Of Merge**
- Use `pd.merge_asof()` to join fundamentals based on availability
- For each date `t`, we get the most recent fundamentals where `report_date <= t`
- This simulates real-world conditions

### Example Timeline:

```
Quarter End: 2024-09-30 (Q3 2024)
Report Date: 2024-11-15 (45 days later)

On 2024-11-01: Use Q2 2024 fundamentals (most recent available)
On 2024-11-20: Use Q3 2024 fundamentals (now available)
On 2024-12-15: Use Q3 2024 fundamentals (still most recent)
On 2025-02-20: Use Q4 2024 fundamentals (now available)
```

## Pipeline Usage

### Fetch Quarterly Fundamentals

```bash
# Fetch quarterly fundamentals with 45-day reporting lag
python run_pipeline.py \
  --n_equities 1000 \
  --start_date 2020-01-01 \
  --end_date 2025-02-09 \
  --steps fundamentals
```

This will:
1. Fetch quarterly financial statements for each stock
2. Calculate fundamental ratios (profit margin, ROE, ROA, etc.)
3. Add 45-day reporting lag to each quarter
4. Save to `data/raw/fundamentals_quarterly/quarterly_fundamentals.parquet`

### Merge with Point-in-Time Logic

```bash
# Merge all data sources (including point-in-time fundamentals)
python run_pipeline.py \
  --n_equities 1000 \
  --start_date 2020-01-01 \
  --end_date 2025-02-09 \
  --steps merge
```

The merge step uses `merge_fundamentals_point_in_time()` which:
1. Sorts price data by ticker and date
2. Sorts fundamentals by ticker and report_date
3. Uses `pd.merge_asof()` with `direction='backward'`
4. For each ticker/date, gets the most recent fundamentals where `report_date <= date`

## Data Structure

### Quarterly Fundamentals DataFrame

| Column | Description |
|--------|-------------|
| `ticker` | Stock ticker symbol |
| `quarter_end_date` | Last day of the fiscal quarter |
| `report_date` | Date when data became available (quarter_end + lag) |
| `revenue` | Total revenue for the quarter |
| `net_income` | Net income for the quarter |
| `operating_income` | Operating income for the quarter |
| `total_assets` | Total assets (balance sheet) |
| `total_equity` | Shareholders' equity |
| `total_debt` | Total debt |
| `total_cash` | Cash and equivalents |
| `profit_margin` | Net income / revenue |
| `operating_margin` | Operating income / revenue |
| `gross_margin` | Gross profit / revenue |
| `roe` | Return on equity |
| `roa` | Return on assets |
| `debt_to_equity` | Total debt / total equity |
| `current_ratio` | Current assets / current liabilities |
| `quick_ratio` | (Current assets - inventory) / current liabilities |
| `revenue_growth_qoq` | Quarter-over-quarter revenue growth |
| `revenue_growth_yoy` | Year-over-year revenue growth |
| `earnings_growth_qoq` | Quarter-over-quarter earnings growth |
| `earnings_growth_yoy` | Year-over-year earnings growth |

### After Point-in-Time Merge

Your modeling dataset will have:
- All the columns from price/technical data (close, volume, momentum, RSI, etc.)
- All the columns from fundamentals (most recent available as of each date)
- `quarter_end_date` column showing which quarter the fundamentals are from
- `report_date` column showing when that quarter became available

## Validating No Look-Ahead Bias

To verify the merge worked correctly:

```python
import pandas as pd

df = pd.read_parquet('data/processed/final_dataset.parquet')

# Check: report_date should always be <= date
assert (df['report_date'] <= df['date']).all(), "Look-ahead bias detected!"

# Check: Show which quarter is being used for each date
sample = df[df['ticker'] == 'AAPL'].sort_values('date')[['date', 'quarter_end_date', 'report_date', 'revenue', 'profit_margin']].head(20)
print(sample)

# You should see that fundamentals change infrequently (once per quarter)
# and only AFTER the report_date
```

## Handling Missing Fundamentals

Some stocks may have missing fundamentals for early dates (before they IPO'd or if delisted):

```python
# Check coverage
coverage = df['quarter_end_date'].notna().sum() / len(df) * 100
print(f"Fundamentals coverage: {coverage:.1f}%")

# Drop rows without fundamentals if needed for modeling
df_modeling = df[df['quarter_end_date'].notna()].copy()
```

## Benefits of This Approach

1. **No Look-Ahead Bias**: Only use data that was actually available at prediction time
2. **Realistic Backtesting**: Simulates real-world trading conditions
3. **Better Generalization**: Model learns from truly out-of-sample relationships
4. **Production-Ready**: Same logic works in live trading (just use most recent 10-Q/10-K)

## Customizing Reporting Lag

You can adjust the reporting lag based on your assumptions:

```python
# Conservative (60 days): Some companies file late
fundamentals_df = fetch_all_quarterly_fundamentals(
    tickers,
    reporting_lag_days=60
)

# Aggressive (30 days): Assume you process filings quickly
fundamentals_df = fetch_all_quarterly_fundamentals(
    tickers,
    reporting_lag_days=30
)
```

## Advanced: Actual Filing Dates

For production systems, you can fetch actual 10-Q/10-K filing dates from SEC EDGAR:
- Use SEC API to get filing dates
- Replace our estimated `report_date` with actual filing dates
- This gives the most accurate point-in-time data

Example:
```python
# Pseudo-code for fetching actual filing dates
from sec_api import QueryApi

query = QueryApi(api_key="YOUR_KEY")
filings = query.get_filings(ticker="AAPL", form_type="10-Q")
for filing in filings:
    actual_report_date = filing['filedAt']
    # Use this instead of quarter_end_date + 45 days
```

## Common Issues

**Issue**: All fundamentals are NaN for early dates
**Solution**: This is expected - stocks don't have fundamentals before their IPO. Filter by `quarter_end_date.notna()`.

**Issue**: Fundamentals don't change every day
**Solution**: This is correct! Fundamentals only update once per quarter (every ~90 days).

**Issue**: Some ratios seem delayed
**Solution**: This is the point! In real trading, you only get quarterly updates, not daily updates.

## Next Steps

1. Run the pipeline with quarterly fundamentals
2. Verify no look-ahead bias using the validation code above
3. Use these fundamentals in your ranking model
4. Compare model performance with/without point-in-time handling
