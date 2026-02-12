# Stock Ranking Model Guide

This guide explains how to use the cross-sectional features for building a stock ranking model.

## Overview

The cross-sectional feature module transforms raw features (price, fundamentals, technical indicators, macro) into **relative metrics** that are comparable across stocks and time periods. This is essential for ranking stocks to identify top/bottom performers.

## Key Concept: Cross-Sectional vs. Time-Series

**Cross-Sectional Modeling** (what we're doing):
- Goal: Rank stocks relative to each other at each point in time
- Question: "Which stocks will outperform their peers?"
- Features: Percentile ranks, z-scores, sector-relative metrics
- Model output: Rankings or quintiles

**Time-Series Modeling** (NOT what we're doing):
- Goal: Predict absolute returns for individual stocks
- Question: "What will AAPL's return be?"
- Less suitable for portfolio construction

## Features Added

The `add_cross_sectional_features.py` module adds the following types of features:

### 1. Percentile Ranks (`*_rank`)
Percentile rank of each feature within each date (0 to 1).

**Example:**
- If a stock has `momentum_60d_rank = 0.95`, it's in the top 5% of momentum stocks that day
- Comparable across time periods despite changing market conditions

### 2. Z-Scores (`*_zscore`)
Standardized values within each time period (mean=0, std=1).

**Example:**
- `pe_ratio_zscore = 2.0` means the P/E ratio is 2 standard deviations above the cross-sectional mean
- Captures how unusual a value is relative to peers

### 3. Sector-Relative Features (`*_sector_rel`)
Value minus sector mean (if sector column provided).

**Example:**
- `roe_sector_rel = 0.05` means ROE is 5% higher than sector average
- Helps identify sector-neutral opportunities

### 4. Sector Ranks (`*_sector_rank`)
Percentile rank within sector.

**Example:**
- `momentum_60d_sector_rank = 0.80` means in top 20% within its sector
- Useful for sector-neutral portfolios

### 5. Quintile Buckets (`*_quintile`)
Categorical buckets (0-4) for key features.

**Example:**
- `momentum_60d_quintile = 4` means in the top quintile
- Good for simple sorting strategies

### 6. Rank Momentum (`*_rank_change_Xd`)
Change in percentile rank over time.

**Example:**
- `momentum_60d_rank_change_20d = 0.30` means rank increased by 30 percentiles in 20 days
- Captures improving/deteriorating relative position

### 7. Target Variable (`target`)
Cross-sectional ranking of forward returns.

**Options:**
- `quintile`: 0 (worst 20%) to 4 (best 20%)
- `decile`: 0 (worst 10%) to 9 (best 10%)
- `binary_top_bottom`: 0 (bottom 15%), 1 (middle 70%), 2 (top 15%)
- `continuous_rank`: Percentile rank of returns (0 to 1)

## Running the Pipeline

### Full Pipeline (with cross-sectional features):

```bash
python run_pipeline.py \
  --n_equities 1000 \
  --start_date 2020-01-01 \
  --end_date 2025-02-09 \
  --steps universe,ohlcv,fundamentals,macro,features,merge,cross_sectional
```

### Skip macro (faster):

```bash
python run_pipeline.py \
  --n_equities 1000 \
  --start_date 2020-01-01 \
  --end_date 2025-02-09 \
  --steps universe,ohlcv,fundamentals,features,merge,cross_sectional
```

### Only add cross-sectional features to existing data:

```bash
python src/modeling/add_cross_sectional_features.py \
  --input data/processed/final_dataset.parquet \
  --output data/processed/modeling_dataset.parquet \
  --target_type quintile
```

## Output

**File:** `data/processed/modeling_dataset.parquet`

**Structure:**
- `ticker`: Stock ticker
- `date`: Date
- `target`: Target variable (quintile/decile/binary)
- Raw features: `close`, `volume`, `momentum_60d`, `pe_ratio`, etc.
- Rank features: `momentum_60d_rank`, `pe_ratio_rank`, etc.
- Z-score features: `momentum_60d_zscore`, `pe_ratio_zscore`, etc.
- Quintile features: `momentum_60d_quintile`, etc.
- Rank momentum: `momentum_60d_rank_change_1d`, etc.

## Modeling Example

### 1. Load Data

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Load modeling dataset
df = pd.read_parquet('data/processed/modeling_dataset.parquet')

# Remove rows with missing target
df = df[df['target'].notna()].copy()

print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Tickers: {df['ticker'].nunique()}")
```

### 2. Define Features

```python
# Use rank and z-score features (not raw values!)
feature_cols = [
    # Rank features - most important for cross-sectional models
    'momentum_60d_rank', 'momentum_20d_rank',
    'volatility_60d_rank', 'rsi_14d_rank',
    'pe_ratio_rank', 'pb_ratio_rank',
    'roe_rank', 'profit_margin_rank',

    # Z-score features
    'momentum_60d_zscore', 'rsi_14d_zscore',
    'pe_ratio_zscore', 'roe_zscore',

    # Quintile features (categorical)
    'momentum_60d_quintile', 'rsi_14d_quintile',

    # Rank momentum (changes in rank)
    'momentum_60d_rank_change_5d', 'momentum_60d_rank_change_20d',

    # You can also use raw features, but ranks are usually better
    'close', 'volume'
]

# Filter to features that exist
feature_cols = [f for f in feature_cols if f in df.columns]

print(f"Using {len(feature_cols)} features")
```

### 3. Train-Test Split (Time-Based)

```python
# Use time-based split (CRITICAL for stock data)
train_end_date = '2024-01-01'
test_start_date = '2024-01-01'

train_df = df[df['date'] < train_end_date].copy()
test_df = df[df['date'] >= test_start_date].copy()

X_train = train_df[feature_cols]
y_train = train_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"Train: {len(train_df)} rows, {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Test:  {len(test_df)} rows, {test_df['date'].min()} to {test_df['date'].max()}")
```

### 4. Train Model (Classification)

```python
# Train LightGBM classifier
model = lgb.LGBMClassifier(
    objective='multiclass',  # For quintile classification
    num_class=5,             # 5 classes (0-4)
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
```

### 5. Evaluate for Long-Short Portfolio

```python
# Add predictions to test set
test_df['predicted_quintile'] = y_pred

# For each date, long top quintile (4), short bottom quintile (0)
monthly_returns = []

for date in test_df['date'].unique():
    date_df = test_df[test_df['date'] == date].copy()

    # Long top predicted quintile
    long_stocks = date_df[date_df['predicted_quintile'] == 4]['return_t+1']

    # Short bottom predicted quintile
    short_stocks = date_df[date_df['predicted_quintile'] == 0]['return_t+1']

    if len(long_stocks) > 0 and len(short_stocks) > 0:
        # Long-short return
        ls_return = long_stocks.mean() - short_stocks.mean()
        monthly_returns.append({
            'date': date,
            'long_return': long_stocks.mean(),
            'short_return': short_stocks.mean(),
            'ls_return': ls_return
        })

returns_df = pd.DataFrame(monthly_returns)
print(f"\nLong-Short Performance:")
print(f"  Mean return: {returns_df['ls_return'].mean():.4f}")
print(f"  Std: {returns_df['ls_return'].std():.4f}")
print(f"  Sharpe: {returns_df['ls_return'].mean() / returns_df['ls_return'].std():.4f}")
```

## Best Practices

### 1. Always Use Time-Based Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(df.groupby('date')):
    # Train on earlier months, test on later months
    train_dates = df.groupby('date').groups.keys()[train_idx]
    test_dates = df.groupby('date').groups.keys()[test_idx]
```

### 2. Focus on Rank Features
Percentile ranks are more stable than raw values and generalize better across market regimes.

### 3. Consider Sector-Neutral Models
If you have sector data, use sector-relative features or train separate models per sector.

### 4. Use Classification, Not Regression
Predicting quintiles/deciles is more robust than predicting exact returns.

### 5. Evaluate on Portfolio Returns
Don't just look at accuracy - simulate actual portfolio construction and measure returns.

## Target Variable Selection

**Quintile (recommended to start):**
- 5 classes, easiest to interpret
- "Top 20% vs Bottom 20%" is a clear strategy

**Decile:**
- 10 classes, more granular
- Good if you want top/bottom 10%

**Binary Top/Bottom:**
- 3 classes: bottom 15%, middle 70%, top 15%
- Focus on extreme performers
- Good for high-conviction portfolios

**Continuous Rank:**
- Regression on percentile rank
- Most information preserved
- Harder to interpret

## Next Steps

1. **Experiment with features**: Try different combinations of rank, z-score, and sector-relative features
2. **Tune hyperparameters**: Use cross-validation to optimize model parameters
3. **Add more signals**: Incorporate macro regime indicators, sentiment, etc.
4. **Backtest carefully**: Account for transaction costs, slippage, and realistic execution
5. **Monitor performance**: Track model degradation over time and retrain regularly

## Common Issues

**Issue:** Model accuracy is low (~20%)
**Solution:** For quintile classification, random guessing is 20%. Focus on portfolio returns, not accuracy.

**Issue:** Model works in-sample but not out-of-sample
**Solution:** Check for look-ahead bias. Ensure you're using time-based splits and only ranking within each date.

**Issue:** Some features have many NaN values
**Solution:** This is normal for fundamentals (quarterly data). Use forward-fill or drop these features.

**Issue:** Performance degrades over time
**Solution:** Retrain regularly with recent data. Market regimes change.

## Resources

- Example notebook: `notebooks/model_training_example.ipynb` (create this)
- Feature importance: Analyze which rank features matter most
- Portfolio simulation: Backtest with realistic constraints
