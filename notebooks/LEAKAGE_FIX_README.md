# Cross-Sectional Data Leakage Fix - CORRECTED

## Problem Identified
The model showed unrealistically high performance due to **temporal leakage** in cross-sectional features.

### Root Cause
The `_csrank` and `_csnorm` features in the GKX parquet file were computed on the **entire dataset** (1993-2026) at once. This means:

- Training data from 2000 has CS ranks computed using statistics from the entire time range
- This creates subtle temporal leakage even though ranks are computed per-month
- Example: median imputation and rank distributions "know" about future periods

### Why This Matters for Temporal CV
Even though CS ranks are computed month-by-month (correct for inference), computing them on the full dataset creates leakage:

```
Dataset: 1993-2026 (all at once)
  ↓
For each month, compute CS ranks across all stocks
  ↓
Then split into train/val/holdout by time

Problem: Training period statistics are influenced by validation/holdout periods!
```

## Solution Implemented

### Corrected Approach
Compute CS features **separately for each time split**:

1. **Training split (e.g., 1993-2005)**: Compute CS ranks on this period only
2. **Validation split (e.g., 2006-2008)**: Compute CS ranks on this period only
3. **Holdout split (e.g., 2009-2011)**: Compute CS ranks on this period only

Within each split, for any given month, rank ALL stocks in that month against each other. This is correct because:
- At inference time for month T+1, you have all month T data available
- The split is temporal, not cross-sectional
- All stocks appear in all periods

### New Functions Added

**`compute_cs_features_by_split(df, split_mask, feature_list)`**
```python
# For a specific time period (train/val/holdout):
split_df = df.loc[split_mask]  # e.g., only 1993-2005 for train

# Group by month and rank within each month
grouped = split_df.groupby("month_end")
cs_rank = grouped[feature].rank(pct=True)
cs_norm = 2.0 * cs_rank - 1.0
```

**`prepare_fold_data_correct(model_df, fold_info, raw_features)`**
- Calls `compute_cs_features_by_split()` separately for train and val
- Returns properly split data with correct CS features

**Updated `evaluate_params_on_folds()`**
- Uses `prepare_fold_data_correct()` instead of pre-computed features
- Computes CS features fresh for each fold

## Example: How It Works

### Fold 0 Structure
```
Train:    1993-01 to 2002-12  (10 years)
Val:      2003-01 to 2005-12  (3 years)
Holdout:  2006-01 to 2008-12  (3 years)
```

### Training CS Features (2000-01-31)
```python
# Only use training period (1993-2002)
train_data_jan_2000 = df.loc[train_mask & (month == 2000-01-31)]

# Rank all stocks in this month against each other
stocks = ['AAPL', 'MSFT', 'GOOGL', ...]  # All stocks in Jan 2000
mom12m_values = [0.10, 0.25, 0.15, ...]
mom12m_csrank = rank(mom12m_values)  # Percentile ranks

# This is correct: at inference, you'd have all Jan 2000 data
```

### Validation CS Features (2006-01-31)
```python
# Only use validation period (2003-2005) - INDEPENDENT from training
val_data_jan_2006 = df.loc[val_mask & (month == 2006-01-31)]

# Rank all stocks using validation period statistics
# NOT using training period statistics!
stocks = ['AAPL', 'MSFT', 'GOOGL', ...]
mom12m_values = [0.12, 0.22, 0.18, ...]
mom12m_csrank = rank(mom12m_values)  # Different distribution!
```

**Key**: The validation CS features are computed independently using only validation period data.

## Testing the Fix

### Quick Test
```python
# Test on one fold with corrected CS features
test_params = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}

print("WITH corrected CS features (computed per split):")
results = evaluate_params_on_folds(test_params, [folds[0]], use_raw_features_only=False)
print(f"  Spread Sharpe: {score_fold_spreads(results['fold_spreads']):.3f}")

print("\nRAW features only (no CS transforms):")
results_raw = evaluate_params_on_folds(test_params, [folds[0]], use_raw_features_only=True)
print(f"  Spread Sharpe: {score_fold_spreads(results_raw['fold_spreads']):.3f}")
```

### Expected Results
The corrected approach should show:
- Lower but more realistic performance than the leaky version
- Performance may be similar to or slightly better than raw features (CS transforms can help tree models)
- IC in the range of 0.02-0.04 (realistic for stock prediction)

## What Changed

| Aspect | Before (Leaky) | After (Fixed) |
|--------|---------------|---------------|
| CS Feature Computation | On full dataset (1993-2026) | Per split (train/val/holdout separately) |
| Training 2000 CS rank | Uses stats from 1993-2026 | Uses stats from 1993-2002 only |
| Validation 2006 CS rank | Uses stats from 1993-2026 | Uses stats from 2003-2005 only |
| Expected IC | ~0.08 (too high) | ~0.02-0.04 (realistic) |

## Implementation Details

### Feature Count
- Same as before: raw + _csrank + _csnorm for each GKX feature
- ~94 raw + 94 csrank + 94 csnorm = ~282 GKX features
- Plus macro features

### Performance Impact
- **Accuracy**: Lower but correct (removes leakage bias)
- **Training Speed**: ~10-20% slower (computing CS features per fold)
- **Memory**: Slightly higher (temporary DataFrames per split)

### Compatibility
- Phase 1 random search: Works automatically (uses `evaluate_params_on_folds`)
- Phase 2 Optuna: Works automatically (uses `evaluate_params_on_folds`)
- Holdout evaluation: Needs update to use `prepare_fold_data_correct`

## Next Steps

1. ✅ Functions implemented and integrated
2. ⏳ Re-run Phase 1 grid search (cell 11)
3. ⏳ Re-run Phase 2 Optuna (cell 19)
4. ⏳ Update holdout evaluation (cell 28) to use corrected approach
5. ⏳ Compare results and verify IC is in realistic range

## Verification Checklist

- [ ] Test single fold shows different results than before
- [ ] Corrected approach shows lower performance than leaky version
- [ ] IC drops to realistic levels (0.02-0.04)
- [ ] Raw features and corrected CS features have similar performance
- [ ] Re-run full hyperparameter search with corrected features
- [ ] Document final performance metrics
