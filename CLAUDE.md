# CLAUDE.md — multimodal-stock-ranking

## Project Overview

A cross-sectional stock ranking system that combines quarterly fundamentals (SEC EDGAR, point-in-time), price/technical features, GKX-94 factor proxies, macroeconomic indicators, and optionally news embeddings (FinBERT) to rank ~1000 liquid US equities by expected relative performance.

**Key framing**: This is a *ranking* model (which stocks outperform peers this month?), not an absolute return predictor.

---

## Directory Structure

```
├── run_pipeline.py              # Main entry point — orchestrates all pipeline steps
├── src/
│   ├── data_fetch/              # Data ingestion (EDGAR, yfinance, FRED, Finnhub)
│   │   ├── get_universe.py      # Top N by dollar-volume liquidity (NASDAQ official data)
│   │   ├── fetch_ohlcv.py       # Price/volume data via yfinance
│   │   ├── fetch_fundamentals_quarterly.py  # Point-in-time quarterly financials from SEC EDGAR
│   │   ├── fetch_macro.py       # FRED macroeconomic indicators
│   │   ├── fetch_news.py        # News headlines via Finnhub API
│   │   └── build_daily_dataset.py  # Merges all sources
│   ├── embeddings/
│   │   └── embed_news.py        # FinBERT 768-dim embeddings + sentiment scores
│   ├── features/
│   │   ├── build_gkx_characteristics.py  # GKX-94 proxy factor construction
│   │   └── gkx_registry.py      # GKX feature definitions and metadata
│   ├── modeling/
│   │   └── add_cross_sectional_features.py  # Per-date rank, z-score, quintile transforms
│   ├── ensemble/                # Ensemble strategies and corrected CV (new module)
│   │   ├── ensemble.py          # Combination strategies
│   │   ├── rankers.py           # Individual ranking model implementations
│   │   ├── regime.py            # Market regime detection
│   │   └── cv.py                # Leakage-corrected cross-validation utilities
│   ├── backtest/                # Portfolio simulation and performance metrics
│   │   ├── engine.py
│   │   ├── strategies.py
│   │   ├── metrics.py           # Sharpe, IC, Sortino, etc.
│   │   └── plots.py
│   └── utils/
│       ├── purged_cv.py         # Purged walk-forward cross-validation
│       ├── target_builder.py    # Target variable construction (quintiles, returns)
│       ├── add_trading_metrics.py  # RSI, momentum, volatility indicators
│       └── feature_engineering.py
├── notebooks/
│   ├── 002b_model_tuning.ipynb        # Primary XGBoost/LightGBM tuning
│   ├── 003_mlp_tuning.ipynb           # Deep learning experiments
│   ├── 004_gkx_missingness_diagnosis.ipynb
│   ├── 005_backtest_examples.ipynb
│   └── LEAKAGE_FIX_README.md          # CRITICAL: documents cross-sectional leakage fix
├── data/
│   ├── raw/                     # Raw downloads (OHLCV, fundamentals, macro, news)
│   ├── processed/               # Merged, model-ready datasets
│   ├── universe/                # Ticker lists and metadata
│   └── cache/                   # SEC CIK mappings and other API caches
├── DATA_COLLECTION_GUIDE.md
├── FUNDAMENTALS_GUIDE.md
├── MODELING_GUIDE.md
└── QUICK_START.md
```

---

## Running the Pipeline

### Full run
```bash
python run_pipeline.py \
  --n_equities 1000 \
  --start_date 2020-01-01 \
  --end_date 2025-02-09 \
  --steps universe,ohlcv,fundamentals,macro,features,merge,gkx_features,cross_sectional
```

### Quick test (small sample)
```bash
python run_pipeline.py \
  --n_equities 50 \
  --start_date 2023-01-01 \
  --end_date 2025-02-09 \
  --steps universe,ohlcv,fundamentals,features,merge
```

### Individual steps (can run standalone)
```bash
python src/data_fetch/get_universe.py
python src/data_fetch/fetch_fundamentals_quarterly.py
python src/modeling/add_cross_sectional_features.py \
  --input data/processed/final_dataset.parquet \
  --output data/processed/modeling_dataset.parquet \
  --target_type quintile
```

Outputs land in `data/processed/{timestamp}/modeling_dataset_monthly.parquet`.

---

## API Keys

Stored in `.env` at project root:
```
FINNHUB_API_KEY=your_key
FRED_API_KEY=your_key
```

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| NASDAQ official files | Universe of all US stocks |
| yfinance | OHLCV, sector/industry metadata |
| SEC EDGAR API | Quarterly financials with actual filing dates (point-in-time) |
| FRED | ~20 macro indicators (rates, inflation, VIX, employment) |
| Finnhub | News headlines (~5 years) |
| FinBERT (HuggingFace) | 768-dim news embeddings + sentiment |

---

## CRITICAL: Look-Ahead Bias Rules

This is the most important constraint in the project. Violating it produces unrealistically high backtest metrics.

### 1. Point-in-time fundamentals
- Uses **actual SEC filing dates**, not quarter-end dates (there is a 45-90 day lag)
- Merge: `pd.merge_asof(direction='backward')` — for date `t`, only use fundamentals where `report_date <= t`
- Current valuation columns (price-dependent) are stripped from the final dataset (see `run_pipeline.py` lines 33-53)

### 2. Cross-sectional feature leakage — FIXED (see `notebooks/LEAKAGE_FIX_README.md`)
- **Problem**: Computing percentile ranks over the full dataset before train/val split leaks future information into training
- **Solution**: Ranks/z-scores must be computed *per split* — training dates only for train, validation dates separately
- **Symptom of leakage**: IC ~0.08+. Realistic IC is ~0.02–0.04
- **Corrected functions**: `prepare_fold_data_correct`, `compute_cs_features_by_split` in `src/ensemble/cv.py`

### 3. General rules
- All forward returns use `pd.shift()` before rolling windows
- Technical indicators are backward-looking only
- Fundamentals snapped to month-end for monthly models
- Purged walk-forward CV with gaps between train/val splits (`src/utils/purged_cv.py`)

---

## Feature Conventions

| Pattern | Meaning |
|---------|---------|
| `momentum_60d`, `profit_margin` | Raw features (lowercase, underscore) |
| `{feature}_rank` | Cross-sectional percentile rank |
| `{feature}_zscore` | Cross-sectional z-score |
| `{feature}_sector_rel` | Sector-relative value |
| `{feature}_quintile` | Quintile bucket (0–4) |

**~250–300+ features per row**: 94 GKX proxies, ~188 cross-sectional transforms, ~20 macro, ~15 technical, optional 768-dim news embeddings.

---

## Modeling Conventions

- **Monthly model preferred** — use month-end snapshots of GKX features
- **Target**: quintile (0–4) or decile (0–9) of forward 1-month returns
- **Prefer rank features** over raw values — more stable across market regimes
- **Primary models**: XGBoost, LightGBM (see `notebooks/002b_model_tuning.ipynb`)
- **CV**: Purged walk-forward only — never random shuffle or standard k-fold

---

## Tech Stack

- Python 3.8+, pandas, numpy, pyarrow
- scikit-learn, xgboost, lightgbm
- transformers, torch (FinBERT)
- yfinance, fredapi, tqdm, ta

---

## Coding Patterns

- **Lazy init**: expensive models (FinBERT) initialized once and cached globally
- **Batch + resume**: pipeline checks progress files and skips completed steps; `failed_tickers.json` for retries
- **Memory**: `_downcast_float64()` converts float64→float32 for wide panels; explicit `gc.collect()` between heavy steps
- **Storage**: Parquet preferred over CSV for processed data
- **API rate limiting**: exponential backoff for SEC EDGAR (200ms between requests), yfinance auto-retry

---

## Common Debugging

| Symptom | Likely cause |
|---------|-------------|
| IC > 0.06 | Cross-sectional leakage — re-read `LEAKAGE_FIX_README.md` |
| Fundamentals all NaN early in period | Expected — no data before IPO/filing date |
| "No CIK found for {ticker}" | Delisted or alternate symbol; check `data/cache/sec_cik_mapping.json` |
| OOM on 1000 stocks | Reduce `--n_equities` or shorten date range |
| yfinance errors | Rate limiting — system auto-retries; reduce `--n_workers` if needed |

---

## Key Files to Read First

1. `run_pipeline.py` — full workflow overview
2. `notebooks/LEAKAGE_FIX_README.md` — temporal leakage fix (read before touching CV or CS features)
3. `FUNDAMENTALS_GUIDE.md` — point-in-time fundamentals concepts
4. `MODELING_GUIDE.md` — cross-sectional ranking framework
5. `src/ensemble/cv.py` — corrected CV logic
