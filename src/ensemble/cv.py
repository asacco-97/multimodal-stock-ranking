"""
Anchored-expanding cross-validation and hyperparameter tuning utilities.

Designed for monthly stock-panel data where train always starts at the
earliest date and the validation/holdout windows slide forward.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Anchored-expanding CV
# ---------------------------------------------------------------------------
def anchored_expanding_cv(
    df: pd.DataFrame,
    date_col: str = "month_end",
    min_train_months: int = 60,
    val_months: int = 12,
    holdout_months: int = 12,
    step_months: int = 12,
    embargo_months: int = 1,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Yield boolean masks for anchored-expanding train/val/holdout splits.

    The train window always starts at the earliest date. The val and holdout
    windows slide forward by ``step_months`` each iteration. An embargo gap
    separates train from val to prevent information leakage.

    Parameters
    ----------
    df : DataFrame
        Panel data with one row per stock-month.
    date_col : str
        Column containing month-end dates.
    min_train_months : int
        Minimum number of unique months in the first training window.
    val_months : int
        Number of months in the validation window.
    holdout_months : int
        Number of months in the holdout (test) window.
    step_months : int
        Number of months to slide forward between folds.
    embargo_months : int
        Gap months between train end and val start (leak prevention).

    Yields
    ------
    dict with keys:
        fold : int
        train_mask : np.ndarray[bool]
        val_mask : np.ndarray[bool]
        holdout_mask : np.ndarray[bool]
        train_end : pd.Timestamp
        val_start : pd.Timestamp
        val_end : pd.Timestamp
        holdout_start : pd.Timestamp
        holdout_end : pd.Timestamp
    """
    dates = np.sort(df[date_col].unique())
    n_dates = len(dates)

    # We need at least min_train + embargo + val + holdout months
    required = min_train_months + embargo_months + val_months + holdout_months
    if n_dates < required:
        raise ValueError(
            f"Only {n_dates} unique dates but need at least {required} "
            f"(min_train={min_train_months} + embargo={embargo_months} "
            f"+ val={val_months} + holdout={holdout_months})."
        )

    fold = 0
    train_end_idx = min_train_months - 1  # inclusive

    while True:
        val_start_idx = train_end_idx + embargo_months + 1
        val_end_idx = val_start_idx + val_months - 1
        holdout_start_idx = val_end_idx + 1
        holdout_end_idx = holdout_start_idx + holdout_months - 1

        # Stop if holdout extends beyond available data
        if holdout_end_idx >= n_dates:
            break

        train_end = dates[train_end_idx]
        val_start = dates[val_start_idx]
        val_end = dates[val_end_idx]
        holdout_start = dates[holdout_start_idx]
        holdout_end = dates[holdout_end_idx]

        date_vals = df[date_col].values
        train_mask = date_vals <= train_end
        val_mask = (date_vals >= val_start) & (date_vals <= val_end)
        holdout_mask = (date_vals >= holdout_start) & (date_vals <= holdout_end)

        yield {
            "fold": fold,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "holdout_mask": holdout_mask,
            "train_end": pd.Timestamp(train_end),
            "val_start": pd.Timestamp(val_start),
            "val_end": pd.Timestamp(val_end),
            "holdout_start": pd.Timestamp(holdout_start),
            "holdout_end": pd.Timestamp(holdout_end),
        }

        fold += 1
        train_end_idx += step_months

        # Stop if next fold won't have enough room
        next_holdout_end = train_end_idx + embargo_months + 1 + val_months + holdout_months - 1
        if next_holdout_end >= n_dates:
            break


# ---------------------------------------------------------------------------
# Hyperparameter tuning (two-phase)
# ---------------------------------------------------------------------------
def tune_hyperparameters(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    param_grid: List[Dict[str, Any]],
    scorer: Callable[[np.ndarray, np.ndarray], float],
    model_factory: Callable[[Dict[str, Any]], Any],
    date_col: str = "month_end",
    score_col: Optional[str] = None,
    phase1_n_random: int = 25,
    phase1_n_folds: int = 6,
    phase2_top_k: int = 3,
    cv_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Two-phase hyperparameter tuning using anchored-expanding CV.

    Phase 1: Random sample of ``phase1_n_random`` configs evaluated on the
    last ``phase1_n_folds`` folds (fast screening).

    Phase 2: Top ``phase2_top_k`` configs evaluated on ALL folds (full
    evaluation for robust selection).

    Parameters
    ----------
    model_df : DataFrame
        Panel data with features, target, and date column.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column used for model training (e.g. decile bins).
    param_grid : list[dict]
        List of hyperparameter configurations to evaluate.
    scorer : callable(y_true, y_pred) -> float
        Scoring function (higher is better). E.g., information coefficient.
    model_factory : callable(params) -> model
        Factory that returns a model with .fit(X, y, X_val, y_val) and
        .predict(X) methods given a params dict.
    date_col : str
        Date column for CV splitting.
    score_col : str or None
        Column used as y_true for the scorer. If None, uses target_col.
        Use this when the training target differs from the evaluation target
        (e.g. train on decile bins, score IC against continuous returns).
    phase1_n_random : int
        Number of random configs to sample in phase 1.
    phase1_n_folds : int
        Number of (latest) folds to use in phase 1.
    phase2_top_k : int
        Number of top configs to fully evaluate in phase 2.
    cv_kwargs : dict or None
        Extra keyword arguments for anchored_expanding_cv.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        best_params : dict
        best_score : float
        phase1_results : list[dict]  (config, mean_score, fold_scores)
        phase2_results : list[dict]
    """
    cv_kwargs = cv_kwargs or {}
    score_col = score_col or target_col

    # Generate all folds upfront (store masks)
    all_folds = list(anchored_expanding_cv(model_df, date_col=date_col, **cv_kwargs))
    n_folds = len(all_folds)
    if verbose:
        print(f"Generated {n_folds} CV folds.")

    # ---- Phase 1: fast screening on last N folds ----
    rng = np.random.RandomState(42)
    n_sample = min(phase1_n_random, len(param_grid))
    sampled_indices = rng.choice(len(param_grid), size=n_sample, replace=False)
    sampled_configs = [param_grid[i] for i in sampled_indices]

    # Use last phase1_n_folds folds
    phase1_folds = all_folds[-min(phase1_n_folds, n_folds):]

    if verbose:
        print(f"Phase 1: {n_sample} configs x {len(phase1_folds)} folds")

    phase1_results = []
    for i, params in enumerate(sampled_configs):
        fold_scores = _evaluate_config(
            model_df, feature_cols, target_col, score_col, params,
            scorer, model_factory, phase1_folds,
        )
        mean_score = np.mean(fold_scores)
        phase1_results.append({
            "params": params,
            "mean_score": mean_score,
            "fold_scores": fold_scores,
        })
        if verbose:
            print(f"  [{i+1}/{n_sample}] mean_score={mean_score:.4f}")

    # Sort descending by mean score
    phase1_results.sort(key=lambda x: x["mean_score"], reverse=True)

    # ---- Phase 2: full evaluation of top-k ----
    top_configs = phase1_results[:phase2_top_k]
    if verbose:
        print(f"Phase 2: top {phase2_top_k} configs x {n_folds} folds")

    phase2_results = []
    for i, entry in enumerate(top_configs):
        params = entry["params"]
        fold_scores = _evaluate_config(
            model_df, feature_cols, target_col, score_col, params,
            scorer, model_factory, all_folds,
        )
        mean_score = np.mean(fold_scores)
        phase2_results.append({
            "params": params,
            "mean_score": mean_score,
            "fold_scores": fold_scores,
        })
        if verbose:
            print(f"  [{i+1}/{phase2_top_k}] mean_score={mean_score:.4f}")

    phase2_results.sort(key=lambda x: x["mean_score"], reverse=True)

    best = phase2_results[0]
    if verbose:
        print(f"Best score: {best['mean_score']:.4f}")
        print(f"Best params: {best['params']}")

    return {
        "best_params": best["params"],
        "best_score": best["mean_score"],
        "phase1_results": phase1_results,
        "phase2_results": phase2_results,
    }


def _evaluate_config(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    score_col: str,
    params: Dict[str, Any],
    scorer: Callable,
    model_factory: Callable,
    folds: List[Dict],
) -> List[float]:
    """Evaluate a single config across given folds. Returns list of scores.

    Parameters
    ----------
    target_col : str
        Column used for model training.
    score_col : str
        Column used as y_true for the scorer (may differ from target_col).
    """
    scores = []
    for fold_info in folds:
        train_mask = fold_info["train_mask"]
        val_mask = fold_info["val_mask"]

        X_train = model_df.loc[train_mask, feature_cols]
        y_train = model_df.loc[train_mask, target_col]
        X_val = model_df.loc[val_mask, feature_cols]
        y_val = model_df.loc[val_mask, target_col]

        model = model_factory(params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)

        # Score against score_col (e.g. continuous returns for IC)
        y_score = model_df.loc[val_mask, score_col]
        score = scorer(y_score.values, preds)
        scores.append(score)

    return scores
