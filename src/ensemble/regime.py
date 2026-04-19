"""
Market regime detector.

Fits a Gaussian HMM (preferred) or Gaussian Mixture Model (fallback) on
macro features (VIX, yield curve, Fed Funds rate, realized volatility) to
classify market states. States are relabeled so that 0 = bull (lowest mean
VIX), highest = bear.

Output plugs directly into RegimeConditionalStrategy via a `regime_pred`
column in the backtest DataFrame.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ["VIXCLS", "T10Y2Y", "DFF", "spy_retvol_21d"]

# Check hmmlearn availability at import time
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM  # noqa: F401

    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector for market-level state classification.

    Falls back to sklearn GaussianMixture if hmmlearn is not installed
    (loses temporal transition structure but preserves state clustering).

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3: bull / transition / bear).
    features : list[str] or None
        Macro columns to use. Defaults to VIX, 10Y-2Y spread, Fed Funds
        rate, and 21-day realized SPY volatility.
    n_iter : int
        Maximum EM iterations for fitting.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 3,
        features: Optional[List[str]] = None,
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.features = features or list(DEFAULT_FEATURES)
        self.n_iter = n_iter
        self.random_state = random_state

        self.model_ = None
        self.scaler_ = None
        self.state_order_ = None  # mapping: model label -> canonical label
        self.state_stats_ = None
        self._backend = None  # "hmm" or "gmm"

    # ------------------------------------------------------------------
    def fit(self, macro_df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit regime model on monthly macro observations.

        Parameters
        ----------
        macro_df : DataFrame
            Must contain `month_end` column and all columns in self.features.
            Should be one row per month_end (deduplicate beforehand).
        """
        from sklearn.preprocessing import StandardScaler

        df = self._prepare(macro_df)
        X = df[self.features].values.astype(np.float64)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        if _HAS_HMMLEARN:
            from hmmlearn.hmm import GaussianHMM

            self._backend = "hmm"
            self.model_ = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model_.fit(X_scaled)
            raw_states = self.model_.predict(X_scaled)
        else:
            from sklearn.mixture import GaussianMixture

            self._backend = "gmm"
            self.model_ = GaussianMixture(
                n_components=self.n_states,
                covariance_type="full",
                max_iter=self.n_iter,
                random_state=self.random_state,
                n_init=3,
            )
            self.model_.fit(X_scaled)
            raw_states = self.model_.predict(X_scaled)

        # Relabel states: sort by mean VIX (first feature assumed to be VIX
        # or a volatility proxy). Lowest VIX mean = bull (state 0).
        vix_idx = 0  # first feature = VIXCLS by convention
        state_vix_means = {}
        for s in range(self.n_states):
            mask = raw_states == s
            if mask.any():
                state_vix_means[s] = X[mask, vix_idx].mean()
            else:
                state_vix_means[s] = float("inf")

        sorted_states = sorted(state_vix_means, key=state_vix_means.get)
        self.state_order_ = {old: new for new, old in enumerate(sorted_states)}

        # Store state statistics for summary()
        self._compute_state_stats(X, raw_states, df["month_end"].values)

        return self

    # ------------------------------------------------------------------
    def predict(self, macro_df: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for each month_end.

        Returns
        -------
        Series indexed by month_end with integer labels
        (0 = bull, n_states-1 = bear).
        """
        df = self._prepare(macro_df)
        X = df[self.features].values.astype(np.float64)
        X_scaled = self.scaler_.transform(X)
        raw = self.model_.predict(X_scaled)
        canonical = np.array([self.state_order_[s] for s in raw])
        return pd.Series(canonical, index=df["month_end"].values, name="regime_pred")

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarizing each state: feature means and
        the fraction of months spent in each state.
        """
        if self.state_stats_ is None:
            raise RuntimeError("Call fit() before summary().")
        return self.state_stats_

    def transition_matrix(self) -> pd.DataFrame:
        """Return the learned transition probability matrix (HMM only)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before transition_matrix().")
        if self._backend != "hmm":
            raise RuntimeError(
                "Transition matrix is only available with hmmlearn backend. "
                "Install hmmlearn for HMM support."
            )
        n = self.n_states
        # Reorder rows/cols to match canonical labeling
        inv = {v: k for k, v in self.state_order_.items()}
        reorder = [inv[i] for i in range(n)]
        mat = self.model_.transmat_[np.ix_(reorder, reorder)]
        labels = [f"state_{i}" for i in range(n)]
        return pd.DataFrame(mat, index=labels, columns=labels)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _prepare(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input DataFrame."""
        missing = [f for f in self.features if f not in macro_df.columns]
        if missing:
            raise ValueError(f"Missing features in macro_df: {missing}")
        if "month_end" not in macro_df.columns:
            raise ValueError("macro_df must have a 'month_end' column.")

        df = macro_df[["month_end"] + self.features].copy()
        df = df.sort_values("month_end").drop_duplicates("month_end")
        df = df.ffill().bfill()
        # Replace remaining NaN with 0 (shouldn't happen after ffill/bfill)
        df[self.features] = df[self.features].fillna(0)
        return df.reset_index(drop=True)

    def _compute_state_stats(
        self, X: np.ndarray, raw_states: np.ndarray, month_ends: np.ndarray
    ):
        """Compute per-state summary statistics."""
        rows = []
        n_total = len(raw_states)
        for raw_s, canonical_s in sorted(self.state_order_.items(), key=lambda kv: kv[1]):
            mask = raw_states == raw_s
            count = mask.sum()
            row = {
                "state": canonical_s,
                "label": ["bull", "transition", "bear"][canonical_s]
                if self.n_states == 3
                else f"state_{canonical_s}",
                "n_months": int(count),
                "pct_months": float(count / n_total * 100),
            }
            for j, feat in enumerate(self.features):
                row[f"mean_{feat}"] = float(X[mask, j].mean()) if mask.any() else np.nan
            rows.append(row)
        self.state_stats_ = pd.DataFrame(rows)
