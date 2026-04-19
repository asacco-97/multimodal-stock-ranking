"""
Ensemble combiner for stock ranking models.

Combines per-stock scores from multiple rankers using rank-averaging
or learned stacking.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .rankers import BaseRanker


class StockRankerEnsemble:
    """
    Combine multiple BaseRanker models into a single scoring pipeline.

    Methods:
        rank_average: Cross-sectional rank each model's scores within each
            month, then average ranks. Robust to scale differences.
        stacking: Train an XGBoost meta-learner (max_depth=2) on the
            per-model rank columns. Requires OOF predictions.
    """

    def __init__(
        self,
        rankers: List[BaseRanker],
        method: str = "rank_average",
    ):
        if not rankers:
            raise ValueError("Must provide at least one ranker.")
        self.rankers = rankers
        self.method = method
        self.meta_model_ = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        month_end_train: pd.Series,
        month_end_val: pd.Series,
    ) -> "StockRankerEnsemble":
        """
        Fit all sub-models. If method='stacking', also trains the meta-model
        on validation-set rank columns.
        """
        for ranker in self.rankers:
            print(f"  Fitting {ranker.name}...")
            ranker.fit(X_train, y_train, X_val, y_val)

        if self.method == "stacking":
            self._fit_meta(X_val, y_val, month_end_val)

        return self

    def _fit_meta(self, X_val, y_val, month_end_val):
        """Train XGBoost meta-learner on per-model rank columns."""
        import xgboost as xgb

        rank_df = self._build_rank_df(X_val, month_end_val)
        meta_features = [f"{r.name}_rank" for r in self.rankers]

        self.meta_model_ = xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=2,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=42,
        )
        self.meta_model_.fit(rank_df[meta_features], y_val.values)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame, month_end: pd.Series) -> np.ndarray:
        """
        Return combined ensemble score (higher = better predicted return).
        """
        rank_df = self._build_rank_df(X, month_end)
        meta_features = [f"{r.name}_rank" for r in self.rankers]

        if self.method == "stacking" and self.meta_model_ is not None:
            return self.meta_model_.predict(rank_df[meta_features])

        # Default: simple rank average
        return rank_df[meta_features].mean(axis=1).values

    def predict_with_components(
        self, X: pd.DataFrame, month_end: pd.Series
    ) -> pd.DataFrame:
        """
        Return DataFrame with individual model raw scores, ranks, and
        combined ensemble score. Useful for diagnostics.
        """
        rank_df = self._build_rank_df(X, month_end)
        rank_df["ensemble_score"] = self.predict(X, month_end)
        return rank_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_rank_df(
        self, X: pd.DataFrame, month_end: pd.Series
    ) -> pd.DataFrame:
        """Get per-model raw scores and cross-sectional ranks."""
        out = pd.DataFrame({"month_end": month_end.values})
        for ranker in self.rankers:
            raw = ranker.predict(X)
            out[f"{ranker.name}_raw"] = raw
            out[f"{ranker.name}_rank"] = (
                out.groupby("month_end")[f"{ranker.name}_raw"]
                .rank(pct=True, method="average")
            )
        return out


# # ---------------------------------------------------------------------------
# # Stacking variant (uncomment to use)
# # ---------------------------------------------------------------------------
# # To switch from rank_average to learned stacking:
# #
# #   ensemble = StockRankerEnsemble(
# #       rankers=[xgb_ranker, mlp_ranker, enet_ranker],
# #       method="stacking",
# #   )
# #   ensemble.fit(X_train, y_train, X_val, y_val, me_train, me_val)
# #
# # The meta-model (XGBoost, max_depth=2) learns optimal weights and
# # interactions between model rank columns. It is trained on the
# # validation set's rank columns with actual returns as the target.
# #
# # For production use with proper leakage prevention, generate
# # out-of-fold predictions using anchored_expanding_cv and train the
# # meta-model on those instead of the validation set directly.
