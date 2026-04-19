"""
Individual model wrappers with a common interface for stock ranking.

Each ranker produces per-stock scores that can be combined by the ensemble.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """Common interface for all stock ranking models."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseRanker":
        """Fit the model. X_val/y_val used for early stopping where applicable."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw per-stock scores (higher = better predicted return)."""
        ...

    def rank(self, X: pd.DataFrame, month_end: pd.Series) -> np.ndarray:
        """Cross-sectional percentile rank of predict() within each month."""
        scores = self.predict(X)
        tmp = pd.DataFrame({"score": scores, "month_end": month_end.values})
        tmp["rank"] = tmp.groupby("month_end")["score"].rank(pct=True, method="average")
        return tmp["rank"].values


# ---------------------------------------------------------------------------
# XGBoost ranker
# ---------------------------------------------------------------------------
class XGBRanker(BaseRanker):
    """
    XGBoost multi-class classifier predicting return decile bins.

    predict() returns the expected decile (sum of p_i * i for i in 0..9),
    which gives a continuous score suitable for ranking.
    """

    name = "xgb"

    def __init__(
        self,
        params: Optional[Dict] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        num_class: int = 10,
    ):
        import xgboost as xgb  # noqa: F401

        self.num_class = num_class
        defaults = dict(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
        )
        if params:
            defaults.update(params)
        self._params = defaults
        self._early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.best_iteration_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        import xgboost as xgb

        callbacks = []
        if X_val is not None:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self._early_stopping_rounds, save_best=True
                )
            )

        self.model_ = xgb.XGBClassifier(**self._params, callbacks=callbacks)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model_.fit(X, y, eval_set=eval_set, verbose=False)
        self.best_iteration_ = getattr(self.model_, "best_iteration", None)
        return self

    def predict(self, X):
        iter_range = (
            (0, self.best_iteration_) if self.best_iteration_ else None
        )
        proba = self.model_.predict_proba(X, iteration_range=iter_range)
        # Expected decile: continuous score from 0 to num_class-1
        decile_values = np.arange(self.num_class, dtype=np.float32)
        return proba @ decile_values


# ---------------------------------------------------------------------------
# MLP ranker (PyTorch)
# ---------------------------------------------------------------------------
class MLPRanker(BaseRanker):
    """
    Simple feedforward MLP.
    
    Architecture: input -> 256 -> 128 -> 64 -> 1
    Uses BatchNorm, Dropout, ReLU. Trained with MSE on continuous excess returns.
    Features are standard-scaled (fit on train).
    """

    name = "mlp"

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 4096,
        patience: int = 10,
    ):
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model_ = None
        self.scaler_ = None
        self.feature_medians_ = None

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        layers = []
        prev = input_dim
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _impute_and_scale(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        arr = X.values.astype(np.float32)
        if fit:
            self.feature_medians_ = np.nanmedian(arr, axis=0)
            self.scaler_ = StandardScaler()

        # Median impute
        for j in range(arr.shape[1]):
            mask = np.isnan(arr[:, j])
            if mask.any():
                arr[mask, j] = self.feature_medians_[j]

        # Replace any remaining NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            arr = self.scaler_.fit_transform(arr)
        else:
            arr = self.scaler_.transform(arr)
        return arr

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Impute and scale
        X_arr = self._impute_and_scale(X, fit=True)
        y_arr = y.values.astype(np.float32).reshape(-1, 1)
        y_arr = np.nan_to_num(y_arr, nan=0.0)

        train_ds = TensorDataset(
            torch.from_numpy(X_arr), torch.from_numpy(y_arr)
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = self._impute_and_scale(X_val)
            y_val_arr = y_val.values.astype(np.float32).reshape(-1, 1)
            y_val_arr = np.nan_to_num(y_val_arr, nan=0.0)
            val_X_t = torch.from_numpy(X_val_arr).to(device)
            val_y_t = torch.from_numpy(y_val_arr).to(device)

        self.model_ = self._build_model(X_arr.shape[1]).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(val_X_t), val_y_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.model_.to("cpu")
        return self

    def predict(self, X):
        import torch

        X_arr = self._impute_and_scale(X)
        X_t = torch.from_numpy(X_arr)
        with torch.no_grad():
            preds = self.model_(X_t).squeeze(-1).numpy()
        return preds


# ---------------------------------------------------------------------------
# Elastic net ranker
# ---------------------------------------------------------------------------
class ElasticNetRanker(BaseRanker):
    """
    ElasticNetCV linear model. Minimal tuning — sklearn handles the
    regularization path via built-in cross-validation.
    """

    name = "enet"

    def __init__(self, l1_ratios: Optional[List[float]] = None, cv: int = 5):
        self.l1_ratios = l1_ratios or [0.1, 0.5, 0.9]
        self.cv = cv
        self.model_ = None
        self.feature_medians_ = None

    def _impute(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        arr = X.values.astype(np.float64)
        if fit:
            self.feature_medians_ = np.nanmedian(arr, axis=0)
        for j in range(arr.shape[1]):
            mask = np.isnan(arr[:, j])
            if mask.any():
                arr[mask, j] = self.feature_medians_[j]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.linear_model import ElasticNetCV

        X_arr = self._impute(X, fit=True)
        y_arr = y.values.astype(np.float64)
        y_arr = np.nan_to_num(y_arr, nan=0.0)

        self.model_ = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=self.cv,
            n_alphas=20,
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )
        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        X_arr = self._impute(X)
        return self.model_.predict(X_arr).astype(np.float32)
