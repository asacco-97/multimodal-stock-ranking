import numpy as np
import pandas as pd
from datetime import timedelta


class PurgedTimeSeriesSplit:
    """
    Time-series cross-validator with purge and embargo gaps.

    Splits data into n_splits sequential validation folds. For each fold,
    training data excludes observations within a purge window before the
    validation start and an embargo window after the validation end. This
    prevents information leakage from overlapping forward-return labels.

    Parameters:
        n_splits (int): Number of validation folds.
        purge_days (int): Days to remove before each validation period to
            avoid label leakage from overlapping forward returns.
        embargo_days (int): Days to remove after each validation period to
            prevent using data that could be influenced by the validation set.
    """

    def __init__(self, n_splits=3, purge_days=28, embargo_days=5):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, dates):
        """
        Generate train/validation index pairs.

        Parameters:
            X (pd.DataFrame): Feature matrix (used for index alignment).
            dates (pd.Series): Date column aligned with X.

        Yields:
            tuple[pd.Index, pd.Index]: Train and validation indices.
        """
        unique_dates = np.sort(dates.unique())
        fold_size = len(unique_dates) // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else len(unique_dates)

            val_dates = unique_dates[start:end]
            val_start = val_dates[0]
            val_end = val_dates[-1]

            purge_before = val_start - timedelta(days=self.purge_days)
            purge_after = val_end + timedelta(days=self.embargo_days)

            train_idx = X[
                (dates < purge_before) | (dates > purge_after)
            ].index

            val_idx = X[
                (dates >= val_start) & (dates <= val_end)
            ].index

            yield train_idx, val_idx
