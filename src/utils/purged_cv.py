import numpy as np
import pandas as pd
from datetime import timedelta


class PurgedTimeSeriesSplit:

    def __init__(self, n_splits=3, purge_days=28, embargo_days=5):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, dates):

        # Ensure datetime
        dates = pd.to_datetime(dates)

        unique_dates = dates.sort_values().unique()

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

