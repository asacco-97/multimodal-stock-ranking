from datetime import timedelta
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    Forward-only purged time series CV splitter.

    Guarantees that all training dates are strictly before test dates.
    A date gap is enforced between train end and test start to reduce leakage.

    Notes:
    - `purge_days` removes observations immediately before each test window.
    - `embargo_days` adds an extra buffer before test start in this past-only setup.
    - Effective train/test gap is `purge_days + embargo_days` days.
    """

    def __init__(
        self,
        n_splits: int = 3,
        purge_days: int = 21, # Approximately 1 trading month
        embargo_days: int = 5,
        test_size: int | None = None,
        min_train_days: int = 1000,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if purge_days < 0 or embargo_days < 0:
            raise ValueError("purge_days and embargo_days must be >= 0")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be >= 1 when provided")
        if min_train_days < 1:
            raise ValueError("min_train_days must be >= 1")

        self.n_splits = int(n_splits)
        self.purge_days = int(purge_days)
        self.embargo_days = int(embargo_days)
        self.test_size = test_size
        self.min_train_days = int(min_train_days)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, dates) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        dates = pd.to_datetime(dates)

        if len(dates) != n_samples:
            raise ValueError("dates and X must have the same length")
        if pd.isna(dates).any():
            raise ValueError("dates contains NaT values; clean them before splitting")

        # Use numpy arrays for low-overhead boolean indexing.
        date_arr = dates.to_numpy()
        unique_dates = np.sort(pd.unique(date_arr))
        n_dates = len(unique_dates)

        if n_dates < 2:
            raise ValueError("Need at least 2 unique dates to create train/test splits")

        if self.test_size is None:
            # Expanding-window layout:
            # initial train block + n_splits test blocks.
            test_size = n_dates // (self.n_splits + 1)
        else:
            test_size = int(self.test_size)

        if test_size < 1:
            raise ValueError("Computed test_size is 0; reduce n_splits or provide test_size")

        required_dates = self.n_splits * test_size + self.min_train_days
        if n_dates < required_dates:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for n_splits={self.n_splits}, "
                f"test_size={test_size}, min_train_days={self.min_train_days}"
            )

        initial_train_size = n_dates - self.n_splits * test_size
        gap_days = self.purge_days + self.embargo_days

        for i in range(self.n_splits):
            test_start_pos = initial_train_size + i * test_size
            # Keep the final fold as remainder-aware to fully utilize history.
            if i < self.n_splits - 1:
                test_end_pos = test_start_pos + test_size
            else:
                test_end_pos = n_dates

            test_dates = unique_dates[test_start_pos:test_end_pos]
            if len(test_dates) == 0:
                continue

            test_start = test_dates[0]
            test_end = test_dates[-1]

            # Strictly before the test window, with purge+embargo gap.
            train_cutoff = test_start - np.timedelta64(gap_days, "D")
            train_mask = date_arr < train_cutoff
            test_mask = (date_arr >= test_start) & (date_arr <= test_end)

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

