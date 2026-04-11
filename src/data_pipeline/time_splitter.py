"""
Temporal Split with Gap — Walk-Forward Cross-Validation.
Matches notebook 06 (v4.3.4).

Key design: GAP_SIZE = 1000 samples between train and validation
to prevent temporal leakage from overlapping time periods.
"""
import numpy as np


class TemporalSplitter:
    """Walk-forward CV splitter with temporal gap between train/validation."""

    def __init__(self, n_splits=5, gap_size=1000):
        self.n_splits = n_splits
        self.gap_size = gap_size

    def split(self, X, sort_col=None):
        """Generate train/validation indices with temporal gap.

        Args:
            X: Feature matrix (or DataFrame).
            sort_col: Optional column index to sort by (e.g., TransactionDT).
                      If None, assumes data is already time-sorted.

        Yields:
            (train_idx, val_idx) tuples for each fold.
        """
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for fold_i in range(self.n_splits):
            train_end = fold_size * (fold_i + 1)
            val_start = train_end + self.gap_size
            val_end = val_start + fold_size

            if val_end > n:
                val_end = n
            if val_start >= n:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            print(f"    Fold {fold_i+1}: train[:{train_end}] -> gap({self.gap_size}) -> val[{val_start}:{val_end}]")
            yield train_idx, val_idx

    def split_holdout(self, X, test_ratio=0.15):
        """Simple temporal holdout: last test_ratio% as test set with gap.

        Returns:
            (train_idx, test_idx) tuple.
        """
        n = len(X)
        test_start_raw = int(n * (1 - test_ratio))
        test_start = test_start_raw + self.gap_size
        if test_start >= n:
            test_start = test_start_raw

        train_idx = np.arange(0, test_start_raw)
        test_idx = np.arange(test_start, n)

        print(f"  Holdout: train[:{test_start_raw}] -> gap({self.gap_size}) -> test[{test_start}:{n}]")
        return train_idx, test_idx
