"""
View 2: Sequential Feature Construction — 3D tensor builder for LSTM.
Matches notebook 06 — backward-only sliding window (T=10).
"""
import numpy as np
from collections import defaultdict


class ZeroSequenceArray:
    """Array-like all-zero sequence tensor without materializing the full tensor."""

    is_zero_sequence = True

    def __init__(self, n_samples, seq_len, n_features):
        self.shape = (int(n_samples), int(seq_len), int(n_features))
        self.dtype = np.float32

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < 0:
                item += self.shape[0]
            if item < 0 or item >= self.shape[0]:
                raise IndexError(item)
            return np.zeros(self.shape[1:], dtype=np.float32)

        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            n_rows = max(0, (stop - start + (step - 1)) // step)
        else:
            n_rows = len(item)
        return np.zeros((n_rows, self.shape[1], self.shape[2]), dtype=np.float32)


class SequentialTensorBuilder:
    """Build 3D tensor [N, T, F] for LSTM from 2D tabular data.

    Uses backward-only sliding window per card account to prevent data leakage.
    """
    def __init__(self, seq_len=10):
        self.seq_len = seq_len

    def build_card_sequences(self, X, entity_ids):
        """Pre-compute all sequences using dict lookup O(1).

        Each transaction only looks BACKWARD (past history) → no leakage.

        Args:
            X: Feature matrix (numpy array, shape: [N, F]).
            entity_ids: 1D array of account/entity identifiers aligned with X.

        Returns:
            Numpy array of shape [N, seq_len, F].
        """
        n_samples, n_features = X.shape
        if np.unique(entity_ids).size == n_samples:
            print(f"  View 2 (Sequential): Unique entities; using lazy zero tensor "
                  f"({n_samples}, {self.seq_len}, {n_features})")
            return ZeroSequenceArray(n_samples, self.seq_len, n_features)

        sequences = np.zeros((n_samples, self.seq_len, n_features), dtype=np.float32)
        card_history = defaultdict(list)
        entity_ids = np.asarray(entity_ids)

        for idx in range(n_samples):
            card_id = entity_ids[idx]
            history = card_history[card_id]

            if len(history) >= self.seq_len:
                sequences[idx] = X[history[-self.seq_len:]]
            elif len(history) > 0:
                pad_len = self.seq_len - len(history)
                sequences[idx, pad_len:] = X[history]
            # else: all zeros (padding for first transaction)

            history.append(idx)
            if len(history) > 30:  # Keep last 30 for T=10
                card_history[card_id] = history[-30:]

        print(f"  View 2 (Sequential): Built tensor {sequences.shape} "
              f"(T={self.seq_len}, {n_features} features per step)")
        return sequences
