"""
View 2: Sequential Feature Construction — 3D tensor builder for LSTM.
Matches notebook 06 — backward-only sliding window (T=10).
"""
import numpy as np
from collections import defaultdict


class SequentialTensorBuilder:
    """Build 3D tensor [N, T, F] for LSTM from 2D tabular data.

    Uses backward-only sliding window per card account to prevent data leakage.
    """
    def __init__(self, seq_len=10):
        self.seq_len = seq_len

    def build_card_sequences(self, X, card_col_idx):
        """Pre-compute all sequences using dict lookup O(1).

        Each transaction only looks BACKWARD (past history) → no leakage.

        Args:
            X: Feature matrix (numpy array, shape: [N, F]).
            card_col_idx: Column index for card1 (account identifier).

        Returns:
            Numpy array of shape [N, seq_len, F].
        """
        n_samples, n_features = X.shape
        sequences = np.zeros((n_samples, self.seq_len, n_features), dtype=np.float32)
        card_history = defaultdict(list)

        for idx in range(n_samples):
            card_id = X[idx, card_col_idx]
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
