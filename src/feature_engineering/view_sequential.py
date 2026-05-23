"""
View 2: Sequential Feature Construction — 3D tensor builder for LSTM.
Matches notebook 06 — backward-only sliding window (T=10).
"""
import numpy as np
import pandas as pd


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


class IndexedSequenceArray:
    """Lazy sequence tensor backed by an index matrix.

    This avoids materializing an (N, T, F) tensor for large datasets. The
    sequence index matrix stores the previous row indices for each sample; -1
    means left-padding with zeros.
    """

    is_indexed_sequence = True

    def __init__(self, X, seq_indices):
        self.X = np.asarray(X, dtype=np.float32)
        self.seq_indices = np.asarray(seq_indices, dtype=np.int32)
        self.shape = (
            int(self.seq_indices.shape[0]),
            int(self.seq_indices.shape[1]),
            int(self.X.shape[1]),
        )
        self.dtype = np.float32

    def __len__(self):
        return self.shape[0]

    def get_batch(self, item):
        idx = np.arange(self.shape[0])[item] if isinstance(item, slice) else np.asarray(item)
        scalar = idx.ndim == 0
        idx = idx.reshape(1) if scalar else idx

        seq_idx = self.seq_indices[idx]
        out = np.zeros((len(idx), self.shape[1], self.shape[2]), dtype=np.float32)
        valid = seq_idx >= 0
        if valid.any():
            out[valid] = self.X[seq_idx[valid]]
        return out[0] if scalar else out

    def __getitem__(self, item):
        return self.get_batch(item)


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
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        entity_ids = np.asarray(entity_ids)

        if pd.Index(entity_ids).is_unique:
            print(f"  View 2 (Sequential): Unique entities; using lazy zero tensor "
                  f"({n_samples}, {self.seq_len}, {n_features})")
            return ZeroSequenceArray(n_samples, self.seq_len, n_features)

        codes = pd.factorize(entity_ids, sort=False)[0].astype(np.int64, copy=False)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]

        seq_sorted = np.full((n_samples, self.seq_len), -1, dtype=np.int32)
        positions = np.arange(n_samples)
        for lag in range(1, self.seq_len + 1):
            target_pos = positions[lag:]
            same_entity = sorted_codes[lag:] == sorted_codes[:-lag]
            if same_entity.any():
                seq_sorted[target_pos[same_entity], self.seq_len - lag] = order[target_pos[same_entity] - lag]

        seq_indices = np.empty_like(seq_sorted)
        seq_indices[order] = seq_sorted

        if not (seq_indices >= 0).any():
            print(f"  View 2 (Sequential): No repeated history; using lazy zero tensor "
                  f"({n_samples}, {self.seq_len}, {n_features})")
            return ZeroSequenceArray(n_samples, self.seq_len, n_features)

        dense_size_mb = n_samples * self.seq_len * n_features * np.dtype(np.float32).itemsize / (1024 ** 2)
        index_size_mb = seq_indices.nbytes / (1024 ** 2)
        if dense_size_mb <= 1024:
            sequences = np.zeros((n_samples, self.seq_len, n_features), dtype=np.float32)
            valid = seq_indices >= 0
            sequences[valid] = X[seq_indices[valid]]
            print(f"  View 2 (Sequential): Built dense tensor {sequences.shape} "
                  f"(~{dense_size_mb:.1f} MB)")
            return sequences

        print(f"  View 2 (Sequential): Built lazy indexed tensor "
              f"({n_samples}, {self.seq_len}, {n_features}) "
              f"(index ~{index_size_mb:.1f} MB, avoided dense ~{dense_size_mb:.1f} MB)")
        return IndexedSequenceArray(X, seq_indices)
