"""
Feature scaling pipeline.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np


class FeatureScalerPipeline:
    """StandardScaler wrapper for neural-network preprocessing."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X_train):
        X_scaled = self.scaler.fit_transform(X_train).astype(np.float32, copy=False)
        self._fitted = True
        print(f"  StandardScaler: fit on {X_train.shape} -> mean~=0, std~=1")
        return X_scaled

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("FeatureScalerPipeline: call fit_transform() first")
        return self.scaler.transform(X).astype(np.float32, copy=False)

    def inverse_transform(self, X_scaled):
        return self.scaler.inverse_transform(X_scaled)
