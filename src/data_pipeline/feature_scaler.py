"""
Feature Scaling Pipeline — StandardScaler for neural network inputs.
Matches notebook 06 (v4.3.4).

StandardScaler is applied before MLP and LSTM training to normalize
features to zero mean and unit variance, critical for gradient-based
optimization convergence.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np


class FeatureScalerPipeline:
    """StandardScaler wrapper for multi-view neural network preprocessing."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X_train):
        """Fit scaler on training data and transform.

        Args:
            X_train: Training feature matrix (numpy array).

        Returns:
            Scaled training features.
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self._fitted = True
        print(f"  StandardScaler: fit on {X_train.shape} → mean≈0, std≈1")
        return X_scaled

    def transform(self, X):
        """Transform data using fitted scaler parameters.

        Args:
            X: Feature matrix to scale (numpy array).

        Returns:
            Scaled features.

        Raises:
            RuntimeError: If scaler has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("FeatureScalerPipeline: call fit_transform() first")
        return self.scaler.transform(X)

    def inverse_transform(self, X_scaled):
        """Inverse transform back to original scale (for interpretability)."""
        return self.scaler.inverse_transform(X_scaled)
