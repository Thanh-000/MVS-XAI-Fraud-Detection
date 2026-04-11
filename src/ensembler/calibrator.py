"""
Probability Calibration Utility — Platt Scaling / Isotonic Regression.
Complementary to the meta-learner's built-in calibration.
"""
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


class ProbabilityCalibrator:
    """Post-hoc probability calibration for any base estimator.

    Uses Platt scaling (sigmoid) or isotonic regression to ensure
    predicted probabilities reflect true event frequencies.
    """

    def __init__(self, base_estimator, method='sigmoid', cv='prefit'):
        """Initialize calibrator.

        Args:
            base_estimator: Pre-fitted model to calibrate.
            method: 'sigmoid' (Platt scaling) or 'isotonic' (non-parametric).
            cv: 'prefit' if base_estimator is already fitted.
        """
        self.calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=method,
            cv=cv
        )
        self.is_fitted = False

    def fit(self, X_val, y_val):
        """Fit calibration mapping on validation set.

        Args:
            X_val: Validation features (or 1D predicted probabilities).
            y_val: Validation labels.
        """
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(-1, 1)

        self.calibrator.fit(X_val, y_val)
        self.is_fitted = True
        print("  Calibration fitted — probabilities now reflect true risk")

    def predict_proba(self, X_test):
        """Return calibrated probabilities.

        Args:
            X_test: Test features (or 1D predicted probabilities).

        Returns:
            Calibrated probability of fraud (class 1).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)

        return self.calibrator.predict_proba(X_test)[:, 1]
