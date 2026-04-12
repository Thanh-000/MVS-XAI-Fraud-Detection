"""
Meta-learner: Logistic Regression (L2) + Platt calibration.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class MetaEnsembler:
    """L2 Logistic Regression meta-learner with Platt calibration."""

    def __init__(self, C=0.01, cv_cal=3, seed=42):
        self.base_lr = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=300,
            random_state=seed,
        )
        self.cv_cal = cv_cal
        self.calibrated_model = None
        self.feature_names = None

    def fit(self, oof_matrix, y_true, feature_names=None):
        X_frame = self._ensure_frame(oof_matrix, feature_names=feature_names)
        self.base_lr.fit(X_frame, y_true)

        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_lr,
            cv=self.cv_cal,
            method="sigmoid",
        )
        self.calibrated_model.fit(X_frame, y_true)

        coefs = self.base_lr.coef_[0]
        print(f"\n  Meta-Learner Weights: {np.round(coefs, 4)}")
        print(f"  Meta-Learner Bias: {self.base_lr.intercept_[0]:.4f}")
        return self

    def predict_proba(self, X):
        X_frame = self._ensure_frame(X)
        return self.calibrated_model.predict_proba(X_frame)[:, 1]

    def evaluate(self, X, y_true):
        proba = self.predict_proba(X)
        auc = roc_auc_score(y_true, proba)
        ece = self._compute_ece(y_true, proba)
        status = "OK" if ece < 0.05 else "WARN"

        print("\n  Meta-Learner Evaluation:")
        print(f"    AUC:   {auc:.4f}")
        print(f"    ECE:   {ece:.4f} [{status}] (target < 0.05)")
        return {"auc": auc, "ece": ece, "probabilities": proba}

    @staticmethod
    def _compute_ece(y_true, y_pred, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
        return ece

    def _ensure_frame(self, X, feature_names=None):
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            if self.feature_names is None:
                self.feature_names = cols
            return X.loc[:, self.feature_names]

        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif self.feature_names is None:
            self.feature_names = [f"model_{i}" for i in range(np.asarray(X).shape[1])]

        return pd.DataFrame(X, columns=self.feature_names)
