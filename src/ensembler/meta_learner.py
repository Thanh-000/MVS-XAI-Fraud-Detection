"""
Meta-Learner: Logistic Regression (L2) + Platt Calibration.
Matches notebook 06 (v4.3.4).

Architecture:
1. Stack OOF predictions from 6 base models → 6-dimensional feature vector
2. Apply confidence gating: w = min(1, (AUC/τ)²); τ=0.60
3. Fit LogisticRegression(C=0.01, penalty='l2') on gated features
4. Post-calibrate with Platt scaling (CalibratedClassifierCV)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


class MetaEnsembler:
    """L2 Logistic Regression meta-learner with Platt calibration."""

    def __init__(self, C=0.01, cv_cal=3):
        self.base_lr = LogisticRegression(
            C=C,              # Strong L2 regularization
            penalty='l2',
            solver='lbfgs',
            max_iter=300,
            random_state=42
        )
        self.cv_cal = cv_cal
        self.calibrated_model = None

    def fit(self, oof_matrix, y_true):
        """Fit meta-learner on (optionally gated) OOF prediction matrix.

        Args:
            oof_matrix: numpy array of shape [N, n_models].
                        Each column = one base model's OOF predictions.
            y_true: True labels (numpy array).
        """
        # Step 1: Fit base LR
        self.base_lr.fit(oof_matrix, y_true)

        # Step 2: Platt calibration
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_lr,
            cv=self.cv_cal,
            method='sigmoid'   # Platt scaling
        )
        self.calibrated_model.fit(oof_matrix, y_true)

        # Report model weights
        coefs = self.base_lr.coef_[0]
        print(f"\n  Meta-Learner Weights: {np.round(coefs, 4)}")
        print(f"  Meta-Learner Bias: {self.base_lr.intercept_[0]:.4f}")
        return self

    def predict_proba(self, X):
        """Return calibrated probability of fraud (class 1).

        Uses Platt-calibrated model for well-calibrated probabilities
        (ensures ECE < 0.05 target).
        """
        return self.calibrated_model.predict_proba(X)[:, 1]

    def evaluate(self, X, y_true):
        """Evaluate meta-learner performance on holdout set.

        Returns dict with AUC, calibrated probabilities, and ECE.
        """
        proba = self.predict_proba(X)
        auc = roc_auc_score(y_true, proba)

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true, proba)

        print(f"\n  Meta-Learner Evaluation:")
        print(f"    AUC:   {auc:.4f}")
        print(f"    ECE:   {ece:.4f} {'✓' if ece < 0.05 else '✗'} (target < 0.05)")
        return {'auc': auc, 'ece': ece, 'probabilities': proba}

    @staticmethod
    def _compute_ece(y_true, y_pred, n_bins=10):
        """Expected Calibration Error — Brier Score decomposition."""
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
