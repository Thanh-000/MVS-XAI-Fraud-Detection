"""
Evaluation Metrics — AUC, PR-AUC, F1, Optimal Threshold.
Matches notebook 06 (v4.3.4).
"""
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve, f1_score
)


class ModelEvaluator:
    """Comprehensive model evaluation suite for binary fraud detection."""

    def __init__(self, y_true):
        self.y_true = y_true

    def print_comprehensive_report(self, y_pred_proba, threshold=0.5):
        """Print ROC-AUC, PR-AUC, F1, and classification report.

        Args:
            y_pred_proba: Predicted fraud probabilities.
            threshold: Decision threshold (default: 0.5).
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        roc_auc = roc_auc_score(self.y_true, y_pred_proba)
        pr_auc = average_precision_score(self.y_true, y_pred_proba)
        f1 = f1_score(self.y_true, y_pred)

        print("=" * 50)
        print("  COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 50)
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  PR-AUC:      {pr_auc:.4f}")
        print(f"  F1-Score:    {f1:.4f} (threshold={threshold:.2f})")
        print(f"\n  Classification Report (threshold={threshold:.2f}):")
        print(classification_report(self.y_true, y_pred,
              target_names=['Legitimate', 'Fraud']))

    def find_optimal_threshold(self, y_pred_proba):
        """Find threshold that maximizes F1-score using precision-recall curve.

        Returns:
            Optimal threshold (float).
        """
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_true, y_pred_proba
        )
        f1_scores = (
            2 * (precisions[:-1] * recalls[:-1]) /
            (precisions[:-1] + recalls[:-1] + 1e-9)
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"  Optimal F1 Threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
        return best_threshold

    def threshold_sweep(self, y_pred_proba, thresholds=None):
        """Sweep across thresholds and report F1/Precision/Recall.

        Args:
            y_pred_proba: Predicted fraud probabilities.
            thresholds: List of thresholds to evaluate.

        Returns:
            List of (threshold, f1, precision, recall) tuples.
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.05)

        results = []
        for t in thresholds:
            y_pred = (y_pred_proba >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            prec = precision_score(self.y_true, y_pred, zero_division=0)
            rec = recall_score(self.y_true, y_pred, zero_division=0)
            results.append((t, f1, prec, rec))
            print(f"    t={t:.2f}: F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

        return results