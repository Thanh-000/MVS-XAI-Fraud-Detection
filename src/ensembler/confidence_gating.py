"""
Confidence-Based OOF Gating — Matches notebook 06 (v4.3.4).

Gating weight: w_m = min(1.0, (AUC_m / τ)²)
Where τ = 0.60 is the confidence threshold.

Models with AUC < τ get downweighted, preventing weak learners from
contaminating the meta-learner input.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


class ConfidenceGating:
    """Apply confidence-based gating to OOF predictions.

    Models performing below threshold τ are multiplicatively downweighted,
    ensuring the meta-learner receives quality-filtered inputs.
    """
    def __init__(self, tau=0.60):
        self.tau = tau

    def compute_gate_weight(self, y_true, y_pred_proba):
        """Compute gating weight based on fold AUC.

        w = min(1.0, (AUC / τ)²)
        """
        auc = roc_auc_score(y_true, y_pred_proba)
        w = min(1.0, (auc / self.tau) ** 2)
        return w, auc

    def apply_gating(self, y_true, predictions_dict):
        """Apply gating to a dictionary of model predictions.

        Args:
            y_true: True labels.
            predictions_dict: Dict of {model_name: predictions}.

        Returns:
            Dict of {model_name: gated_predictions} and weights dict.
        """
        gated = {}
        weights = {}
        for name, preds in predictions_dict.items():
            w, auc = self.compute_gate_weight(y_true, preds)
            gated[name] = preds * w
            weights[name] = {'auc': auc, 'gate_weight': w}
            print(f"      {name:5s} AUC: {auc:.4f} | Gate W: {w:.3f}")
        return gated, weights
