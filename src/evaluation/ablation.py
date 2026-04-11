"""
Ablation study - leave-one-out model contribution analysis.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class AblationStudy:
    """Leave-one-out ablation to quantify each model's contribution."""

    def run_leave_one_out(self, oof_dict, y_true, model_names=None):
        if model_names is None:
            model_names = list(oof_dict.keys())

        X_full = np.column_stack([oof_dict[m] for m in model_names])
        meta_full = LogisticRegression(penalty="l2", C=0.01, max_iter=300, random_state=42)
        meta_full.fit(X_full, y_true)
        full_auc = roc_auc_score(y_true, meta_full.predict_proba(X_full)[:, 1])

        print("\n  Ablation Study (Leave-One-Out)")
        print(f"  Full Ensemble AUC: {full_auc:.4f}")
        print(f"  {'Model':<10} {'AUC w/o':<12} {'Drop':<10} {'Contribution'}")
        print(f"  {'-' * 50}")

        results = {}
        for model_name in model_names:
            remaining = [m for m in model_names if m != model_name]
            X_ablated = np.column_stack([oof_dict[m] for m in remaining])

            temp_meta = LogisticRegression(penalty="l2", C=0.01, max_iter=300, random_state=42)
            temp_meta.fit(X_ablated, y_true)
            auc_ablated = roc_auc_score(y_true, temp_meta.predict_proba(X_ablated)[:, 1])
            drop = full_auc - auc_ablated

            results[model_name] = {"auc_without": auc_ablated, "auc_drop": drop}
            bar = "#" * max(1, int(drop * 1000))
            print(f"  {model_name:<10} {auc_ablated:<12.4f} {drop:<10.4f} {bar}")

        mvp = max(results, key=lambda key: results[key]["auc_drop"])
        print(
            f"\n  MVP (Most Valuable Model): {mvp} "
            f"(removing it drops AUC by {results[mvp]['auc_drop']:.4f})"
        )
        return results
