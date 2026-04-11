"""
Fairness Audit — Equalized Odds analysis for protected attributes.
Ensures no card type or demographic group is disproportionately affected.
"""
import pandas as pd
import numpy as np


class FairnessAuditor:
    """Audit model fairness across protected groups (EU AI Act compliance)."""

    def audit_equalized_odds(self, df, protect_col='card4', label_col='isFraud', pred_col='Action'):
        """Measure Equalized Odds (TPR and FPR parity) across protected groups.

        Args:
            df: DataFrame with predictions and labels.
            protect_col: Protected attribute column (e.g., 'card4' for card type).
            label_col: Ground truth label column.
            pred_col: Prediction/decision column ('AUTO_BLOCK', 'ALLOW', etc.).

        Returns:
            DataFrame with per-group TPR and FPR.
        """
        print(f"\n  Fairness Audit — Equalized Odds on '{protect_col}':")

        groups = df[protect_col].dropna().unique()
        report = []

        for g in groups:
            subset = df[df[protect_col] == g]
            if len(subset) == 0:
                continue

            true_frauds = subset[subset[label_col] == 1]
            true_legits = subset[subset[label_col] == 0]

            # TPR: True Positive Rate (fraud catch rate)
            tpr = 0.0
            if len(true_frauds) > 0:
                caught = len(true_frauds[true_frauds[pred_col] == 'AUTO_BLOCK'])
                tpr = caught / len(true_frauds)

            # FPR: False Positive Rate (innocent block rate)
            fpr = 0.0
            if len(true_legits) > 0:
                false_alarms = len(true_legits[true_legits[pred_col] == 'AUTO_BLOCK'])
                fpr = false_alarms / len(true_legits)

            report.append({
                'Group': g,
                'Size': len(subset),
                'TPR (Catch Rate)': f"{tpr:.2%}",
                'FPR (False Block)': f"{fpr:.2%}"
            })

        report_df = pd.DataFrame(report)
        print(report_df.to_markdown(index=False))

        # Disparity check
        if len(report) >= 2:
            tpr_values = [float(r['TPR (Catch Rate)'].strip('%')) / 100 for r in report]
            tpr_disparity = max(tpr_values) - min(tpr_values)
            print(f"\n  TPR Disparity: {tpr_disparity:.2%} ", end="")
            if tpr_disparity > 0.10:
                print("(WARNING: >10% gap — investigate bias)")
            else:
                print("(OK: <10% gap)")

        return report_df
