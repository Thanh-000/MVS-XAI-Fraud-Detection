"""
Fairness audit.
"""
import pandas as pd


class FairnessAuditor:
    """Audit model fairness across protected groups."""

    def audit_equalized_odds(self, df, protect_col="card4", label_col="isFraud", pred_col="Action"):
        print(f"\n  Fairness Audit - Equalized Odds on '{protect_col}':")

        groups = df[protect_col].dropna().unique()
        report = []

        for group in groups:
            subset = df[df[protect_col] == group]
            if len(subset) == 0:
                continue

            true_frauds = subset[subset[label_col] == 1]
            true_legits = subset[subset[label_col] == 0]

            tpr = 0.0
            if len(true_frauds) > 0:
                caught = len(true_frauds[true_frauds[pred_col] == "AUTO_BLOCK"])
                tpr = caught / len(true_frauds)

            fpr = 0.0
            if len(true_legits) > 0:
                false_alarms = len(true_legits[true_legits[pred_col] == "AUTO_BLOCK"])
                fpr = false_alarms / len(true_legits)

            report.append(
                {
                    "Group": group,
                    "Size": len(subset),
                    "TPR (Catch Rate)": f"{tpr:.2%}",
                    "FPR (False Block)": f"{fpr:.2%}",
                }
            )

        report_df = pd.DataFrame(report)
        print(report_df.to_markdown(index=False))

        if len(report) >= 2:
            tpr_values = [float(item["TPR (Catch Rate)"].strip("%")) / 100 for item in report]
            tpr_disparity = max(tpr_values) - min(tpr_values)
            print(f"\n  TPR Disparity: {tpr_disparity:.2%} ", end="")
            if tpr_disparity > 0.10:
                print("(WARNING: >10% gap - investigate bias)")
            else:
                print("(OK: <10% gap)")

        return report_df
