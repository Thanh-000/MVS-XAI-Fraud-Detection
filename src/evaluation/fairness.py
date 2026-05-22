"""Fairness audit utilities for protected-group disparity checks."""
import pandas as pd


class FairnessAuditor:
    """Audit model fairness across protected groups."""

    @staticmethod
    def _print_table(df):
        try:
            print(df.to_markdown(index=False))
        except ImportError:
            print(df.to_string(index=False))

    @staticmethod
    def _block_rate(subset, pred_col):
        if len(subset) == 0:
            return 0.0
        return float((subset[pred_col] == "AUTO_BLOCK").mean())

    def audit_demographic_parity(self, df, protect_col="card4", pred_col="Action"):
        """Compute demographic parity via the AUTO_BLOCK rate gap."""
        print(f"\n  Fairness Audit - Demographic Parity on '{protect_col}':")

        groups = df[protect_col].dropna().unique()
        report = []

        for group in groups:
            subset = df[df[protect_col] == group]
            if len(subset) == 0:
                continue

            block_rate = self._block_rate(subset, pred_col)
            report.append(
                {
                    "Group": group,
                    "Size": len(subset),
                    "Block Rate": f"{block_rate:.2%}",
                }
            )

        report_df = pd.DataFrame(report)
        if not report_df.empty:
            self._print_table(report_df)

        disparity = 0.0
        if len(report) >= 2:
            block_rates = [float(item["Block Rate"].strip("%")) / 100 for item in report]
            disparity = max(block_rates) - min(block_rates)
            print(f"\n  Demographic Parity Difference: {disparity:.2%}", end=" ")
            print("(OK: <5% gap)" if disparity < 0.05 else "(WARNING: >=5% gap)")

        return {"report": report_df, "dp_difference": disparity}

    def audit_equalized_odds(self, df, protect_col="card4", label_col="isFraud", pred_col="Action"):
        """Compute Equalized Odds through TPR/FPR group gaps."""
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
        if not report_df.empty:
            self._print_table(report_df)

        tpr_disparity = 0.0
        fpr_disparity = 0.0
        if len(report) >= 2:
            tpr_values = [float(item["TPR (Catch Rate)"].strip("%")) / 100 for item in report]
            fpr_values = [float(item["FPR (False Block)"].strip("%")) / 100 for item in report]
            tpr_disparity = max(tpr_values) - min(tpr_values)
            fpr_disparity = max(fpr_values) - min(fpr_values)
            print(f"\n  Equalized Odds TPR Difference: {tpr_disparity:.2%}", end=" ")
            print("(OK: <5% gap)" if tpr_disparity < 0.05 else "(WARNING: >=5% gap)")
            print(f"  Equalized Odds FPR Difference: {fpr_disparity:.2%}", end=" ")
            print("(OK: <5% gap)" if fpr_disparity < 0.05 else "(WARNING: >=5% gap)")

        return {
            "report": report_df,
            "tpr_difference": tpr_disparity,
            "fpr_difference": fpr_disparity,
        }

    def audit_full_report(self, df, protect_col="card4", label_col="isFraud", pred_col="Action"):
        """Run both demographic parity and equalized odds."""
        dp = self.audit_demographic_parity(df, protect_col=protect_col, pred_col=pred_col)
        eo = self.audit_equalized_odds(
            df,
            protect_col=protect_col,
            label_col=label_col,
            pred_col=pred_col,
        )
        return {"demographic_parity": dp, "equalized_odds": eo}
