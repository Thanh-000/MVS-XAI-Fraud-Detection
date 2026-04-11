"""
Wasserstein Distance Drift Detection — Earth Mover's Distance.
Detects feature distribution shift between train and test sets.
"""
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np


class DriftDetector:
    """Detect concept drift using Wasserstein (Earth Mover's) Distance."""

    def calculate_drift(self, df_train, df_test, feature_cols, warning_threshold=0.1):
        """Compute Wasserstein distance per feature between train and test.

        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
            feature_cols: List of feature column names.
            warning_threshold: W-distance above this triggers a warning.

        Returns:
            DataFrame sorted by drift severity (descending).
        """
        drifts = []
        print(f"\n  Wasserstein Drift Detection ({len(feature_cols)} features):")

        for col in feature_cols:
            train_dist = df_train[col].dropna().values
            test_dist = df_test[col].dropna().values

            if len(train_dist) == 0 or len(test_dist) == 0:
                continue

            wd = wasserstein_distance(train_dist, test_dist)
            status = "DRIFT" if wd > warning_threshold else "STABLE"
            drifts.append({'Feature': col, 'Wasserstein_Distance': wd, 'Status': status})

        drift_df = pd.DataFrame(drifts).sort_values('Wasserstein_Distance', ascending=False)

        n_drift = len(drift_df[drift_df['Status'] == 'DRIFT'])
        n_stable = len(drift_df[drift_df['Status'] == 'STABLE'])
        print(f"    Drifted features:  {n_drift}")
        print(f"    Stable features:   {n_stable}")

        if n_drift > 0:
            print(f"\n  Top 10 drifted features:")
            print(drift_df.head(10).to_markdown(index=False))

        return drift_df


def wasserstein_drift_report(X_train, X_test, feature_names, threshold=0.1):
    """Convenience function for numpy array inputs.

    Args:
        X_train: Training feature matrix (numpy).
        X_test: Test feature matrix (numpy).
        feature_names: List of feature names.
        threshold: Warning threshold.

    Returns:
        DataFrame with drift analysis.
    """
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_test = pd.DataFrame(X_test, columns=feature_names)
    detector = DriftDetector()
    return detector.calculate_drift(df_train, df_test, feature_names, threshold)
