"""
PSI drift monitoring.
"""
import numpy as np


def compute_psi(expected, actual, n_bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = (expected_counts + 1) / (expected_counts.sum() + len(expected_counts))
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + len(actual_counts))
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))


class PSIDriftMonitor:
    """Monitor feature and score distribution drift using PSI."""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def monitor_features(self, X_train, X_test, feature_names):
        psi_results = []
        for i, fname in enumerate(feature_names):
            psi_val = compute_psi(X_train[:, i], X_test[:, i], self.n_bins)
            psi_results.append((fname, psi_val))

        psi_results.sort(key=lambda item: -item[1])

        n_stable = sum(1 for _, p in psi_results if p < 0.10)
        n_moderate = sum(1 for _, p in psi_results if 0.10 <= p < 0.25)
        n_significant = sum(1 for _, p in psi_results if p >= 0.25)

        print(f"\n  PSI Summary ({len(feature_names)} features):")
        print(f"    Stable (PSI < 0.10):       {n_stable:3d}")
        print(f"    Moderate (0.10-0.25):      {n_moderate:3d}")
        print(f"    Significant (PSI >= 0.25): {n_significant:3d}")
        return psi_results

    def monitor_score_drift(self, oof_scores, test_scores):
        score_psi = compute_psi(oof_scores, test_scores, self.n_bins)
        print(f"\n  Score PSI: {score_psi:.4f}", end=" ")
        if score_psi >= 1.0:
            print("(Extreme imbalance - use Wasserstein instead)")
        elif score_psi >= 0.25:
            print("(Significant drift)")
        elif score_psi >= 0.10:
            print("(Moderate drift)")
        else:
            print("(Stable)")
        return score_psi
