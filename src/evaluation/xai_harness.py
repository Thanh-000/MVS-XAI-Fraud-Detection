"""Lightweight XAI evaluation harness for proposal-aligned reporting."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class XAIEvaluationHarness:
    """Evaluate explanation artifacts with proxy metrics that run in-repo."""

    def __init__(self, auditor):
        self.auditor = auditor

    def _resolve_feature_name(self, raw_name):
        if raw_name in self.auditor._feature_index:
            return raw_name
        for feature_name in self.auditor.feature_names:
            if feature_name in raw_name:
                return feature_name
        return None

    def _top_feature_indices(self, feature_pairs, top_k=5):
        indices = []
        for raw_name, _ in feature_pairs:
            feature_name = self._resolve_feature_name(raw_name)
            if feature_name is None:
                continue
            indices.append(self.auditor._feature_index[feature_name])
            if len(indices) >= top_k:
                break
        return indices

    def feature_removal_degradation(self, instance, shap_pairs, top_k=3):
        instance = np.asarray(instance, dtype=float).copy()
        baseline_score = float(self.auditor._predict_positive_proba(instance.reshape(1, -1))[0])
        top_indices = self._top_feature_indices(shap_pairs, top_k=top_k)
        degraded = instance.copy()
        for feature_idx in top_indices:
            degraded[feature_idx] = self.auditor._background_reference[feature_idx]
        degraded_score = float(self.auditor._predict_positive_proba(degraded.reshape(1, -1))[0])
        return baseline_score - degraded_score

    def lime_local_fidelity_proxy(self, instance, lime_pairs, n_samples=128):
        top_indices = self._top_feature_indices(lime_pairs, top_k=min(5, len(lime_pairs)))
        if not top_indices:
            return float("nan")

        instance = np.asarray(instance, dtype=float)
        neighborhood = np.repeat(instance.reshape(1, -1), n_samples, axis=0)
        rng = np.random.default_rng(42)

        for feature_idx in top_indices:
            scale = self.auditor._background_scale[feature_idx]
            noise = rng.normal(loc=0.0, scale=scale, size=n_samples)
            neighborhood[:, feature_idx] = neighborhood[:, feature_idx] + noise

        X_local = neighborhood[:, top_indices]
        y_local = self.auditor._predict_positive_proba(neighborhood)
        surrogate = LinearRegression()
        surrogate.fit(X_local, y_local)
        return float(surrogate.score(X_local, y_local))

    @staticmethod
    def counterfactual_validity(results):
        if isinstance(results, dict):
            counterfactuals = results.get("counterfactuals", [])
            if not counterfactuals:
                return 0.0
            return float(np.mean([cf.get("flipped", False) for cf in counterfactuals]))
        return float("nan")

    @staticmethod
    def anchor_precision(results):
        return float(getattr(results, "precision", np.nan))

    @staticmethod
    def narrative_completeness(summary):
        required_sections = [
            "Fraud Risk Assessment",
            "Key Factors",
            "Local Explanation",
        ]
        coverage = sum(section in summary for section in required_sections) / len(required_sections)
        return float(coverage)

    def evaluate(self, instance, results):
        shap_pairs = self.auditor._extract_shap_pairs(results["shap"], limit=5)
        lime_pairs = results["lime"].as_list() if results.get("lime") is not None else []
        metrics = {
            "frd_proxy": self.feature_removal_degradation(instance, shap_pairs, top_k=3),
            "lime_local_fidelity_proxy_r2": self.lime_local_fidelity_proxy(instance, lime_pairs),
            "dice_validity_rate": self.counterfactual_validity(results.get("dice")),
            "anchor_precision": self.anchor_precision(results.get("anchors")),
            "nl_completeness": self.narrative_completeness(results.get("llm_summary", "")),
        }

        print("\n  XAI Harness:")
        for metric_name, metric_value in metrics.items():
            if np.isnan(metric_value):
                print(f"    {metric_name}: n/a")
            else:
                print(f"    {metric_name}: {metric_value:.4f}")
        return metrics
