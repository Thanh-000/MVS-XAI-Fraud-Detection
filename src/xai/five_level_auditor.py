"""
XAI module - 5-level explainability framework.

The implementation supports the full optional stack:
1. SHAP
2. LIME
3. DiCE
4. Anchors
5. Natural-language summary

When optional libraries are unavailable, deterministic fallbacks are used so
the audit pipeline can still execute in notebook demos and smoke tests.
"""

from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

try:
    import dice_ml
except ImportError:
    dice_ml = None

try:
    from anchor import anchor_tabular
except ImportError:
    anchor_tabular = None


class FallbackLimeExplanation:
    """Lightweight LIME-compatible container with the as_list API."""

    def __init__(self, feature_weights, score):
        self._feature_weights = list(feature_weights)
        self.score = float(score)

    def as_list(self):
        return list(self._feature_weights)


class FallbackAnchorExplanation:
    """Lightweight Anchors-compatible container with the names API."""

    def __init__(self, rules, precision, coverage, method="fallback"):
        self._rules = list(rules)
        self.precision = float(precision)
        self.coverage = float(coverage)
        self.method = method

    def names(self):
        return list(self._rules)


class UltimateXAIAuditor:
    """5-level explainability framework for fraud detection models."""

    def __init__(self, model, X_background, feature_names):
        self.model = model
        self.X_background = np.asarray(X_background, dtype=float)
        self.feature_names = list(feature_names)
        self._feature_index = {
            feature_name: idx for idx, feature_name in enumerate(self.feature_names)
        }
        self._background_reference = np.nanmedian(self.X_background, axis=0)
        self._background_scale = np.nanstd(self.X_background, axis=0) + 1e-6

    def _predict_positive_proba(self, X):
        X = np.asarray(X, dtype=float)
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(X))
            if probabilities.ndim == 2:
                return probabilities[:, -1]
            return probabilities.astype(float)
        predictions = np.asarray(self.model.predict(X), dtype=float)
        return predictions.reshape(-1)

    def _feature_weights(self):
        if hasattr(self.model, "feature_importances_"):
            weights = np.asarray(self.model.feature_importances_, dtype=float)
        elif hasattr(self.model, "coef_"):
            weights = np.asarray(self.model.coef_, dtype=float)
            if weights.ndim == 2:
                weights = weights[-1]
            weights = np.abs(weights)
        else:
            weights = np.ones(len(self.feature_names), dtype=float)

        if weights.shape[0] != len(self.feature_names):
            weights = np.ones(len(self.feature_names), dtype=float)
        return weights

    def _local_contributions(self, X):
        X = np.asarray(X, dtype=float)
        centered = (X - self._background_reference) / self._background_scale
        return centered * self._feature_weights()

    def _top_feature_pairs(self, instance, num_features=5):
        contributions = self._local_contributions(np.asarray(instance).reshape(1, -1))[0]
        top_idx = np.argsort(np.abs(contributions))[::-1][:num_features]
        return [
            (self.feature_names[idx], float(contributions[idx]))
            for idx in top_idx
        ]

    def _extract_shap_pairs(self, shap_values, limit=5):
        if shap_values is None:
            return []

        if isinstance(shap_values, list):
            shap_array = np.asarray(shap_values[-1], dtype=float)
        else:
            shap_array = np.asarray(shap_values, dtype=float)

        if shap_array.ndim == 1:
            row = shap_array
        else:
            row = shap_array[0]

        top_idx = np.argsort(np.abs(row))[::-1][:limit]
        return [(self.feature_names[idx], float(row[idx])) for idx in top_idx]

    def shap_explain(self, X_explain, max_display=15):
        """Level 1: SHAP explanation, with deterministic fallback."""
        X_explain = np.asarray(X_explain, dtype=float)

        if shap is not None:
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_explain)
                print(
                    f"  [SHAP] Computed with library for {X_explain.shape[0]} samples "
                    f"and {min(max_display, X_explain.shape[1])} displayed features"
                )
                return shap_values
            except Exception as exc:
                print(f"  [SHAP] Library path failed ({exc}); using deterministic fallback")

        contributions = self._local_contributions(X_explain)
        print(
            f"  [SHAP] Fallback contributions computed for {X_explain.shape[0]} samples"
        )
        return contributions

    def lime_explain(self, instance, num_features=10):
        """Level 2: LIME explanation, with deterministic fallback."""
        instance = np.asarray(instance, dtype=float)

        if LimeTabularExplainer is not None:
            try:
                explainer = LimeTabularExplainer(
                    self.X_background,
                    feature_names=self.feature_names,
                    class_names=["Legit", "Fraud"],
                    mode="classification",
                )
                explanation = explainer.explain_instance(
                    instance,
                    self.model.predict_proba,
                    num_features=num_features,
                )
                print(f"  [LIME] Top {num_features} contributing features for instance")
                return explanation
            except Exception as exc:
                print(f"  [LIME] Library path failed ({exc}); using deterministic fallback")

        feature_weights = self._top_feature_pairs(instance, num_features=num_features)
        score = self._predict_positive_proba(instance.reshape(1, -1))[0]
        print(f"  [LIME] Fallback explanation created with {len(feature_weights)} features")
        return FallbackLimeExplanation(feature_weights, score=score)

    def dice_explain(self, instance, total_cfs=3):
        """Level 3: Counterfactual explanation, with heuristic fallback."""
        instance = np.asarray(instance, dtype=float)

        if dice_ml is not None:
            try:
                import pandas as pd

                data = dice_ml.Data(
                    dataframe=pd.DataFrame(
                        self.X_background, columns=self.feature_names
                    ).assign(isFraud=0),
                    continuous_features=self.feature_names,
                    outcome_name="isFraud",
                )
                model = dice_ml.Model(model=self.model, backend="sklearn")
                generator = dice_ml.Dice(data, model, method="random")
                counterfactuals = generator.generate_counterfactuals(
                    pd.DataFrame([instance], columns=self.feature_names),
                    total_CFs=total_cfs,
                )
                print(f"  [DiCE] Generated {total_cfs} counterfactual explanations")
                return counterfactuals
            except Exception as exc:
                print(f"  [DiCE] Library path failed ({exc}); using heuristic fallback")

        counterfactuals = []
        working_instance = instance.copy()
        original_score = float(self._predict_positive_proba(instance.reshape(1, -1))[0])

        for feature_name, _ in self._top_feature_pairs(instance, num_features=len(self.feature_names)):
            feature_idx = self._feature_index[feature_name]
            suggested_value = float(self._background_reference[feature_idx])
            candidate = working_instance.copy()
            candidate[feature_idx] = suggested_value
            candidate_score = float(self._predict_positive_proba(candidate.reshape(1, -1))[0])

            counterfactuals.append(
                {
                    "feature": feature_name,
                    "original_value": float(working_instance[feature_idx]),
                    "suggested_value": suggested_value,
                    "new_score": candidate_score,
                    "flipped": bool(candidate_score < 0.5),
                }
            )

            working_instance = candidate
            if len(counterfactuals) >= total_cfs or candidate_score < 0.5:
                break

        print(f"  [DiCE] Fallback produced {len(counterfactuals)} candidate interventions")
        return {
            "method": "heuristic_fallback",
            "original_score": original_score,
            "counterfactuals": counterfactuals,
        }

    def anchors_explain(self, instance):
        """Level 4: Rule-based explanation, with heuristic fallback."""
        instance = np.asarray(instance, dtype=float)

        if anchor_tabular is not None:
            try:
                explainer = anchor_tabular.AnchorTabularExplainer(
                    class_names=["Legit", "Fraud"],
                    feature_names=self.feature_names,
                    train_data=self.X_background,
                )
                explanation = explainer.explain_instance(
                    instance, self.model.predict, threshold=0.95
                )
                print(f"  [Anchors] Rule: {explanation.names()}")
                return explanation
            except Exception as exc:
                print(f"  [Anchors] Library path failed ({exc}); using heuristic fallback")

        rules = []
        for feature_name, _ in self._top_feature_pairs(instance, num_features=3):
            feature_idx = self._feature_index[feature_name]
            q1, q3 = np.nanpercentile(self.X_background[:, feature_idx], [25, 75])
            value = float(instance[feature_idx])
            if value >= q3:
                rules.append(f"{feature_name} >= {q3:.3f}")
            elif value <= q1:
                rules.append(f"{feature_name} <= {q1:.3f}")
            else:
                rules.append(f"{q1:.3f} < {feature_name} < {q3:.3f}")

        fraud_score = float(self._predict_positive_proba(instance.reshape(1, -1))[0])
        precision = max(fraud_score, 1.0 - fraud_score)
        coverage = 1.0 / max(len(rules), 1)
        print(f"  [Anchors] Fallback rules: {rules}")
        return FallbackAnchorExplanation(
            rules=rules,
            precision=precision,
            coverage=coverage,
            method="heuristic_fallback",
        )

    @staticmethod
    def llm_summary(shap_top_features, lime_top_features, fraud_score):
        """Level 5: Natural-language summary template."""
        risk_level = (
            "HIGH" if fraud_score >= 0.60 else "MEDIUM" if fraud_score >= 0.35 else "LOW"
        )

        summary = (
            "## Fraud Risk Assessment\n"
            f"**Risk Level: {risk_level}** (Score: {fraud_score:.2%})\n\n"
            "### Key Factors (SHAP):\n"
        )

        if shap_top_features:
            for feature_name, impact in shap_top_features[:5]:
                direction = "increases" if impact > 0 else "decreases"
                summary += (
                    f"- **{feature_name}**: {direction} fraud risk "
                    f"(impact: {abs(impact):.4f})\n"
                )
        else:
            summary += "- No SHAP features were available.\n"

        summary += "\n### Local Explanation (LIME):\n"
        if lime_top_features:
            for feature_name, weight in lime_top_features[:5]:
                summary += f"- {feature_name}: weight = {weight:.4f}\n"
        else:
            summary += "- No LIME features were available.\n"

        print(f"  [LLM Summary] Generated for score={fraud_score:.4f}")
        return summary

    def full_audit(self, instance, fraud_score):
        """Run all 5 levels of XAI for a single transaction."""
        instance = np.asarray(instance, dtype=float)
        print(f"\n=== 5-Level XAI Audit (score={fraud_score:.4f}) ===")

        results = {
            "shap": self.shap_explain(instance.reshape(1, -1)),
            "lime": self.lime_explain(instance),
            "dice": self.dice_explain(instance),
            "anchors": self.anchors_explain(instance),
        }

        shap_top = self._extract_shap_pairs(results["shap"], limit=5)
        lime_top = []
        if results["lime"] is not None and hasattr(results["lime"], "as_list"):
            lime_top = results["lime"].as_list()[:5]

        results["llm_summary"] = self.llm_summary(shap_top, lime_top, fraud_score)
        return results
