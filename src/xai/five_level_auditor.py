"""
XAI Module — 5-Level Explainability Framework.
Matches notebook 06 (v4.3.4).

Level 1: SHAP (TreeExplainer for trees, KernelExplainer fallback)
Level 2: LIME (Local Instance-level)
Level 3: DiCE (Counterfactual 'what-if' scenarios)
Level 4: Anchors (Rule-based explanations)
Level 5: LLM Summary (Natural language explanation)
"""
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


class UltimateXAIAuditor:
    """5-Level Explainability Framework for fraud detection models.

    Bridges technical model outputs → business-readable explanations.
    """

    def __init__(self, model, X_background, feature_names):
        """Initialize XAI auditor.

        Args:
            model: Trained model with predict_proba method.
            X_background: Background dataset for SHAP/LIME (numpy array).
            feature_names: List of feature names.
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names

    def shap_explain(self, X_explain, max_display=15):
        """Level 1: SHAP global + local explanations.

        Uses TreeExplainer for tree models, KernelExplainer as fallback.
        """
        if shap is None:
            print("  [SHAP] Not installed – pip install shap")
            return None

        try:
            explainer = shap.TreeExplainer(self.model)
        except Exception:
            explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X_background, 100)
            )

        shap_values = explainer.shap_values(X_explain)
        print(f"  [SHAP] Computed for {X_explain.shape[0]} samples, {X_explain.shape[1]} features")
        return shap_values

    def lime_explain(self, instance, num_features=10):
        """Level 2: LIME local explanation for a single transaction.

        Returns top contributing features for the given prediction.
        """
        if LimeTabularExplainer is None:
            print("  [LIME] Not installed – pip install lime")
            return None

        explainer = LimeTabularExplainer(
            self.X_background,
            feature_names=self.feature_names,
            class_names=['Legit', 'Fraud'],
            mode='classification'
        )
        exp = explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features
        )
        print(f"  [LIME] Top {num_features} contributing features for instance")
        return exp

    def dice_explain(self, instance, total_cfs=3):
        """Level 3: DiCE counterfactual explanations.

        Answers: "What minimal changes would flip this prediction?"
        """
        if dice_ml is None:
            print("  [DiCE] Not installed – pip install dice-ml")
            return None

        import pandas as pd
        d = dice_ml.Data(
            dataframe=pd.DataFrame(self.X_background, columns=self.feature_names).assign(
                isFraud=0  # Placeholder
            ),
            continuous_features=self.feature_names,
            outcome_name='isFraud'
        )
        m = dice_ml.Model(model=self.model, backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')
        cf = exp.generate_counterfactuals(
            pd.DataFrame([instance], columns=self.feature_names),
            total_CFs=total_cfs
        )
        print(f"  [DiCE] Generated {total_cfs} counterfactual explanations")
        return cf

    def anchors_explain(self, instance):
        """Level 4: Anchors rule-based explanation.

        Produces if-then rules that "anchor" the prediction.
        """
        if anchor_tabular is None:
            print("  [Anchors] Not installed – pip install anchor-exp")
            return None

        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=['Legit', 'Fraud'],
            feature_names=self.feature_names,
            train_data=self.X_background
        )
        exp = explainer.explain_instance(instance, self.model.predict, threshold=0.95)
        print(f"  [Anchors] Rule: {exp.names()}")
        return exp

    @staticmethod
    def llm_summary(shap_top_features, lime_top_features, fraud_score):
        """Level 5: Generate natural language explanation template.

        In production, this would call an LLM API. Here we generate
        a structured template for human review.
        """
        risk_level = "HIGH" if fraud_score >= 0.60 else \
                     "MEDIUM" if fraud_score >= 0.35 else "LOW"

        summary = (
            f"## Fraud Risk Assessment\n"
            f"**Risk Level: {risk_level}** (Score: {fraud_score:.2%})\n\n"
            f"### Key Factors (SHAP):\n"
        )
        for feat, val in shap_top_features[:5]:
            direction = "↑ increases" if val > 0 else "↓ decreases"
            summary += f"- **{feat}**: {direction} fraud risk (impact: {abs(val):.4f})\n"

        summary += f"\n### Local Explanation (LIME):\n"
        for feat, val in lime_top_features[:5]:
            summary += f"- {feat}: weight = {val:.4f}\n"

        print(f"  [LLM Summary] Generated for score={fraud_score:.4f}")
        return summary

    def full_audit(self, instance, fraud_score):
        """Run all 5 levels of XAI for a single transaction."""
        import numpy as np
        print(f"\n=== 5-Level XAI Audit (score={fraud_score:.4f}) ===")

        results = {}
        results['shap'] = self.shap_explain(instance.reshape(1, -1))
        results['lime'] = self.lime_explain(instance)
        results['dice'] = self.dice_explain(instance)
        results['anchors'] = self.anchors_explain(instance)

        # LLM summary (using SHAP/LIME results)
        shap_top = []
        if results['shap'] is not None:
            sv = results['shap'][0] if isinstance(results['shap'], list) else results['shap'][0]
            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            shap_top = [(self.feature_names[i], sv[i]) for i in top_idx]

        lime_top = []
        if results['lime'] is not None:
            lime_top = results['lime'].as_list()[:5]

        results['llm_summary'] = self.llm_summary(shap_top, lime_top, fraud_score)
        return results
