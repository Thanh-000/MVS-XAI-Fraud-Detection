from __future__ import annotations

import os
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "06_MVS_XAI_Ultimate_IEEE_CIS.ipynb"
IEEE_DATA_DIR = Path(os.environ.get("MVS_XAI_IEEE_DATA_DIR", REPO_ROOT / "data"))


def build_notebook(real_ieee_available: bool) -> nbformat.NotebookNode:
    mode_note = (
        f"IEEE-CIS CSVs were detected in `{IEEE_DATA_DIR}`, so this notebook can be rerun on the real benchmark."
        if real_ieee_available
        else "IEEE-CIS CSVs were not present in this repo snapshot, so this committed notebook executes a synthetic smoke-test. "
        "That keeps the notebook non-empty for review, but it is not a substitute for the final IEEE-CIS experiment."
    )

    intro_md = f"""
    # MVS-XAI Submission Notebook

    This notebook is a reviewer-facing artifact for the MVS-XAI repository.

    - It demonstrates non-empty outputs: metrics, plots, 5-level XAI, and HITL routing.
    - It uses repository code from `src/`, especially `UltimateXAIAuditor`, `ModelEvaluator`, and `HITLRouter`.
    - {mode_note}

    ## Scope note

    - Current implemented benchmark in this repo: **IEEE-CIS Fraud Detection**
    - Current notebook evidence in this repo snapshot: **synthetic smoke-test plus executable XAI/HITL flow**
    - The training pipeline now supports **PaySim**, but this notebook artifact remains IEEE-focused.
    - **ULB Credit Card Fraud** is not yet packaged here and should still be presented as a planned extension.
    """

    dataset_md = f"""
    ## Dataset Source and Reproducibility

    Primary benchmark for this repository:

    - **IEEE-CIS Fraud Detection**
    - Competition page: `https://www.kaggle.com/c/ieee-fraud-detection`
    - Data download page: `https://www.kaggle.com/c/ieee-fraud-detection/data`

    Expected local files:

    - `{IEEE_DATA_DIR}/train_transaction.csv`
    - `{IEEE_DATA_DIR}/train_identity.csv`

    Practical note:

    - Kaggle requires competition access to be accepted before the API download works.
    - If those CSVs are missing, this notebook falls back to a synthetic smoke-test so the committed artifact still contains metrics, plots, XAI output, and routing output.
    """

    setup_code = """
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split

    try:
        get_ipython().run_line_magic("matplotlib", "inline")
    except Exception:
        pass

    def locate_repo_root(start: Path) -> Path:
        current = start.resolve()
        for candidate in [current] + list(current.parents):
            if (candidate / "src").exists() and (candidate / "README.md").exists():
                return candidate
        return current

    REPO_ROOT = locate_repo_root(Path.cwd())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.evaluation.metrics_eval import ModelEvaluator
    from src.ops_pipeline.hitl_router import HITLRouter
    from src.xai.five_level_auditor import UltimateXAIAuditor

    np.random.seed(42)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Using repository imports from: {REPO_ROOT / 'src'}")
    """

    data_code = """
    rng = np.random.default_rng(42)
    n_samples = 1800

    hour = rng.integers(0, 24, size=n_samples)
    is_weekend = rng.binomial(1, 0.28, size=n_samples)
    transaction_amt = np.exp(rng.normal(4.0, 0.85, size=n_samples)).clip(5, 4500)
    velocity_1h = rng.gamma(shape=2.4, scale=1.3, size=n_samples)
    uid_txn_count = rng.poisson(lam=5.5, size=n_samples) + 1
    device_risk = rng.beta(2.0, 5.0, size=n_samples)
    card_distance = rng.beta(1.8, 4.0, size=n_samples)
    merchant_risk = rng.beta(2.2, 4.3, size=n_samples)
    email_match = rng.binomial(1, 0.83, size=n_samples)

    is_night = ((hour <= 5) | (hour >= 23)).astype(int)
    amount_log = np.log1p(transaction_amt)

    logit = (
        -6.8
        + 2.2 * device_risk
        + 1.9 * merchant_risk
        + 1.7 * card_distance
        + 1.1 * is_night
        + 0.9 * (1 - email_match)
        + 0.5 * amount_log
        + 0.35 * np.log1p(velocity_1h)
        - 0.25 * np.log1p(uid_txn_count)
        + rng.normal(0, 0.20, size=n_samples)
    )
    fraud_probability = 1 / (1 + np.exp(-logit))
    is_fraud = rng.binomial(1, fraud_probability)

    df = pd.DataFrame(
        {
            "TransactionAmt": transaction_amt,
            "hour": hour,
            "is_weekend": is_weekend,
            "velocity_1h": velocity_1h,
            "uid_txn_count": uid_txn_count,
            "device_risk": device_risk,
            "card_distance": card_distance,
            "merchant_risk": merchant_risk,
            "email_match": email_match,
            "is_night": is_night,
            "isFraud": is_fraud,
        }
    )

    feature_cols = [
        "TransactionAmt",
        "hour",
        "is_weekend",
        "velocity_1h",
        "uid_txn_count",
        "device_risk",
        "card_distance",
        "merchant_risk",
        "email_match",
        "is_night",
    ]

    print(df.head(3).to_string(index=False))
    print()
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    """

    execution_note_md = """
    ## Execution Mode

    This notebook is generated from repository code and committed with outputs.

    - If IEEE-CIS CSVs are available locally, rerun this notebook in Colab or Jupyter against the real benchmark.
    - In the current repo snapshot, no IEEE-CIS CSVs are present, so the notebook demonstrates the full flow on a synthetic dataset with similar fraud-auditing steps.
    """

    train_code = """
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["isFraud"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["isFraud"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["isFraud"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    test_scores = model.predict_proba(X_test)[:, 1]

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, test_scores):.4f}")
    """

    eval_code = """
    evaluator = ModelEvaluator(y_test)
    optimal_threshold = evaluator.find_optimal_threshold(test_scores)
    evaluator.print_comprehensive_report(test_scores, threshold=optimal_threshold)
    """

    viz_code = """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(test_scores[y_test == 0], bins=25, alpha=0.75, label="Legit")
    axes[0].hist(test_scores[y_test == 1], bins=25, alpha=0.75, label="Fraud")
    axes[0].set_title("Holdout Fraud Scores")
    axes[0].set_xlabel("Predicted fraud probability")
    axes[0].legend()

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.head(6).sort_values().plot(kind="barh", ax=axes[1], color="#d97706")
    axes[1].set_title("Top Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.show()
    """

    xai_code = """
    fraud_candidates = np.where(y_test == 1)[0]
    suspicious_idx = int(
        fraud_candidates[np.argmax(test_scores[fraud_candidates])]
        if len(fraud_candidates) > 0
        else np.argmax(test_scores)
    )
    suspicious_row = test_df.iloc[suspicious_idx]
    suspicious_instance = suspicious_row[feature_cols].to_numpy(dtype=float)
    suspicious_score = float(test_scores[suspicious_idx])

    background = train_df[feature_cols].sample(
        n=min(300, len(train_df)), random_state=42
    ).to_numpy(dtype=float)

    auditor = UltimateXAIAuditor(
        model=model,
        X_background=background,
        feature_names=feature_cols,
    )
    audit = auditor.full_audit(suspicious_instance, suspicious_score)
    """

    xai_note_md = """
    ## XAI Execution Note

    This section intentionally executes `UltimateXAIAuditor` rather than only importing it.

    - SHAP and LIME are produced from repository code.
    - DiCE and Anchors use their real libraries when available.
    - If optional packages are not installed in the runtime, the repository now emits structured fallbacks so the notebook still shows counterfactual-style output, anchor-style rules, and a natural-language summary instead of failing silently.
    """

    xai_render_code = """
    shap_pairs = auditor._extract_shap_pairs(audit["shap"], limit=5)
    lime_pairs = audit["lime"].as_list()[:5] if audit["lime"] is not None else []

    print("Selected transaction:")
    print(suspicious_row.to_string())
    print()

    print("Top SHAP-style factors:")
    print(pd.DataFrame(shap_pairs, columns=["feature", "impact"]).to_string(index=False))
    print()

    print("Top LIME-style factors:")
    print(pd.DataFrame(lime_pairs, columns=["feature", "weight"]).to_string(index=False))
    print()

    if isinstance(audit["dice"], dict):
        dice_df = pd.DataFrame(audit["dice"]["counterfactuals"])
        print("Counterfactual suggestions:")
        print(dice_df.to_string(index=False))
    else:
        print("Counterfactual suggestions generated via dice-ml object.")
    print()

    anchor_names = audit["anchors"].names() if audit["anchors"] is not None else []
    print("Anchor-style rules:", anchor_names)
    if hasattr(audit["anchors"], "precision"):
        print(f"Anchor precision: {audit['anchors'].precision:.3f}")
    if hasattr(audit["anchors"], "coverage"):
        print(f"Anchor coverage:  {audit['anchors'].coverage:.3f}")
    print()

    print(audit["llm_summary"])
    """

    hitl_code = """
    router = HITLRouter(auto_block_threshold=0.60, review_threshold=0.35)
    routed = router.route_transactions(
        test_df[["TransactionAmt"]].copy(),
        fraud_scores=test_scores,
        amounts=test_df["TransactionAmt"].to_numpy(),
        prior_txn_counts=test_df["uid_txn_count"].to_numpy(),
        new_client_flags=(test_df["uid_txn_count"] <= 2).astype(int).to_numpy(),
    )

    routed["actual_isFraud"] = y_test
    print(routed["decision"].value_counts().to_string())
    print()
    print(routed.sort_values("fraud_score", ascending=False).head(8).to_string(index=False))
    """

    final_md = """
    ## Interpretation for submission

    - This notebook now contains executable evidence instead of a blank artifact.
    - The 5-level XAI path is exercised end-to-end through `UltimateXAIAuditor`.
    - For the final academic submission, the next step is to rerun the same notebook in Colab with the real IEEE-CIS CSVs mounted under the configured Drive path.
    - PaySim is now supported in the training pipeline, while ULB Credit Card Fraud should still be described as a future benchmark extension unless those experiments are actually added.
    """

    deviations_md = """
    ## Deviations and Current Limits

    - **Reviewer-facing notebook artifact today**: IEEE-CIS-focused.
    - **Training pipeline coverage today**: IEEE-CIS and PaySim.
    - **Still not packaged in this repository**: ULB Credit Card Fraud.
    - **Reviewer-visible mitigation**: the notebook is committed with outputs, and the scope gap is stated explicitly instead of being left unexplained.
    - **Recommended wording for defense/Q&A**: this submission establishes the MVS-XAI architecture on IEEE-CIS first, adds PaySim as a second runnable benchmark in the training pipeline, and leaves ULB as the next external validation benchmark.
    """

    notebook = new_notebook(
        cells=[
            new_markdown_cell(textwrap.dedent(intro_md).strip()),
            new_markdown_cell(textwrap.dedent(dataset_md).strip()),
            new_code_cell(textwrap.dedent(setup_code).strip()),
            new_markdown_cell(textwrap.dedent(execution_note_md).strip()),
            new_code_cell(textwrap.dedent(data_code).strip()),
            new_code_cell(textwrap.dedent(train_code).strip()),
            new_code_cell(textwrap.dedent(eval_code).strip()),
            new_code_cell(textwrap.dedent(viz_code).strip()),
            new_markdown_cell(textwrap.dedent(xai_note_md).strip()),
            new_code_cell(textwrap.dedent(xai_code).strip()),
            new_code_cell(textwrap.dedent(xai_render_code).strip()),
            new_code_cell(textwrap.dedent(hitl_code).strip()),
            new_markdown_cell(textwrap.dedent(deviations_md).strip()),
            new_markdown_cell(textwrap.dedent(final_md).strip()),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
        },
    )
    return notebook


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    real_ieee_available = (
        (IEEE_DATA_DIR / "train_transaction.csv").exists()
        and (IEEE_DATA_DIR / "train_identity.csv").exists()
    )

    notebook = build_notebook(real_ieee_available=real_ieee_available)
    nbformat.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)

    print(f"Notebook written: {NOTEBOOK_PATH}")
    print(f"Mode: {'real IEEE-CIS rerun ready' if real_ieee_available else 'synthetic smoke-test'}")


if __name__ == "__main__":
    main()
