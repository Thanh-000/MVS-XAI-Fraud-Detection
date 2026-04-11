from __future__ import annotations

import os
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "07_MVS_XAI_PaySim.ipynb"
PAYSIM_DATA_DIR = Path(os.environ.get("MVS_XAI_PAYSIM_DATA_DIR", REPO_ROOT / "data"))


def build_notebook(real_paysim_available: bool) -> nbformat.NotebookNode:
    mode_note = (
        f"A PaySim CSV was detected under `{PAYSIM_DATA_DIR}`, so this notebook can be rerun on the real benchmark."
        if real_paysim_available
        else "A PaySim CSV was not present in this repo snapshot, so this committed notebook executes a synthetic PaySim-like smoke-test. "
        "That keeps the notebook non-empty for review, but it is not a substitute for the final PaySim experiment."
    )

    intro_md = f"""
    # MVS-XAI PaySim Reviewer Notebook

    This notebook is the reviewer-facing PaySim artifact for the MVS-XAI repository.

    - It demonstrates non-empty outputs: metrics, plots, 5-level XAI, and HITL routing.
    - It uses repository code from `src/`, especially `DataLoader`, `UIDFeatureEngineer`, `BehavioralExtractor`, `ModelEvaluator`, `UltimateXAIAuditor`, and `HITLRouter`.
    - {mode_note}

    ## Scope note

    - Current training pipeline coverage in this repo: **IEEE-CIS** and **PaySim**
    - Current notebook evidence in this file: **PaySim-focused executable artifact**
    - **ULB Credit Card Fraud** is still not packaged and should remain a stated future benchmark extension.
    """

    dataset_md = f"""
    ## Dataset Source and Reproducibility

    PaySim is expected as a single CSV placed under the configured dataset directory.

    Accepted filenames:

    - `{PAYSIM_DATA_DIR}/paysim.csv`
    - `{PAYSIM_DATA_DIR}/PS_20174392719_1491204439457_log.csv`
    - `{PAYSIM_DATA_DIR}/paysim_log.csv`

    Expected core columns:

    - `step`
    - `type`
    - `amount`
    - `nameOrig`
    - `nameDest`
    - `oldbalanceOrg`
    - `newbalanceOrig`
    - `oldbalanceDest`
    - `newbalanceDest`
    - `isFraud`

    Practical note:

    - If the CSV is missing, this notebook falls back to a synthetic PaySim-like smoke-test so the committed artifact still contains metrics, plots, XAI output, and routing output.
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

    from src.data_pipeline.data_loader import DataLoader
    from src.feature_engineering.view_tabular import TabularFeatureExtractor
    from src.feature_engineering.uid_features import UIDFeatureEngineer
    from src.feature_engineering.view_behavioral import BehavioralExtractor
    from src.evaluation.metrics_eval import ModelEvaluator
    from src.ops_pipeline.hitl_router import HITLRouter
    from src.xai.five_level_auditor import UltimateXAIAuditor

    np.random.seed(42)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Using repository imports from: {REPO_ROOT / 'src'}")
    """

    data_code = """
    import os
    data_dir = Path(os.environ.get("MVS_XAI_PAYSIM_DATA_DIR", REPO_ROOT / "data"))
    paysim_candidates = [
        data_dir / "paysim.csv",
        data_dir / "PS_20174392719_1491204439457_log.csv",
        data_dir / "paysim_log.csv",
    ]
    real_paysim_path = next((path for path in paysim_candidates if path.exists()), None)

    if real_paysim_path is not None:
        df = DataLoader.load_dataset(str(data_dir), dataset="paysim")
        source_mode = f"real PaySim CSV: {real_paysim_path.name}"
    else:
        rng = np.random.default_rng(42)
        n_samples = 2400
        steps = np.arange(1, n_samples + 1)
        tx_type = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"], size=n_samples, p=[0.35, 0.25, 0.30, 0.10])
        amount = np.exp(rng.normal(4.2, 0.9, size=n_samples)).clip(1, 50000)
        oldbalance_org = np.exp(rng.normal(7.0, 0.8, size=n_samples)).clip(50, 200000)
        newbalance_orig = np.maximum(oldbalance_org - amount + rng.normal(0, 20, size=n_samples), 0)
        oldbalance_dest = np.exp(rng.normal(6.5, 0.9, size=n_samples)).clip(0, 150000)
        newbalance_dest = np.maximum(oldbalance_dest + amount + rng.normal(0, 20, size=n_samples), 0)
        name_orig = np.array([f"C{rng.integers(1, 250)}" for _ in range(n_samples)])
        name_dest = np.array([f"M{rng.integers(1, 120)}" for _ in range(n_samples)])

        logit = (
            -6.0
            + 1.3 * (tx_type == "TRANSFER").astype(float)
            + 1.1 * (tx_type == "CASH_OUT").astype(float)
            + 0.7 * np.log1p(amount)
            + 0.4 * (oldbalance_org < amount * 1.2).astype(float)
            + 0.8 * (name_orig == "C7").astype(float)
            + rng.normal(0, 0.5, size=n_samples)
        )
        fraud_probability = 1 / (1 + np.exp(-logit))
        is_fraud = rng.binomial(1, fraud_probability)

        raw_paysim = pd.DataFrame(
            {
                "step": steps,
                "type": tx_type,
                "amount": amount,
                "nameOrig": name_orig,
                "oldbalanceOrg": oldbalance_org,
                "newbalanceOrig": newbalance_orig,
                "nameDest": name_dest,
                "oldbalanceDest": oldbalance_dest,
                "newbalanceDest": newbalance_dest,
                "isFraud": is_fraud,
                "isFlaggedFraud": np.zeros(n_samples, dtype=int),
            }
        )

        import tempfile
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name) / "paysim.csv"
        raw_paysim.to_csv(temp_path, index=False)
        df = DataLoader.load_dataset(temp_dir.name, dataset="paysim")
        source_mode = "synthetic PaySim-like smoke-test"

    df = TabularFeatureExtractor.extract_time_features(df)
    df = TabularFeatureExtractor.encode_categoricals(df)
    df = UIDFeatureEngineer.apply_all(df, dataset_name="paysim")
    df = BehavioralExtractor.engineer_velocity(df)
    df = TabularFeatureExtractor.clean_high_nan_columns(df, threshold=0.7)

    keep_features = [
        "TransactionAmt",
        "step",
        "card4",
        "origin_delta",
        "dest_delta",
        "balance_gap",
        "hour",
        "is_weekend",
        "UID_txn_count",
        "UID_amt_zscore",
        "Card_Velocity_7d",
        "Card_Spending_Velocity",
        "Amt_Deviation",
        "Card_Prior_Txn_Count",
        "is_new_client",
    ]
    feature_cols = [col for col in keep_features if col in df.columns]

    print(f"Source mode: {source_mode}")
    print(df[feature_cols + ['isFraud']].head(3).to_string(index=False))
    print()
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    print(f"Feature set used in notebook: {feature_cols}")
    """

    execution_note_md = """
    ## Execution Mode

    This notebook is generated from repository code and committed with outputs.

    - If a real PaySim CSV is available locally, rerun this notebook in Colab or Jupyter against the real dataset.
    - In the current repo snapshot, a synthetic fallback is used when no PaySim CSV is present.
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

    X_train = train_df[feature_cols].fillna(-999).to_numpy()
    y_train = train_df["isFraud"].to_numpy()
    X_test = test_df[feature_cols].fillna(-999).to_numpy()
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
    axes[0].set_title("PaySim Holdout Fraud Scores")
    axes[0].set_xlabel("Predicted fraud probability")
    axes[0].legend()

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.head(8).sort_values().plot(kind="barh", ax=axes[1], color="#0f766e")
    axes[1].set_title("Top Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.show()
    """

    xai_note_md = """
    ## XAI Execution Note

    This section intentionally executes `UltimateXAIAuditor` on a PaySim-style transaction rather than only importing it.

    - SHAP and LIME are produced from repository code.
    - DiCE and Anchors use their real libraries when available.
    - If optional packages are not installed in the runtime, the repository emits structured fallbacks so the notebook still shows counterfactual-style output, anchor-style rules, and a natural-language summary.
    """

    xai_code = """
    fraud_candidates = np.where(y_test == 1)[0]
    suspicious_idx = int(
        fraud_candidates[np.argmax(test_scores[fraud_candidates])]
        if len(fraud_candidates) > 0
        else np.argmax(test_scores)
    )
    suspicious_row = test_df.iloc[suspicious_idx]
    suspicious_instance = suspicious_row[feature_cols].fillna(-999).to_numpy(dtype=float)
    suspicious_score = float(test_scores[suspicious_idx])

    background = train_df[feature_cols].fillna(-999).sample(
        n=min(300, len(train_df)), random_state=42
    ).to_numpy(dtype=float)

    auditor = UltimateXAIAuditor(
        model=model,
        X_background=background,
        feature_names=feature_cols,
    )
    audit = auditor.full_audit(suspicious_instance, suspicious_score)
    """

    xai_render_code = """
    shap_pairs = auditor._extract_shap_pairs(audit["shap"], limit=5)
    lime_pairs = audit["lime"].as_list()[:5] if audit["lime"] is not None else []

    print("Selected transaction:")
    print(suspicious_row[feature_cols + ['isFraud']].to_string())
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
        prior_txn_counts=test_df["Card_Prior_Txn_Count"].to_numpy() if "Card_Prior_Txn_Count" in test_df.columns else None,
        new_client_flags=test_df["is_new_client"].to_numpy() if "is_new_client" in test_df.columns else None,
    )

    routed["actual_isFraud"] = y_test
    print(routed["decision"].value_counts().to_string())
    print()
    print(routed.sort_values("fraud_score", ascending=False).head(8).to_string(index=False))
    """

    deviations_md = """
    ## Deviations and Current Limits

    - **Reviewer-facing notebook artifact today**: PaySim-focused.
    - **Training pipeline coverage today**: IEEE-CIS and PaySim.
    - **Still not packaged in this repository**: ULB Credit Card Fraud.
    - **Reviewer-visible mitigation**: this notebook is committed with outputs, so PaySim is not only a CLI claim.
    - **Recommended wording for defense/Q&A**: IEEE-CIS remains the flagship benchmark, while PaySim serves as a second runnable validation dataset inside the same repository.
    """

    final_md = """
    ## Interpretation for submission

    - This notebook provides a second runnable reviewer artifact alongside the IEEE-focused notebook.
    - The 5-level XAI path is exercised end-to-end on PaySim-style data through `UltimateXAIAuditor`.
    - For the final academic submission, rerun this notebook in Colab with the real PaySim CSV mounted under the configured Drive path.
    - ULB Credit Card Fraud should still be described as a future benchmark extension unless those experiments are actually added.
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
    real_paysim_available = any(
        (PAYSIM_DATA_DIR / filename).exists()
        for filename in ["paysim.csv", "PS_20174392719_1491204439457_log.csv", "paysim_log.csv"]
    )

    notebook = build_notebook(real_paysim_available=real_paysim_available)
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
    print(f"Mode: {'real PaySim rerun ready' if real_paysim_available else 'synthetic smoke-test'}")


if __name__ == "__main__":
    main()
