"""
MVS-XAI end-to-end training pipeline.

This version performs a proper temporal holdout evaluation:
1. Load and engineer features.
2. Reserve the latest time slice as holdout.
3. Build out-of-fold predictions only on the training slice.
4. Fit the meta-learner on training OOF predictions.
5. Retrain base models on the full training slice and evaluate on holdout.
6. Run drift, fairness, HITL, and XAI on the holdout predictions.

Usage:
    python main_train_pipeline.py --dataset ieee --data_dir data/ --device cuda
    python main_train_pipeline.py --dataset paysim --data_dir data/ --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_scaler import FeatureScalerPipeline
from src.data_pipeline.time_splitter import TemporalSplitter
from src.ensembler.confidence_gating import ConfidenceGating
from src.ensembler.meta_learner import MetaEnsembler
from src.evaluation.ablation import AblationStudy
from src.evaluation.fairness import FairnessAuditor
from src.evaluation.metrics_eval import ModelEvaluator
from src.evaluation.psi_drift import PSIDriftMonitor
from src.evaluation.wasserstein import wasserstein_drift_report
from src.feature_engineering.uid_features import UIDFeatureEngineer, UIDPostProcessor
from src.feature_engineering.view_behavioral import BehavioralExtractor
from src.feature_engineering.view_sequential import SequentialTensorBuilder
from src.feature_engineering.view_tabular import TabularFeatureExtractor
from src.models.base_trees import TreeEnsembleFactory
from src.models.nn_focal_mlp import train_mlp_focal
from src.models.nn_lstm import train_lstm_fold
from src.ops_pipeline.hitl_router import HITLRouter
from src.xai import UltimateXAIAuditor


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def build_feature_frame(df, dataset):
    df = TabularFeatureExtractor.extract_time_features(df)
    df = TabularFeatureExtractor.encode_categoricals(df)
    df = UIDFeatureEngineer.apply_all(df, dataset_name=dataset)
    df = BehavioralExtractor.engineer_velocity(df)
    df = TabularFeatureExtractor.clean_high_nan_columns(df, threshold=0.7)
    return df


def get_active_model_names(device_type):
    model_builders = {
        "RF": lambda: TreeEnsembleFactory.get_random_forest(),
        "XGB": lambda: TreeEnsembleFactory.get_xgboost(use_gpu=(device_type == "cuda")),
        "LGB": lambda: TreeEnsembleFactory.get_lightgbm(),
        "CAT": lambda: TreeEnsembleFactory.get_catboost(use_gpu=(device_type == "cuda")),
        "MLP": None,
        "LSTM": None,
    }

    active = []
    for model_name, builder in model_builders.items():
        if builder is None:
            active.append(model_name)
            continue
        try:
            builder()
            active.append(model_name)
        except Exception as exc:
            print(f"  Skipping {model_name}: {exc}")
    return active


def fit_tree_model(model_name, X_train, y_train, X_val, y_val, device_type):
    builders = {
        "RF": lambda: TreeEnsembleFactory.get_random_forest(),
        "XGB": lambda: TreeEnsembleFactory.get_xgboost(use_gpu=(device_type == "cuda")),
        "LGB": lambda: TreeEnsembleFactory.get_lightgbm(),
        "CAT": lambda: TreeEnsembleFactory.get_catboost(use_gpu=(device_type == "cuda")),
    }
    model = builders[model_name]()
    if model_name == "RF" or X_val is None or y_val is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    if X_val is None:
        return model, None
    return model, model.predict_proba(X_val)[:, 1]


def build_sequence_train_val(X_train_scaled, X_val_scaled, entity_train, entity_val, seq_len=10):
    builder = SequentialTensorBuilder(seq_len=seq_len)
    combined_X = np.vstack([X_train_scaled, X_val_scaled])
    combined_ids = np.concatenate([entity_train, entity_val])
    combined_seq = builder.build_card_sequences(combined_X, combined_ids)
    split_idx = len(X_train_scaled)
    return combined_seq[:split_idx], combined_seq[split_idx:]


def build_gate_applied_matrix(predictions_dict, model_names, gate_weights):
    gated = {
        name: predictions_dict[name] * gate_weights[name]["gate_weight"]
        for name in model_names
    }
    matrix = np.column_stack([gated[name] for name in model_names])
    return gated, matrix


def make_internal_eval_split(X_train_all, y_train_all, entity_train_all, splitter, eval_ratio=0.10):
    try:
        train_idx, val_idx = splitter.split_holdout(X_train_all, test_ratio=eval_ratio)
    except Exception:
        train_idx, val_idx = np.array([], dtype=int), np.array([], dtype=int)

    if len(val_idx) == 0 or len(train_idx) == 0:
        split_idx = max(int(len(X_train_all) * (1 - eval_ratio)), 1)
        split_idx = min(split_idx, len(X_train_all) - 1)
        train_idx = np.arange(0, split_idx)
        val_idx = np.arange(split_idx, len(X_train_all))
        print(f"  Internal validation fallback: train[:{split_idx}] test[{split_idx}:{len(X_train_all)}]")

    return (
        X_train_all[train_idx],
        y_train_all[train_idx],
        entity_train_all[train_idx],
        X_train_all[val_idx],
        y_train_all[val_idx],
        entity_train_all[val_idx],
    )


def predict_mlp(model, X, device, batch_size=4096):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(torch.sigmoid(model(X_batch.to(device))).cpu().numpy())
    return np.concatenate(preds).flatten()


def predict_lstm(model, X_seq, device, batch_size=4096):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.zeros(len(X_seq)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(torch.sigmoid(model(X_batch.to(device))).cpu().numpy())
    return np.concatenate(preds).flatten()


def run_meta_xai_audit(background_matrix, explain_matrix, fraud_scores, model_names, meta_model):
    if len(explain_matrix) == 0:
        return

    top_idx = int(np.argmax(fraud_scores))
    instance = explain_matrix[top_idx]
    fraud_score = float(fraud_scores[top_idx])

    print_section("PHASE 4.5: Meta-Learner XAI Audit")
    print(f"  Auditing holdout sample index {top_idx} with score={fraud_score:.4f}")

    auditor = UltimateXAIAuditor(
        model=meta_model,
        X_background=background_matrix[: min(500, len(background_matrix))],
        feature_names=model_names,
    )
    results = auditor.full_audit(instance, fraud_score)

    print("\n  LIME-style top contributors:")
    if results["lime"] is not None:
        for feat, weight in results["lime"].as_list()[:5]:
            print(f"    {feat:<8} {weight:+.4f}")

    if isinstance(results["dice"], dict):
        print("\n  Counterfactual suggestions:")
        for cf in results["dice"].get("counterfactuals", []):
            print(
                f"    {cf['feature']}: {cf['original_value']:.4f} -> "
                f"{cf['suggested_value']:.4f} (new_score={cf['new_score']:.4f})"
            )

    if results["anchors"] is not None:
        print(f"\n  Anchor-style rules: {results['anchors'].names()}")

    print("\n" + results["llm_summary"])


def save_holdout_artifacts(dataset, holdout_frame, fraud_scores, decisions):
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / f"{dataset}_holdout_predictions.csv"

    saved = holdout_frame.copy()
    saved["fraud_score"] = fraud_scores
    saved["decision"] = decisions
    saved.to_csv(output_path, index=False)
    print(f"\n  Saved holdout predictions to: {output_path.resolve()}")


def main(data_dir="data/", device="cuda", dataset="ieee", test_ratio=0.15, n_splits=5, gap_size=1000):
    import torch

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("=== MVS-XAI Pipeline v4.4.0 ===")
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")

    print_section("PHASE 1: Data Loading and Feature Engineering")
    df = DataLoader.load_dataset(data_dir, dataset=dataset)
    df = build_feature_frame(df, dataset=dataset)

    y = df["isFraud"].values
    base_drop_cols = ["isFraud", "TransactionID", "TransactionDT", "UID"]
    feature_cols = [col for col in df.columns if col not in base_drop_cols]
    X = df[feature_cols].fillna(-999).values
    entity_ids = df["card1"].values if "card1" in df.columns else df["UID"].values

    print(f"\n  Final dataset: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"  Fraud rate: {y.mean() * 100:.2f}%")

    splitter = TemporalSplitter(n_splits=n_splits, gap_size=gap_size)
    train_idx, holdout_idx = splitter.split_holdout(X, test_ratio=test_ratio)

    X_train_all, X_holdout = X[train_idx], X[holdout_idx]
    y_train_all, y_holdout = y[train_idx], y[holdout_idx]
    entity_train_all, entity_holdout = entity_ids[train_idx], entity_ids[holdout_idx]
    df_train_all = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    active_models = get_active_model_names(device.type)
    if len(active_models) < 2:
        raise RuntimeError("Need at least two active base models to run the stacker.")

    print(f"\n  Active models: {active_models}")

    print_section("PHASE 2: Walk-Forward CV on Training Slice")
    oof_train_predictions = {
        model_name: np.zeros(len(X_train_all), dtype=float) for model_name in active_models
    }

    fold_count = 0
    for fold_i, (fold_train_idx, fold_val_idx) in enumerate(splitter.split(X_train_all), start=1):
        fold_count = fold_i
        print(f"\n--- Fold {fold_i} ---")
        X_trn, X_val = X_train_all[fold_train_idx], X_train_all[fold_val_idx]
        y_trn, y_val = y_train_all[fold_train_idx], y_train_all[fold_val_idx]
        entity_trn = entity_train_all[fold_train_idx]
        entity_val = entity_train_all[fold_val_idx]

        scaler = FeatureScalerPipeline()
        X_trn_scaled = scaler.fit_transform(X_trn)
        X_val_scaled = scaler.transform(X_val)

        for model_name in [name for name in active_models if name in {"RF", "XGB", "LGB", "CAT"}]:
            model, preds = fit_tree_model(model_name, X_trn, y_trn, X_val, y_val, device.type)
            oof_train_predictions[model_name][fold_val_idx] = preds

        if "MLP" in active_models:
            mlp_preds, _ = train_mlp_focal(X_trn_scaled, y_trn, X_val_scaled, y_val, device)
            oof_train_predictions["MLP"][fold_val_idx] = mlp_preds

        if "LSTM" in active_models:
            X_trn_seq, X_val_seq = build_sequence_train_val(
                X_trn_scaled, X_val_scaled, entity_trn, entity_val
            )
            lstm_preds, _ = train_lstm_fold(
                X_trn_seq,
                y_trn,
                X_val_seq,
                y_val,
                n_features=X_trn_scaled.shape[1],
                device=device,
            )
            oof_train_predictions["LSTM"][fold_val_idx] = lstm_preds

    if fold_count == 0:
        raise RuntimeError("Temporal CV produced zero folds. Increase dataset size or reduce the gap/test ratio.")

    print_section("PHASE 3: Meta-Learner Fit on Training OOF")
    gating = ConfidenceGating(tau=0.60)
    gated_train_predictions, gate_weights = gating.apply_gating(y_train_all, oof_train_predictions)
    train_oof_matrix = np.column_stack([gated_train_predictions[name] for name in active_models])
    print(f"\n  Train OOF matrix shape: {train_oof_matrix.shape}")

    meta = MetaEnsembler(C=0.01)
    meta.fit(train_oof_matrix, y_train_all)
    meta_train_result = meta.evaluate(train_oof_matrix, y_train_all)

    print_section("PHASE 3.5: Final Base-Model Fit and Holdout Prediction")
    holdout_base_predictions = {
        model_name: np.zeros(len(X_holdout), dtype=float) for model_name in active_models
    }

    (
        X_final_train,
        y_final_train,
        entity_final_train,
        X_final_val,
        y_final_val,
        entity_final_val,
    ) = make_internal_eval_split(X_train_all, y_train_all, entity_train_all, splitter)

    scaler_final = FeatureScalerPipeline()
    X_train_scaled_full = scaler_final.fit_transform(X_final_train)
    X_val_scaled_full = scaler_final.transform(X_final_val)
    X_holdout_scaled = scaler_final.transform(X_holdout)

    for model_name in [name for name in active_models if name in {"RF", "XGB", "LGB", "CAT"}]:
        final_model, _ = fit_tree_model(
            model_name, X_final_train, y_final_train, X_final_val, y_final_val, device.type
        )
        holdout_base_predictions[model_name] = final_model.predict_proba(X_holdout)[:, 1]

    if "MLP" in active_models:
        _, mlp_model = train_mlp_focal(
            X_train_scaled_full, y_final_train, X_val_scaled_full, y_final_val, device
        )
        holdout_base_predictions["MLP"] = predict_mlp(mlp_model, X_holdout_scaled, device)

    if "LSTM" in active_models:
        X_final_train_seq, X_final_val_seq = build_sequence_train_val(
            X_train_scaled_full, X_val_scaled_full, entity_final_train, entity_final_val
        )
        _, lstm_model = train_lstm_fold(
            X_final_train_seq,
            y_final_train,
            X_final_val_seq,
            y_final_val,
            n_features=X_train_scaled_full.shape[1],
            device=device,
        )
        _, X_holdout_seq = build_sequence_train_val(
            X_train_scaled_full, X_holdout_scaled, entity_final_train, entity_holdout
        )
        holdout_base_predictions["LSTM"] = predict_lstm(lstm_model, X_holdout_seq, device)

    _, holdout_matrix = build_gate_applied_matrix(
        holdout_base_predictions, active_models, gate_weights
    )
    print(f"\n  Holdout stacked matrix shape: {holdout_matrix.shape}")

    raw_holdout_probas = meta.predict_proba(holdout_matrix)

    print_section("PHASE 4: Holdout Evaluation and Post-Processing")
    evaluator = ModelEvaluator(y_holdout)
    print("\n  --- Raw Holdout Meta-Learner ---")
    raw_threshold = evaluator.find_optimal_threshold(raw_holdout_probas)
    evaluator.print_comprehensive_report(raw_holdout_probas, threshold=raw_threshold)

    holdout_result = df_holdout[["UID"]].copy()
    holdout_result["fraud_score"] = raw_holdout_probas
    holdout_result = UIDPostProcessor.uid_average_predictions(
        holdout_result, pred_col="fraud_score", uid_col="UID", blend_ratio=0.7
    )
    smoothed_holdout_probas = holdout_result["fraud_score"].values

    print("\n  --- Holdout After UID Smoothing ---")
    smooth_threshold = evaluator.find_optimal_threshold(smoothed_holdout_probas)
    evaluator.print_comprehensive_report(smoothed_holdout_probas, threshold=smooth_threshold)

    print_section("PHASE 4.2: Ablation and Drift")
    ablation = AblationStudy()
    ablation.run_leave_one_out(gated_train_predictions, y_train_all, active_models)

    psi = PSIDriftMonitor()
    psi.monitor_features(X_train_all, X_holdout, feature_cols)
    psi.monitor_score_drift(meta_train_result["probabilities"], raw_holdout_probas)
    wasserstein_drift_report(X_train_all, X_holdout, feature_cols, threshold=0.1)

    print_section("PHASE 4.3: HITL and Fairness on Holdout")
    router = HITLRouter()
    routed_holdout = router.route_transactions(
        holdout_result,
        smoothed_holdout_probas,
        amounts=df_holdout["TransactionAmt"].values if "TransactionAmt" in df_holdout.columns else None,
        prior_txn_counts=df_holdout["Card_Prior_Txn_Count"].values
        if "Card_Prior_Txn_Count" in df_holdout.columns
        else None,
        new_client_flags=df_holdout["is_new_client"].values if "is_new_client" in df_holdout.columns else None,
    )

    fairness_col = "card4" if "card4" in df_holdout.columns else None
    if fairness_col is not None:
        fairness = FairnessAuditor()
        df_fairness = df_holdout[[fairness_col, "isFraud"]].copy()
        df_fairness["Action"] = routed_holdout["decision"].values
        fairness.audit_equalized_odds(
            df_fairness, protect_col=fairness_col, label_col="isFraud", pred_col="Action"
        )

    run_meta_xai_audit(
        background_matrix=train_oof_matrix,
        explain_matrix=holdout_matrix,
        fraud_scores=smoothed_holdout_probas,
        model_names=active_models,
        meta_model=meta.base_lr,
    )

    save_holdout_artifacts(
        dataset=dataset,
        holdout_frame=df_holdout[["TransactionID", "UID", "isFraud"]],
        fraud_scores=smoothed_holdout_probas,
        decisions=routed_holdout["decision"].values,
    )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVS-XAI Training Pipeline")
    parser.add_argument("--dataset", type=str, default="ieee", choices=["ieee", "paysim"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--gap_size", type=int, default=1000)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        device=args.device,
        dataset=args.dataset,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        gap_size=args.gap_size,
    )
