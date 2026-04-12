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
import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_scaler import FeatureScalerPipeline
from src.data_pipeline.time_splitter import TemporalSplitter
from src.ensembler.confidence_gating import ConfidenceGating
from src.ensembler.meta_learner import MetaEnsembler
from src.evaluation.adwin_monitor import ADWINDriftMonitor
from src.evaluation.ablation import AblationStudy
from src.evaluation.fairness import FairnessAuditor
from src.evaluation.metrics_eval import ModelEvaluator
from src.evaluation.psi_drift import PSIDriftMonitor
from src.evaluation.statistical_tests import cohens_d, mcnemar_test, paired_ttest
from src.evaluation.wasserstein import wasserstein_drift_report
from src.evaluation.xai_harness import XAIEvaluationHarness
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


def set_global_seed(seed):
    """Seed Python, NumPy, and Torch for reproducible multi-seed evaluation."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_feature_frame(df, dataset):
    df = TabularFeatureExtractor.extract_time_features(df)
    df = TabularFeatureExtractor.encode_categoricals(df)
    df = UIDFeatureEngineer.apply_all(df, dataset_name=dataset)
    df = BehavioralExtractor.engineer_velocity(df)
    df = TabularFeatureExtractor.clean_high_nan_columns(df, threshold=0.7)
    return df


def get_active_model_names(device_type, seed):
    model_builders = {
        "RF": lambda: TreeEnsembleFactory.get_random_forest(seed=seed),
        "XGB": lambda: TreeEnsembleFactory.get_xgboost(use_gpu=(device_type == "cuda"), seed=seed),
        "LGB": lambda: TreeEnsembleFactory.get_lightgbm(seed=seed),
        "CAT": lambda: TreeEnsembleFactory.get_catboost(use_gpu=(device_type == "cuda"), seed=seed),
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


def ensure_feature_frame(X, feature_names):
    """Keep tree-model inputs aligned with training feature names."""
    if isinstance(X, pd.DataFrame):
        return X.loc[:, feature_names]
    return pd.DataFrame(X, columns=feature_names)


def fit_tree_model(model_name, X_train, y_train, X_val, y_val, device_type, seed, feature_names):
    builders = {
        "RF": lambda: TreeEnsembleFactory.get_random_forest(seed=seed),
        "XGB": lambda: TreeEnsembleFactory.get_xgboost(use_gpu=(device_type == "cuda"), seed=seed),
        "LGB": lambda: TreeEnsembleFactory.get_lightgbm(seed=seed),
        "CAT": lambda: TreeEnsembleFactory.get_catboost(use_gpu=(device_type == "cuda"), seed=seed),
    }
    model = builders[model_name]()
    X_train_frame = ensure_feature_frame(X_train, feature_names)
    X_val_frame = None if X_val is None else ensure_feature_frame(X_val, feature_names)
    if model_name == "RF" or X_val is None or y_val is None:
        model.fit(X_train_frame, y_train)
    else:
        model.fit(X_train_frame, y_train, eval_set=[(X_val_frame, y_val)])
    if X_val is None:
        return model, None
    return model, model.predict_proba(X_val_frame)[:, 1]


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


def prepare_tree_training_data(
    X_train,
    y_train,
    feature_names,
    device_type,
    seed,
    smote_strategy=0.30,
    ctgan_samples=0,
    ctgan_epochs=30,
):
    """Apply in-fold imbalance handling for tree models only."""
    from src.data_pipeline.data_sampler import DataBalanceEngine

    X_work = np.asarray(X_train, dtype=float)
    y_work = np.asarray(y_train).astype(int)
    engine = DataBalanceEngine(random_state=seed)

    if smote_strategy and smote_strategy > 0:
        try:
            X_work, y_work = engine.apply_kmeans_smote(X_work, y_work, strategy=smote_strategy)
        except Exception as exc:
            print(f"  KMeansSMOTE skipped: {exc}")

    if ctgan_samples and ctgan_samples > 0:
        try:
            df_train = pd.DataFrame(X_work, columns=feature_names)
            df_train["isFraud"] = y_work
            augmented = engine.apply_ctgan_synthesis(
                df_train,
                target_col="isFraud",
                num_synthetic_samples=ctgan_samples,
                epochs=ctgan_epochs,
                use_gpu=(device_type == "cuda"),
            )
            y_work = augmented["isFraud"].to_numpy(dtype=int)
            X_work = augmented[feature_names].to_numpy(dtype=float)
        except Exception as exc:
            print(f"  CTGAN skipped: {exc}")

    return X_work, y_work


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
        return None

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
    harness = XAIEvaluationHarness(auditor)
    harness_metrics = harness.evaluate(instance, results)

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
    return {"results": results, "harness": harness_metrics, "instance_index": top_idx}


def save_holdout_artifacts(dataset, holdout_frame, fraud_scores, decisions):
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / f"{dataset}_holdout_predictions.csv"

    saved = holdout_frame.copy()
    saved["fraud_score"] = fraud_scores
    saved["decision"] = decisions
    saved.to_csv(output_path, index=False)
    print(f"\n  Saved holdout predictions to: {output_path.resolve()}")


def benchmark_decision_latency(
    final_models,
    active_models,
    holdout_payload,
    gate_weights,
    meta_model,
    device,
    feature_names,
):
    """Measure single-transaction decision latency after feature extraction."""
    import torch

    if not final_models:
        return {"median_ms": float("nan"), "target_met": False}

    x_tree = holdout_payload["tree"]
    x_mlp = holdout_payload["mlp"]
    x_lstm = holdout_payload["lstm"]

    timings_ms = []
    for _ in range(25):
        start = time.perf_counter()
        per_model_scores = {}

        for model_name in active_models:
            if model_name in {"RF", "XGB", "LGB", "CAT"}:
                per_model_scores[model_name] = float(
                    final_models[model_name].predict_proba(
                        ensure_feature_frame(x_tree, feature_names)
                    )[0, 1]
                )
            elif model_name == "MLP":
                per_model_scores[model_name] = float(predict_mlp(final_models["MLP"], x_mlp, device, batch_size=1)[0])
            elif model_name == "LSTM":
                per_model_scores[model_name] = float(predict_lstm(final_models["LSTM"], x_lstm, device, batch_size=1)[0])

        _, stacked = build_gate_applied_matrix(
            {name: np.array([per_model_scores[name]]) for name in active_models},
            active_models,
            gate_weights,
        )
        _ = float(meta_model.predict_proba(stacked)[0])

        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = float(np.median(timings_ms))
    print_section("PHASE 4.6: Latency Benchmark")
    print(f"  Median single-transaction decision latency: {median_ms:.2f} ms")
    print(f"  Target < 50 ms: {'YES' if median_ms < 50.0 else 'NO'}")
    return {"median_ms": median_ms, "target_met": bool(median_ms < 50.0)}


def run_single_seed(
    data_dir="data/",
    device="cuda",
    dataset="ieee",
    test_ratio=0.15,
    n_splits=5,
    gap_size=1000,
    seed=42,
    smote_strategy=0.30,
    ctgan_samples=0,
    ctgan_epochs=30,
):
    import torch

    set_global_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("=== MVS-XAI Pipeline v4.4.0 ===")
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")

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

    active_models = get_active_model_names(device.type, seed)
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
        tree_X_trn, tree_y_trn = prepare_tree_training_data(
            X_trn,
            y_trn,
            feature_cols,
            device.type,
            seed=seed + fold_i,
            smote_strategy=smote_strategy,
            ctgan_samples=ctgan_samples,
            ctgan_epochs=ctgan_epochs,
        )

        for model_name in [name for name in active_models if name in {"RF", "XGB", "LGB", "CAT"}]:
            model, preds = fit_tree_model(
                model_name,
                tree_X_trn,
                tree_y_trn,
                X_val,
                y_val,
                device.type,
                seed=seed + fold_i,
                feature_names=feature_cols,
            )
            oof_train_predictions[model_name][fold_val_idx] = preds

        if "MLP" in active_models:
            mlp_preds, _ = train_mlp_focal(
                X_trn_scaled,
                y_trn,
                X_val_scaled,
                y_val,
                device,
                seed=seed + fold_i,
            )
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
                seed=seed + fold_i,
            )
            oof_train_predictions["LSTM"][fold_val_idx] = lstm_preds
        gc.collect()

    if fold_count == 0:
        raise RuntimeError("Temporal CV produced zero folds. Increase dataset size or reduce the gap/test ratio.")

    print_section("PHASE 3: Meta-Learner Fit on Training OOF")
    raw_train_oof_matrix = np.column_stack([oof_train_predictions[name] for name in active_models])
    gating = ConfidenceGating(tau=0.60)
    gated_train_predictions, gate_weights = gating.apply_gating(y_train_all, oof_train_predictions)
    train_oof_matrix = np.column_stack([gated_train_predictions[name] for name in active_models])
    print(f"\n  Train OOF matrix shape: {train_oof_matrix.shape}")

    meta = MetaEnsembler(C=0.01, seed=seed)
    meta.fit(train_oof_matrix, y_train_all, feature_names=active_models)
    meta_train_result = meta.evaluate(train_oof_matrix, y_train_all)
    meta_raw = MetaEnsembler(C=0.01, seed=seed)
    meta_raw.fit(raw_train_oof_matrix, y_train_all, feature_names=active_models)
    meta_raw_train_result = meta_raw.evaluate(raw_train_oof_matrix, y_train_all)

    print_section("PHASE 3.5: Final Base-Model Fit and Holdout Prediction")
    holdout_base_predictions = {
        model_name: np.zeros(len(X_holdout), dtype=float) for model_name in active_models
    }
    final_models = {}

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
    tree_X_final_train, tree_y_final_train = prepare_tree_training_data(
        X_final_train,
        y_final_train,
        feature_cols,
        device.type,
        seed=seed + 999,
        smote_strategy=smote_strategy,
        ctgan_samples=ctgan_samples,
        ctgan_epochs=ctgan_epochs,
    )

    for model_name in [name for name in active_models if name in {"RF", "XGB", "LGB", "CAT"}]:
        final_model, _ = fit_tree_model(
            model_name,
            tree_X_final_train,
            tree_y_final_train,
            X_final_val,
            y_final_val,
            device.type,
            seed=seed,
            feature_names=feature_cols,
        )
        final_models[model_name] = final_model
        holdout_base_predictions[model_name] = final_model.predict_proba(
            ensure_feature_frame(X_holdout, feature_cols)
        )[:, 1]

    if "MLP" in active_models:
        _, mlp_model = train_mlp_focal(
            X_train_scaled_full,
            y_final_train,
            X_val_scaled_full,
            y_final_val,
            device,
            seed=seed,
        )
        final_models["MLP"] = mlp_model
        holdout_base_predictions["MLP"] = predict_mlp(mlp_model, X_holdout_scaled, device)

    X_holdout_seq = None
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
            seed=seed,
        )
        final_models["LSTM"] = lstm_model
        _, X_holdout_seq = build_sequence_train_val(
            X_train_scaled_full, X_holdout_scaled, entity_final_train, entity_holdout
        )
        holdout_base_predictions["LSTM"] = predict_lstm(lstm_model, X_holdout_seq, device)

    _, holdout_matrix = build_gate_applied_matrix(
        holdout_base_predictions, active_models, gate_weights
    )
    raw_holdout_matrix = np.column_stack([holdout_base_predictions[name] for name in active_models])
    print(f"\n  Holdout stacked matrix shape: {holdout_matrix.shape}")

    gated_holdout_probas = meta.predict_proba(holdout_matrix)
    ungated_holdout_probas = meta_raw.predict_proba(raw_holdout_matrix)

    print_section("PHASE 4: Holdout Evaluation and Post-Processing")
    evaluator = ModelEvaluator(y_holdout)
    print("\n  --- Gated Holdout Meta-Learner ---")
    gated_threshold = evaluator.find_optimal_threshold(gated_holdout_probas)
    evaluator.print_comprehensive_report(gated_holdout_probas, threshold=gated_threshold)
    gated_metrics = evaluator.compute_metrics(gated_holdout_probas, threshold=gated_threshold)

    print("\n  --- Ungated Holdout Meta-Learner ---")
    ungated_threshold = evaluator.find_optimal_threshold(ungated_holdout_probas)
    evaluator.print_comprehensive_report(ungated_holdout_probas, threshold=ungated_threshold)
    ungated_metrics = evaluator.compute_metrics(ungated_holdout_probas, threshold=ungated_threshold)

    holdout_result = df_holdout[["UID"]].copy()
    holdout_result["fraud_score"] = gated_holdout_probas
    holdout_result = UIDPostProcessor.uid_average_predictions(
        holdout_result, pred_col="fraud_score", uid_col="UID", blend_ratio=0.7
    )
    smoothed_holdout_probas = holdout_result["fraud_score"].values

    print("\n  --- Holdout After UID Smoothing ---")
    smooth_threshold = evaluator.find_optimal_threshold(smoothed_holdout_probas)
    evaluator.print_comprehensive_report(smoothed_holdout_probas, threshold=smooth_threshold)
    smoothed_metrics = evaluator.compute_metrics(smoothed_holdout_probas, threshold=smooth_threshold)

    print_section("PHASE 4.2: Ablation and Drift")
    ablation = AblationStudy()
    ablation.run_leave_one_out(gated_train_predictions, y_train_all, active_models)

    psi = PSIDriftMonitor()
    psi.monitor_features(X_train_all, X_holdout, feature_cols)
    psi.monitor_score_drift(meta_train_result["probabilities"], gated_holdout_probas)
    wasserstein_drift_report(X_train_all, X_holdout, feature_cols, threshold=0.1)
    adwin = ADWINDriftMonitor(delta=0.002)
    adwin_result = adwin.monitor_score_stream(meta_train_result["probabilities"], gated_holdout_probas)

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
        fairness_result = fairness.audit_full_report(
            df_fairness, protect_col=fairness_col, label_col="isFraud", pred_col="Action"
        )
    else:
        fairness_result = None

    xai_result = run_meta_xai_audit(
        background_matrix=train_oof_matrix,
        explain_matrix=holdout_matrix,
        fraud_scores=smoothed_holdout_probas,
        model_names=active_models,
        meta_model=meta.base_lr,
    )

    mcnemar = mcnemar_test(
        y_holdout,
        (gated_holdout_probas >= 0.5).astype(int),
        (ungated_holdout_probas >= 0.5).astype(int),
    )
    print_section("PHASE 4.4: Statistical Comparison")
    print("  Gated vs Ungated McNemar:")
    print(f"    b={mcnemar['b']} c={mcnemar['c']} discordant={mcnemar['discordant']}")
    print(f"    statistic={mcnemar['statistic']:.4f} p-value={mcnemar['p_value']:.4f}")

    latency = benchmark_decision_latency(
        final_models=final_models,
        active_models=active_models,
        holdout_payload={
            "tree": X_holdout[:1],
            "mlp": X_holdout_scaled[:1],
            "lstm": X_holdout_seq[:1] if X_holdout_seq is not None else np.zeros((1, 10, X_holdout.shape[1])),
        },
        gate_weights=gate_weights,
        meta_model=meta,
        device=device,
        feature_names=feature_cols,
    )

    save_holdout_artifacts(
        dataset=dataset,
        holdout_frame=df_holdout[["TransactionID", "UID", "isFraud"]],
        fraud_scores=smoothed_holdout_probas,
        decisions=routed_holdout["decision"].values,
    )

    print("\n=== Pipeline Complete ===")
    return {
        "seed": seed,
        "gated_auc": gated_metrics["roc_auc"],
        "ungated_auc": ungated_metrics["roc_auc"],
        "smoothed_auc": smoothed_metrics["roc_auc"],
        "smoothed_f1": smoothed_metrics["f1"],
        "mcnemar_p": mcnemar["p_value"],
        "latency_ms": latency["median_ms"],
        "fairness": fairness_result,
        "adwin": adwin_result,
        "xai": xai_result,
        "meta_train_auc": meta_train_result["auc"],
        "meta_raw_train_auc": meta_raw_train_result["auc"],
    }


def main(
    data_dir="data/",
    device="cuda",
    dataset="ieee",
    test_ratio=0.15,
    n_splits=5,
    gap_size=1000,
    seed=42,
    n_seeds=1,
    smote_strategy=0.30,
    ctgan_samples=0,
    ctgan_epochs=30,
):
    seed_results = []
    for seed_offset in range(n_seeds):
        current_seed = seed + seed_offset
        seed_results.append(
            run_single_seed(
                data_dir=data_dir,
                device=device,
                dataset=dataset,
                test_ratio=test_ratio,
                n_splits=n_splits,
                gap_size=gap_size,
                seed=current_seed,
                smote_strategy=smote_strategy,
                ctgan_samples=ctgan_samples,
                ctgan_epochs=ctgan_epochs,
            )
        )

    if len(seed_results) > 1:
        print_section("MULTI-SEED SUMMARY")
        gated_auc = np.array([result["gated_auc"] for result in seed_results], dtype=float)
        ungated_auc = np.array([result["ungated_auc"] for result in seed_results], dtype=float)
        smoothed_auc = np.array([result["smoothed_auc"] for result in seed_results], dtype=float)
        smoothed_f1 = np.array([result["smoothed_f1"] for result in seed_results], dtype=float)
        latency_ms = np.array([result["latency_ms"] for result in seed_results], dtype=float)

        print(f"  Gated AUC:    {gated_auc.mean():.4f} +/- {gated_auc.std(ddof=1):.4f}")
        print(f"  Ungated AUC:  {ungated_auc.mean():.4f} +/- {ungated_auc.std(ddof=1):.4f}")
        print(f"  Smoothed AUC: {smoothed_auc.mean():.4f} +/- {smoothed_auc.std(ddof=1):.4f}")
        print(f"  Smoothed F1:  {smoothed_f1.mean():.4f} +/- {smoothed_f1.std(ddof=1):.4f}")
        print(f"  Latency (ms): {latency_ms.mean():.2f} +/- {latency_ms.std(ddof=1):.2f}")

        d_value = cohens_d(gated_auc, ungated_auc)
        ttest = paired_ttest(gated_auc, ungated_auc)
        print("\n  Gated vs Ungated across seeds:")
        print(f"    Cohen's d: {d_value:.4f}")
        print(f"    Paired t-test: statistic={ttest['statistic']:.4f} p-value={ttest['p_value']:.4f}")

    return seed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVS-XAI Training Pipeline")
    parser.add_argument("--dataset", type=str, default="ieee", choices=["ieee", "paysim"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--gap_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--smote_strategy", type=float, default=0.30)
    parser.add_argument("--ctgan_samples", type=int, default=0)
    parser.add_argument("--ctgan_epochs", type=int, default=30)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        device=args.device,
        dataset=args.dataset,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        gap_size=args.gap_size,
        seed=args.seed,
        n_seeds=args.n_seeds,
        smote_strategy=args.smote_strategy,
        ctgan_samples=args.ctgan_samples,
        ctgan_epochs=args.ctgan_epochs,
    )
