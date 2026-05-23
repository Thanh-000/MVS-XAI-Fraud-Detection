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
import importlib.util
import json
import random
from types import SimpleNamespace
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
from src.feature_engineering.view_sequential import SequentialTensorBuilder, ZeroSequenceArray
from src.feature_engineering.view_tabular import TabularFeatureExtractor
from src.models.base_trees import TreeEnsembleFactory
from src.ops_pipeline.hitl_router import HITLRouter
from src.xai import UltimateXAIAuditor


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def resolve_device(requested_device):
    """Return a torch device when torch exists, otherwise a CPU-like shim."""
    if not module_available("torch"):
        print("  PyTorch is not installed; neural branches will be unavailable.")
        return SimpleNamespace(type="cpu")

    import torch

    return torch.device(requested_device if torch.cuda.is_available() else "cpu")


def set_global_seed(seed):
    """Seed Python, NumPy, and Torch when available."""

    random.seed(seed)
    np.random.seed(seed)
    if not module_available("torch"):
        return

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_feature_frame(df, dataset, fit_idx=None):
    df = TabularFeatureExtractor.extract_time_features(df)
    df = TabularFeatureExtractor.encode_categoricals(df, fit_idx=fit_idx)
    df = UIDFeatureEngineer.apply_all(df, dataset_name=dataset, fit_idx=fit_idx)
    df = BehavioralExtractor.engineer_velocity(df)
    df = TabularFeatureExtractor.clean_high_nan_columns(
        df,
        threshold=0.7,
        fit_idx=fit_idx,
        preserve_cols=["isFraud", "TransactionID", "TransactionDT", "UID"],
    )
    return df


def resolve_model_profile(preset="auto", model_profile=None):
    if model_profile:
        if model_profile not in {"research", "fast"}:
            raise ValueError("model_profile must be one of: research, fast")
        return model_profile
    return "fast" if preset == "fast_mvs" else "research"


def get_active_model_names(device_type, seed, preset="auto", model_profile=None):
    if preset not in {"auto", "tree", "full_mvs", "fast_mvs"}:
        raise ValueError("preset must be one of: auto, tree, full_mvs, fast_mvs")

    profile = resolve_model_profile(preset, model_profile)

    model_builders = {
        "RF": lambda: TreeEnsembleFactory.get_random_forest(seed=seed, profile=profile),
        "XGB": lambda: TreeEnsembleFactory.get_xgboost(
            use_gpu=(device_type == "cuda"), seed=seed, profile=profile
        ),
        "LGB": lambda: TreeEnsembleFactory.get_lightgbm(seed=seed, profile=profile),
        "CAT": lambda: TreeEnsembleFactory.get_catboost(
            use_gpu=(device_type == "cuda"), seed=seed, profile=profile
        ),
    }

    active = []
    for model_name, builder in model_builders.items():
        try:
            builder()
            active.append(model_name)
        except Exception as exc:
            print(f"  Skipping {model_name}: {exc}")

    if preset == "tree":
        print("  Preset 'tree': neural branches disabled for stable production-style training.")
        return active

    if preset == "fast_mvs":
        if module_available("torch"):
            active.append("MLP")
            print("  Preset 'fast_mvs': fast tree profile + MLP; LSTM disabled for Colab runtime.")
        else:
            print("  Preset 'fast_mvs': PyTorch not installed, using tree branches only.")
        return active

    if module_available("torch"):
        active.extend(["MLP", "LSTM"])
    elif preset == "full_mvs":
        raise ImportError("Preset 'full_mvs' requires PyTorch. Install torch or use --preset auto/tree.")
    else:
        print("  Skipping MLP/LSTM: PyTorch is not installed.")
    return active


def ensure_feature_frame(X, feature_names):
    """Keep tree-model inputs aligned with training feature names."""
    if isinstance(X, pd.DataFrame):
        return X.loc[:, feature_names]
    return pd.DataFrame(X, columns=feature_names)


def prepare_xgb_device_input(X, device_type):
    """Move XGBoost inputs onto the booster device when CuPy is available."""
    if device_type != "cuda":
        return X

    try:
        import cupy as cp
    except ImportError:
        return X

    if isinstance(X, pd.DataFrame):
        return cp.asarray(X.to_numpy(dtype=np.float32))
    return cp.asarray(np.asarray(X, dtype=np.float32))


def fit_tree_model(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    device_type,
    seed,
    feature_names,
    model_profile="research",
):
    builders = {
        "RF": lambda: TreeEnsembleFactory.get_random_forest(seed=seed, profile=model_profile),
        "XGB": lambda: TreeEnsembleFactory.get_xgboost(
            use_gpu=(device_type == "cuda"), seed=seed, profile=model_profile
        ),
        "LGB": lambda: TreeEnsembleFactory.get_lightgbm(seed=seed, profile=model_profile),
        "CAT": lambda: TreeEnsembleFactory.get_catboost(
            use_gpu=(device_type == "cuda"), seed=seed, profile=model_profile
        ),
    }
    model = builders[model_name]()
    X_train_frame = ensure_feature_frame(X_train, feature_names)
    X_val_frame = None if X_val is None else ensure_feature_frame(X_val, feature_names)
    if model_name == "XGB":
        X_train_fit = prepare_xgb_device_input(X_train_frame, device_type)
        X_val_fit = None if X_val_frame is None else prepare_xgb_device_input(X_val_frame, device_type)
    else:
        X_train_fit = X_train_frame
        X_val_fit = X_val_frame
    if model_name == "RF" or X_val is None or y_val is None:
        model.fit(X_train_fit, y_train)
    elif model_name == "XGB":
        model.fit(X_train_fit, y_train, eval_set=[(X_val_fit, y_val)], verbose=False)
    else:
        model.fit(X_train_fit, y_train, eval_set=[(X_val_fit, y_val)])
    if X_val is None:
        return model, None
    return model, model.predict_proba(X_val_fit)[:, 1].astype(np.float32, copy=False)


def build_sequence_train_val(X_train_scaled, X_val_scaled, entity_train, entity_val, seq_len=10):
    builder = SequentialTensorBuilder(seq_len=seq_len)
    combined_ids = np.concatenate([entity_train, entity_val])
    if pd.Index(combined_ids).is_unique:
        n_features = X_train_scaled.shape[1]
        print(
            "  View 2 (Sequential): Unique entities; using lazy zero tensors "
            f"train=({len(X_train_scaled)}, {seq_len}, {n_features}), "
            f"val=({len(X_val_scaled)}, {seq_len}, {n_features})"
        )
        return (
            ZeroSequenceArray(len(X_train_scaled), seq_len, n_features),
            ZeroSequenceArray(len(X_val_scaled), seq_len, n_features),
        )

    combined_X = np.vstack([X_train_scaled, X_val_scaled])
    combined_seq = builder.build_card_sequences(combined_X, combined_ids)
    split_idx = len(X_train_scaled)
    return combined_seq[:split_idx], combined_seq[split_idx:]


def build_gate_applied_matrix(predictions_dict, model_names, gate_weights):
    gated = {
        name: (np.asarray(predictions_dict[name], dtype=np.float32) * gate_weights[name]["gate_weight"]).astype(
            np.float32, copy=False
        )
        for name in model_names
    }
    matrix = np.column_stack([gated[name] for name in model_names]).astype(np.float32, copy=False)
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

    X_work = np.asarray(X_train, dtype=np.float32)
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
            X_work = augmented[feature_names].to_numpy(dtype=np.float32)
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
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.zeros(len(X)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(torch.sigmoid(model(X_batch.to(device))).cpu().numpy())
    return np.concatenate(preds).flatten()


def predict_lstm(model, X_seq, device, batch_size=4096):
    import torch
    from src.models.nn_lstm import iter_sequence_batches

    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in iter_sequence_batches(X_seq, batch_size=batch_size, shuffle=False):
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
    shap_top = auditor._extract_shap_pairs(results["shap"], limit=len(model_names))

    print("\n  LIME-style top contributors:")
    lime_top = []
    if results["lime"] is not None:
        lime_top = results["lime"].as_list()
        for feat, weight in lime_top[:5]:
            print(f"    {feat:<8} {weight:+.4f}")

    dice_counterfactuals = []
    if isinstance(results["dice"], dict):
        dice_counterfactuals = results["dice"].get("counterfactuals", [])
        print("\n  Counterfactual suggestions:")
        for cf in dice_counterfactuals:
            print(
                f"    {cf['feature']}: {cf['original_value']:.4f} -> "
                f"{cf['suggested_value']:.4f} (new_score={cf['new_score']:.4f})"
            )

    anchor_rules = []
    if results["anchors"] is not None:
        anchor_rules = results["anchors"].names()
        print(f"\n  Anchor-style rules: {anchor_rules}")

    print("\n" + results["llm_summary"])
    return {
        "results": results,
        "harness": harness_metrics,
        "instance_index": top_idx,
        "fraud_score": fraud_score,
        "shap_top": shap_top,
        "lime_top": lime_top,
        "dice_counterfactuals": dice_counterfactuals,
        "anchor_rules": anchor_rules,
        "llm_summary": results["llm_summary"],
    }


def _json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_xai_artifacts(
    dataset,
    seed,
    xai_result,
    background_matrix,
    explain_matrix,
    fraud_scores,
    model_names,
    meta_model,
    artifacts_dir="artifacts",
):
    """Persist reportable meta-SHAP artifacts for the stacking layer."""

    artifacts_dir = Path(artifacts_dir)
    xai_dir = artifacts_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    background = np.asarray(background_matrix, dtype=float)
    explain = np.asarray(explain_matrix, dtype=float)
    names = list(model_names)
    coefs = np.asarray(getattr(meta_model, "coef_", np.zeros((1, len(names)))), dtype=float)
    if coefs.ndim == 2:
        coefs = coefs[0]
    if coefs.shape[0] != len(names):
        coefs = np.ones(len(names), dtype=float)

    background_mean = background.mean(axis=0)
    contributions = (explain - background_mean) * coefs
    global_rows = []
    for idx, name in enumerate(names):
        feature_contrib = contributions[:, idx]
        global_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "meta_feature": name,
                "coefficient": float(coefs[idx]),
                "background_mean_score": float(background_mean[idx]),
                "holdout_mean_score": float(explain[:, idx].mean()),
                "mean_signed_contribution": float(feature_contrib.mean()),
                "mean_abs_contribution": float(np.abs(feature_contrib).mean()),
                "std_contribution": float(feature_contrib.std(ddof=1)) if len(feature_contrib) > 1 else 0.0,
                "positive_contribution_rate": float((feature_contrib > 0).mean()),
            }
        )

    global_df = pd.DataFrame(global_rows).sort_values("mean_abs_contribution", ascending=False)
    global_df.to_csv(xai_dir / "meta_shap_global.csv", index=False)

    local_idx = int(xai_result["instance_index"]) if xai_result else int(np.argmax(fraud_scores))
    local_contrib = contributions[local_idx]
    local_rows = []
    for idx, name in enumerate(names):
        local_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "instance_index": local_idx,
                "meta_feature": name,
                "base_model_score": float(explain[local_idx, idx]),
                "background_mean_score": float(background_mean[idx]),
                "coefficient": float(coefs[idx]),
                "local_contribution": float(local_contrib[idx]),
                "abs_local_contribution": float(abs(local_contrib[idx])),
            }
        )
    local_df = pd.DataFrame(local_rows).sort_values("abs_local_contribution", ascending=False)
    local_df.to_csv(xai_dir / "meta_shap_top_risk_local.csv", index=False)

    summary = {
        "dataset": dataset,
        "seed": seed,
        "method": "meta_shap_linear_logit_approximation",
        "scope": "stacking_layer_base_model_outputs",
        "explained_features": names,
        "top_risk_instance_index": local_idx,
        "top_risk_score": float(fraud_scores[local_idx]),
        "global_top_features": global_df.head(len(names)).to_dict(orient="records"),
        "local_top_features": local_df.head(len(names)).to_dict(orient="records"),
        "single_instance_audit": {
            "shap_top": xai_result.get("shap_top", []) if xai_result else [],
            "lime_top": xai_result.get("lime_top", []) if xai_result else [],
            "anchor_rules": xai_result.get("anchor_rules", []) if xai_result else [],
            "dice_counterfactuals": xai_result.get("dice_counterfactuals", []) if xai_result else [],
            "harness": xai_result.get("harness", {}) if xai_result else {},
            "llm_summary": xai_result.get("llm_summary", "") if xai_result else "",
        },
    }
    (xai_dir / "xai_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved XAI artifacts to: {xai_dir.resolve()}")
    return {
        "xai_dir": str(xai_dir.resolve()),
        "meta_shap_global": str((xai_dir / "meta_shap_global.csv").resolve()),
        "meta_shap_local": str((xai_dir / "meta_shap_top_risk_local.csv").resolve()),
        "summary": str((xai_dir / "xai_summary.json").resolve()),
    }


def save_holdout_artifacts(dataset, holdout_frame, fraud_scores, decisions, artifacts_dir="artifacts"):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / f"{dataset}_holdout_predictions.csv"

    saved = holdout_frame.copy()
    saved["fraud_score"] = fraud_scores
    saved["decision"] = decisions
    saved.to_csv(output_path, index=False)
    print(f"\n  Saved holdout predictions to: {output_path.resolve()}")


def _compact_seed_result(result):
    keys = [
        "seed",
        "gated_auc",
        "ungated_auc",
        "smoothed_auc",
        "smoothed_f1",
        "mcnemar_p",
        "latency_ms",
        "meta_train_auc",
        "meta_raw_train_auc",
    ]
    compact = {}
    for key in keys:
        value = result.get(key)
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        compact[key] = value
    return compact


def save_metrics_summary(dataset, preset, seed_results, artifacts_dir="artifacts"):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    compact_results = [_compact_seed_result(result) for result in seed_results]

    summary = {
        "dataset": dataset,
        "preset": preset,
        "n_seeds": len(compact_results),
        "results": compact_results,
    }

    if len(compact_results) > 1:
        for metric in ["gated_auc", "ungated_auc", "smoothed_auc", "smoothed_f1", "latency_ms"]:
            values = np.array([row[metric] for row in compact_results], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))

    output_path = artifacts_dir / f"{dataset}_metrics_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Saved metrics summary to: {output_path.resolve()}")
    return summary


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
                tree_input = ensure_feature_frame(x_tree, feature_names)
                if model_name == "XGB":
                    tree_input = prepare_xgb_device_input(tree_input, device.type)
                per_model_scores[model_name] = float(
                    final_models[model_name].predict_proba(tree_input)[0, 1]
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
            import torch

            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = float(np.median(timings_ms))
    print_section("PHASE 4.6: Latency Benchmark")
    print(f"  Median single-transaction decision latency: {median_ms:.2f} ms")
    print(f"  Target < 50 ms: {'YES' if median_ms < 50.0 else 'NO'}")
    return {"median_ms": median_ms, "target_met": bool(median_ms < 50.0)}


def prepare_pipeline_payload(
    data_dir="data/",
    dataset="ieee",
    test_ratio=0.15,
    n_splits=5,
    gap_size=1000,
    paysim_chunk_size=750000,
    paysim_max_rows=None,
    paysim_step_block_size=24,
):
    print_section("PHASE 1: Data Loading and Feature Engineering")
    loader_kwargs = {}
    if dataset == "paysim":
        loader_kwargs = {
            "chunk_size": paysim_chunk_size,
            "max_rows": paysim_max_rows,
            "step_block_size": paysim_step_block_size,
        }
    df_raw = DataLoader.load_dataset(data_dir, dataset=dataset, **loader_kwargs)
    splitter = TemporalSplitter(n_splits=n_splits, gap_size=gap_size)
    train_idx, holdout_idx = splitter.split_holdout(df_raw, test_ratio=test_ratio)

    df = build_feature_frame(df_raw, dataset=dataset, fit_idx=train_idx)

    y = df["isFraud"].to_numpy(dtype=np.int8, copy=False)
    base_drop_cols = ["isFraud", "TransactionID", "TransactionDT", "UID"]
    feature_cols = [col for col in df.columns if col not in base_drop_cols]
    X = df[feature_cols].fillna(-999).to_numpy(dtype=np.float32, copy=True)
    entity_ids = (
        df["card1"].to_numpy(copy=False)
        if "card1" in df.columns
        else df["UID"].to_numpy(copy=False)
    )

    print(f"\n  Final dataset: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"  Fraud rate: {y.mean() * 100:.2f}%")

    return SimpleNamespace(
        df=df,
        X=X,
        y=y,
        entity_ids=entity_ids,
        feature_cols=feature_cols,
        train_idx=train_idx,
        holdout_idx=holdout_idx,
        splitter=splitter,
    )


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
    paysim_chunk_size=750000,
    paysim_max_rows=None,
    paysim_step_block_size=24,
    preset="auto",
    model_profile=None,
    mlp_epochs=None,
    lstm_epochs=None,
    prepared_payload=None,
    artifacts_dir="artifacts",
):
    set_global_seed(seed)
    device = resolve_device(device)
    model_profile = resolve_model_profile(preset, model_profile)
    mlp_epochs = 6 if mlp_epochs is None and model_profile == "fast" else (15 if mlp_epochs is None else mlp_epochs)
    lstm_epochs = 4 if lstm_epochs is None and model_profile == "fast" else (12 if lstm_epochs is None else lstm_epochs)
    print("=== MVS-XAI Pipeline v4.4.0 ===")
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    print(f"Preset: {preset}")
    print(f"Model profile: {model_profile}")

    if prepared_payload is None:
        payload = prepare_pipeline_payload(
            data_dir=data_dir,
            dataset=dataset,
            test_ratio=test_ratio,
            n_splits=n_splits,
            gap_size=gap_size,
            paysim_chunk_size=paysim_chunk_size,
            paysim_max_rows=paysim_max_rows,
            paysim_step_block_size=paysim_step_block_size,
        )
    else:
        print_section("PHASE 1: Data Loading and Feature Engineering")
        print("  Using precomputed dataset/features shared across seeds.")
        payload = prepared_payload

    df = payload.df
    X = payload.X
    y = payload.y
    entity_ids = payload.entity_ids
    feature_cols = payload.feature_cols
    train_idx = payload.train_idx
    holdout_idx = payload.holdout_idx
    splitter = payload.splitter

    X_train_all, X_holdout = X[train_idx], X[holdout_idx]
    y_train_all, y_holdout = y[train_idx], y[holdout_idx]
    entity_train_all, entity_holdout = entity_ids[train_idx], entity_ids[holdout_idx]
    df_train_all = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    active_models = get_active_model_names(
        device.type, seed, preset=preset, model_profile=model_profile
    )
    if len(active_models) < 2:
        raise RuntimeError("Need at least two active base models to run the stacker.")

    print(f"\n  Active models: {active_models}")

    print_section("PHASE 2: Walk-Forward CV on Training Slice")
    oof_train_predictions = {
        model_name: np.full(len(X_train_all), np.nan, dtype=np.float32) for model_name in active_models
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
                model_profile=model_profile,
            )
            oof_train_predictions[model_name][fold_val_idx] = np.asarray(preds, dtype=np.float32)

        if "MLP" in active_models:
            from src.models.nn_focal_mlp import train_mlp_focal

            mlp_preds, _ = train_mlp_focal(
                X_trn_scaled,
                y_trn,
                X_val_scaled,
                y_val,
                device,
                epochs=mlp_epochs,
                seed=seed + fold_i,
            )
            oof_train_predictions["MLP"][fold_val_idx] = np.asarray(mlp_preds, dtype=np.float32)

        if "LSTM" in active_models:
            from src.models.nn_lstm import train_lstm_fold

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
                epochs=lstm_epochs,
                seed=seed + fold_i,
            )
            oof_train_predictions["LSTM"][fold_val_idx] = np.asarray(lstm_preds, dtype=np.float32)
        gc.collect()

    if fold_count == 0:
        raise RuntimeError("Temporal CV produced zero folds. Increase dataset size or reduce the gap/test ratio.")

    print_section("PHASE 3: Meta-Learner Fit on Training OOF")
    oof_mask = np.ones(len(X_train_all), dtype=bool)
    for preds in oof_train_predictions.values():
        oof_mask &= np.isfinite(preds)

    n_oof = int(oof_mask.sum())
    n_missing_oof = int(len(oof_mask) - n_oof)
    print(f"\n  OOF-covered rows: {n_oof:,} / {len(oof_mask):,}")
    if n_missing_oof:
        print(f"  Excluding {n_missing_oof:,} early training rows without OOF predictions")
    if n_oof == 0 or np.unique(y_train_all[oof_mask]).size < 2:
        raise RuntimeError("Not enough valid OOF rows from temporal CV to fit the meta-learner.")

    y_train_meta = y_train_all[oof_mask]
    df_train_meta = df_train_all.iloc[oof_mask].reset_index(drop=True)
    oof_meta_predictions = {
        name: oof_train_predictions[name][oof_mask] for name in active_models
    }

    raw_train_oof_matrix = np.column_stack([oof_meta_predictions[name] for name in active_models]).astype(
        np.float32, copy=False
    )
    gating = ConfidenceGating(tau=0.60)
    gated_train_predictions, gate_weights = gating.apply_gating(y_train_meta, oof_meta_predictions)
    train_oof_matrix = np.column_stack([gated_train_predictions[name] for name in active_models])
    print(f"\n  Train OOF matrix shape: {train_oof_matrix.shape}")

    meta = MetaEnsembler(C=0.01, seed=seed)
    meta.fit(train_oof_matrix, y_train_meta, feature_names=active_models)
    meta_train_result = meta.evaluate(train_oof_matrix, y_train_meta)
    meta_raw = MetaEnsembler(C=0.01, seed=seed)
    meta_raw.fit(raw_train_oof_matrix, y_train_meta, feature_names=active_models)
    meta_raw_train_result = meta_raw.evaluate(raw_train_oof_matrix, y_train_meta)

    print_section("PHASE 3.5: Final Base-Model Fit and Holdout Prediction")
    holdout_base_predictions = {
        model_name: np.zeros(len(X_holdout), dtype=np.float32) for model_name in active_models
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
            model_profile=model_profile,
        )
        final_models[model_name] = final_model
        holdout_tree_input = ensure_feature_frame(X_holdout, feature_cols)
        if model_name == "XGB":
            holdout_tree_input = prepare_xgb_device_input(holdout_tree_input, device.type)
        holdout_base_predictions[model_name] = final_model.predict_proba(holdout_tree_input)[:, 1].astype(
            np.float32, copy=False
        )

    if "MLP" in active_models:
        from src.models.nn_focal_mlp import train_mlp_focal

        _, mlp_model = train_mlp_focal(
            X_train_scaled_full,
            y_final_train,
            X_val_scaled_full,
            y_final_val,
            device,
            epochs=mlp_epochs,
            seed=seed,
        )
        final_models["MLP"] = mlp_model
        holdout_base_predictions["MLP"] = predict_mlp(mlp_model, X_holdout_scaled, device).astype(
            np.float32, copy=False
        )

    X_holdout_seq = None
    if "LSTM" in active_models:
        from src.models.nn_lstm import train_lstm_fold

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
            epochs=lstm_epochs,
            seed=seed,
        )
        final_models["LSTM"] = lstm_model
        _, X_holdout_seq = build_sequence_train_val(
            X_train_scaled_full, X_holdout_scaled, entity_final_train, entity_holdout
        )
        holdout_base_predictions["LSTM"] = predict_lstm(lstm_model, X_holdout_seq, device).astype(
            np.float32, copy=False
        )

    _, holdout_matrix = build_gate_applied_matrix(
        holdout_base_predictions, active_models, gate_weights
    )
    raw_holdout_matrix = np.column_stack([holdout_base_predictions[name] for name in active_models]).astype(
        np.float32, copy=False
    )
    print(f"\n  Holdout stacked matrix shape: {holdout_matrix.shape}")

    gated_holdout_probas = meta.predict_proba(holdout_matrix)
    ungated_holdout_probas = meta_raw.predict_proba(raw_holdout_matrix)

    print_section("PHASE 4: Holdout Evaluation and Post-Processing")
    evaluator = ModelEvaluator(y_holdout)
    threshold_evaluator = ModelEvaluator(y_train_meta)
    print("\n  --- Gated Holdout Meta-Learner ---")
    gated_threshold = threshold_evaluator.find_optimal_threshold(meta_train_result["probabilities"])
    evaluator.print_comprehensive_report(gated_holdout_probas, threshold=gated_threshold)
    gated_metrics = evaluator.compute_metrics(gated_holdout_probas, threshold=gated_threshold)

    print("\n  --- Ungated Holdout Meta-Learner ---")
    ungated_threshold = threshold_evaluator.find_optimal_threshold(meta_raw_train_result["probabilities"])
    evaluator.print_comprehensive_report(ungated_holdout_probas, threshold=ungated_threshold)
    ungated_metrics = evaluator.compute_metrics(ungated_holdout_probas, threshold=ungated_threshold)

    holdout_result = df_holdout[["UID"]].copy()
    holdout_result["fraud_score"] = gated_holdout_probas
    holdout_result = UIDPostProcessor.uid_average_predictions(
        holdout_result, pred_col="fraud_score", uid_col="UID", blend_ratio=0.7
    )
    smoothed_holdout_probas = holdout_result["fraud_score"].values

    train_oof_result = df_train_meta[["UID"]].copy()
    train_oof_result["fraud_score"] = meta_train_result["probabilities"]
    train_oof_result = UIDPostProcessor.uid_average_predictions(
        train_oof_result, pred_col="fraud_score", uid_col="UID", blend_ratio=0.7
    )
    smoothed_train_oof_probas = train_oof_result["fraud_score"].values

    print("\n  --- Holdout After UID Smoothing ---")
    smooth_threshold = threshold_evaluator.find_optimal_threshold(smoothed_train_oof_probas)
    evaluator.print_comprehensive_report(smoothed_holdout_probas, threshold=smooth_threshold)
    smoothed_metrics = evaluator.compute_metrics(smoothed_holdout_probas, threshold=smooth_threshold)

    print_section("PHASE 4.2: Ablation and Drift")
    ablation = AblationStudy()
    ablation.run_leave_one_out(gated_train_predictions, y_train_meta, active_models)

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
    xai_artifacts = save_xai_artifacts(
        dataset=dataset,
        seed=seed,
        xai_result=xai_result,
        background_matrix=train_oof_matrix,
        explain_matrix=holdout_matrix,
        fraud_scores=smoothed_holdout_probas,
        model_names=active_models,
        meta_model=meta.base_lr,
        artifacts_dir=artifacts_dir,
    )

    mcnemar = mcnemar_test(
        y_holdout,
        (gated_holdout_probas >= gated_threshold).astype(int),
        (ungated_holdout_probas >= ungated_threshold).astype(int),
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
        artifacts_dir=artifacts_dir,
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
        "xai_artifacts": xai_artifacts,
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
    paysim_chunk_size=750000,
    paysim_max_rows=None,
    paysim_step_block_size=24,
    preset="auto",
    model_profile=None,
    mlp_epochs=None,
    lstm_epochs=None,
    artifacts_dir="artifacts",
):
    shared_payload = None
    if n_seeds > 1:
        print_section("SHARED DATA PREPARATION")
        print("  Preparing dataset/features once for all seeds.")
        shared_payload = prepare_pipeline_payload(
            data_dir=data_dir,
            dataset=dataset,
            test_ratio=test_ratio,
            n_splits=n_splits,
            gap_size=gap_size,
            paysim_chunk_size=paysim_chunk_size,
            paysim_max_rows=paysim_max_rows,
            paysim_step_block_size=paysim_step_block_size,
        )

    seed_results = []
    for seed_offset in range(n_seeds):
        current_seed = seed + seed_offset
        seed_artifacts_dir = (
            Path(artifacts_dir) / f"seed_{current_seed}"
            if n_seeds > 1
            else Path(artifacts_dir)
        )
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
                paysim_chunk_size=paysim_chunk_size,
                paysim_max_rows=paysim_max_rows,
                paysim_step_block_size=paysim_step_block_size,
                preset=preset,
                model_profile=model_profile,
                mlp_epochs=mlp_epochs,
                lstm_epochs=lstm_epochs,
                prepared_payload=shared_payload,
                artifacts_dir=seed_artifacts_dir,
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

    save_metrics_summary(
        dataset=dataset,
        preset=preset,
        seed_results=seed_results,
        artifacts_dir=artifacts_dir,
    )
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
    parser.add_argument("--paysim_chunk_size", type=int, default=750000)
    parser.add_argument("--paysim_max_rows", type=int, default=None)
    parser.add_argument("--paysim_step_block_size", type=int, default=24)
    parser.add_argument("--preset", type=str, default="auto", choices=["auto", "tree", "full_mvs", "fast_mvs"])
    parser.add_argument("--model_profile", type=str, default=None, choices=["research", "fast"])
    parser.add_argument("--mlp_epochs", type=int, default=None)
    parser.add_argument("--lstm_epochs", type=int, default=None)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
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
        paysim_chunk_size=args.paysim_chunk_size,
        paysim_max_rows=args.paysim_max_rows,
        paysim_step_block_size=args.paysim_step_block_size,
        preset=args.preset,
        model_profile=args.model_profile,
        mlp_epochs=args.mlp_epochs,
        lstm_epochs=args.lstm_epochs,
        artifacts_dir=args.artifacts_dir,
    )
