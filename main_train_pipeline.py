"""
MVS-XAI: End-to-End Training Pipeline.
Orchestrates the complete fraud detection workflow matching notebook 06 (v4.3.4).

Usage:
    python main_train_pipeline.py --data_dir data/ --device cuda
"""
import argparse
import sys
import numpy as np
import pandas as pd

# === Data Pipeline ===
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.time_splitter import TemporalSplitter
from src.data_pipeline.feature_scaler import FeatureScalerPipeline

# === Feature Engineering ===
from src.feature_engineering.uid_features import UIDFeatureEngineer, UIDPostProcessor
from src.feature_engineering.view_tabular import TabularFeatureExtractor
from src.feature_engineering.view_behavioral import BehavioralExtractor
from src.feature_engineering.view_sequential import SequentialTensorBuilder

# === Models ===
from src.models.base_trees import TreeEnsembleFactory
from src.models.nn_focal_mlp import train_mlp_focal
from src.models.nn_lstm import train_lstm_fold

# === Ensembler ===
from src.ensembler.meta_learner import MetaEnsembler
from src.ensembler.confidence_gating import ConfidenceGating

# === Evaluation ===
from src.evaluation.metrics_eval import ModelEvaluator
from src.evaluation.ablation import AblationStudy
from src.evaluation.psi_drift import PSIDriftMonitor
from src.evaluation.wasserstein import wasserstein_drift_report
from src.evaluation.fairness import FairnessAuditor

# === XAI ===
from src.xai import UltimateXAIAuditor

# === HITL ===
from src.ops_pipeline.hitl_router import HITLRouter


def main(data_dir='data/', device='cuda'):
    """Run the complete MVS-XAI pipeline."""
    import torch
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"=== MVS-XAI Pipeline v4.3.4 ===")
    print(f"Device: {device}")

    # ──────────────────────────────────────────────
    # PHASE 1: Data Loading & Feature Engineering
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: Data Loading & Feature Engineering")
    print("="*60)

    df = DataLoader.load_and_merge(data_dir)
    df = TabularFeatureExtractor.extract_time_features(df)
    df = TabularFeatureExtractor.encode_categoricals(df)
    df = UIDFeatureEngineer.apply_all(df)
    df = BehavioralExtractor.engineer_velocity(df)
    df = TabularFeatureExtractor.clean_high_nan_columns(df, threshold=0.7)

    # Separate target
    y = df['isFraud'].values
    drop_cols = ['isFraud', 'TransactionID', 'TransactionDT', 'UID']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(-999).values

    print(f"\n  Final dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  Fraud rate: {y.mean()*100:.2f}%")

    # ──────────────────────────────────────────────
    # PHASE 2: Walk-Forward CV + Multi-Model Training
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: Walk-Forward CV Training")
    print("="*60)

    splitter = TemporalSplitter(n_splits=5, gap_size=1000)
    scaler = FeatureScalerPipeline()
    gating = ConfidenceGating(tau=0.60)

    model_names = ['RF', 'XGB', 'LGB', 'CAT', 'MLP', 'LSTM']
    oof_predictions = {name: np.zeros(len(X)) for name in model_names}

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X)):
        print(f"\n--- Fold {fold_i+1} ---")
        X_trn, X_val = X[train_idx], X[val_idx]
        y_trn, y_val = y[train_idx], y[val_idx]

        # Scale for neural models
        X_trn_scaled = scaler.fit_transform(X_trn)
        X_val_scaled = scaler.transform(X_val)

        # View 1+3: Tree models (original features)
        trees = {
            'RF': TreeEnsembleFactory.get_random_forest(),
            'XGB': TreeEnsembleFactory.get_xgboost(),
            'LGB': TreeEnsembleFactory.get_lightgbm(),
            'CAT': TreeEnsembleFactory.get_catboost(),
        }
        for name, model in trees.items():
            model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)] if name != 'RF' else None)
            oof_predictions[name][val_idx] = model.predict_proba(X_val)[:, 1]

        # View 3: MLP (scaled features)
        mlp_preds, _ = train_mlp_focal(X_trn_scaled, y_trn, X_val_scaled, y_val, device)
        oof_predictions['MLP'][val_idx] = mlp_preds

        # View 2: LSTM (sequential features)
        seq_builder = SequentialTensorBuilder(seq_len=10)
        X_trn_seq = seq_builder.build_card_sequences(X_trn_scaled, card_col_idx=0)
        X_val_seq = seq_builder.build_card_sequences(X_val_scaled, card_col_idx=0)
        lstm_preds, _ = train_lstm_fold(X_trn_seq, y_trn, X_val_seq, y_val,
                                        n_features=X_trn_scaled.shape[1], device=device)
        oof_predictions['LSTM'][val_idx] = lstm_preds

    # ──────────────────────────────────────────────
    # PHASE 3: Confidence Gating + Meta-Learner
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3: Meta-Learner Stacking")
    print("="*60)

    # Apply gating
    gated_preds, gate_weights = gating.apply_gating(y, oof_predictions)

    # Stack into matrix
    oof_matrix = np.column_stack([gated_preds[name] for name in model_names])
    print(f"\n  OOF Matrix shape: {oof_matrix.shape}")

    # Fit meta-learner
    meta = MetaEnsembler(C=0.01)
    meta.fit(oof_matrix, y)
    result = meta.evaluate(oof_matrix, y)
    raw_probas = result['probabilities']

    # ──────────────────────────────────────────────
    # PHASE 3.5: UID Post-Processing (Kaggle Winner Technique)
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3.5: UID Post-Processing")
    print("  Replace individual predictions → UID-average (Chris Deotte)")
    print("="*60)

    # Build result DataFrame for post-processing
    df_result = df[['UID']].copy()
    df_result['fraud_score'] = raw_probas

    # Apply UID-average smoothing (70% individual + 30% UID mean)
    df_result = UIDPostProcessor.uid_average_predictions(
        df_result, pred_col='fraud_score', uid_col='UID', blend_ratio=0.7
    )
    smoothed_probas = df_result['fraud_score'].values

    # Compare raw vs smoothed
    evaluator = ModelEvaluator(y)
    print("\n  --- Raw Meta-Learner ---")
    evaluator.find_optimal_threshold(raw_probas)
    print("  --- After UID Smoothing ---")
    evaluator.find_optimal_threshold(smoothed_probas)

    # ──────────────────────────────────────────────
    # PHASE 4: Evaluation & Auditing
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 4: Evaluation & Auditing")
    print("="*60)

    # Full report with smoothed predictions
    evaluator.print_comprehensive_report(smoothed_probas)

    # Ablation Study
    ablation = AblationStudy()
    ablation.run_leave_one_out(oof_predictions, y, model_names)

    # PSI Drift Monitoring
    psi = PSIDriftMonitor()
    mid = len(X) // 2
    psi.monitor_features(X[:mid], X[mid:], feature_cols)

    # Fairness Audit
    fairness = FairnessAuditor()

    # HITL Routing
    router = HITLRouter()
    df_final = router.route_transactions(
        df_result, smoothed_probas,
        amounts=df['TransactionAmt'].values if 'TransactionAmt' in df.columns else None,
        prior_txn_counts=df['Card_Prior_Txn_Count'].values if 'Card_Prior_Txn_Count' in df.columns else None,
        new_client_flags=df['is_new_client'].values if 'is_new_client' in df.columns else None
    )

    print("\n=== Pipeline Complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVS-XAI Training Pipeline')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(data_dir=args.data_dir, device=args.device)
