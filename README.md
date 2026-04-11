# MVS-XAI: Multi-View Stacking with Explainable AI for Financial Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

MVS-XAI is a research-grade framework for **digital financial transaction fraud detection** that combines:

- **Multi-View Stacking**: Three specialized feature views (Tabular, Sequential, Behavioral) processed by 6 base models, then combined via a meta-learner ensemble.
- **Explainable AI (XAI)**: 5-level explainability framework (SHAP → LIME → DiCE → Anchors → LLM) for transparency and regulatory compliance.
- **Human-in-the-Loop (HITL)**: 3-tier routing system (AUTO_BLOCK / HITL_REVIEW / ALLOW) for operational deployment.

Evaluated on the **IEEE-CIS Fraud Detection** dataset (590,540 transactions, 3.5% fraud rate, 6,381 competing teams on Kaggle 2019).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  IEEE-CIS Raw Data (Transaction + Identity)             │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Feature Engineering (Kaggle Winner: Chris Deotte)       │
│  D1n = TransactionDT/86400 - D1 → client fingerprint    │
│  UID = card1 + addr1 + D1n → stable pseudo-client-ID    │
│  → Expanding Agg (Amt, C13, M9) → V-PCA → Cross-feat   │
└────┬──────────────────┬──────────────────┬───────────────┘
     ▼                  ▼                  ▼
┌─────────┐     ┌─────────────┐     ┌──────────────┐
│ View 1  │     │   View 2    │     │   View 3     │
│ Tabular │     │ Sequential  │     │ Behavioral   │
│ (2D)    │     │ (3D tensor) │     │ (velocity)   │
└────┬────┘     └──────┬──────┘     └──────┬───────┘
     ▼                 ▼                   ▼
┌─────────┐     ┌──────────┐        ┌──────────────┐
│RF + XGB │     │  LSTM    │        │ Focal MLP    │
│LGB +CAT │     │(PyTorch) │        │ (PyTorch)    │
└────┬────┘     └──────┬───┘        └──────┬───────┘
     │                 │                   │
     ▼                 ▼                   ▼
┌──────────────────────────────────────────────────────────┐
│  OOF Predictions → Confidence Gating (τ=0.60)            │
│  → Meta-Learner: LR(L2, C=0.01) + Platt Calibration     │
│  → UID Post-Processing: 70% individual + 30% UID mean   │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  5-Level XAI Audit + HITL Routing + Drift Monitoring     │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
MVS_XAI/
├── README.md                          # This file
├── config.yaml                        # Hyperparameters & experiment config
├── requirements.txt                   # Python dependencies
├── main_train_pipeline.py             # End-to-end training & evaluation script
│
├── data/
│   └── README.md                      # Dataset download instructions
│
└── src/
    ├── __init__.py
    │
    ├── data_pipeline/
    │   ├── data_loader.py             # Merge Transaction + Identity tables
    │   ├── data_sampler.py            # KMeansSMOTE + CTGAN augmentation
    │   ├── time_splitter.py           # Walk-Forward CV (gap=1000)
    │   └── feature_scaler.py          # StandardScaler for neural models
    │
    ├── feature_engineering/
    │   ├── uid_features.py            # Kaggle Winner: UID + expanding agg + V-PCA
    │   ├── view_tabular.py            # View 1: Time + categorical features
    │   ├── view_behavioral.py         # View 3: Rolling velocity features
    │   └── view_sequential.py         # View 2: 3D tensor for LSTM (T=10)
    │
    ├── models/
    │   ├── base_trees.py              # RF, XGB, LGBM, CatBoost factory
    │   ├── nn_focal_mlp.py            # MLP + Focal Loss (PyTorch)
    │   └── nn_lstm.py                 # LSTM + Focal Loss (PyTorch)
    │
    ├── ensembler/
    │   ├── meta_learner.py            # LR(L2) + Platt Calibration
    │   ├── calibrator.py              # Probability calibration utilities
    │   └── confidence_gating.py       # OOF gating: w = min(1, (AUC/τ)²)
    │
    ├── evaluation/
    │   ├── metrics_eval.py            # AUC, PR-AUC, F1, optimal threshold
    │   ├── ablation.py                # Leave-One-Out ablation study
    │   ├── fairness.py                # Equalized Odds audit
    │   ├── wasserstein.py             # Wasserstein distance drift
    │   └── psi_drift.py              # PSI drift monitoring
    │
    ├── xai/
    │   ├── __init__.py
    │   └── five_level_auditor.py      # 5-Level XAI (SHAP+LIME+DiCE+Anchors+LLM)
    │
    └── ops_pipeline/
        └── hitl_router.py             # HITL 3-tier routing
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd MVS_XAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Download the IEEE-CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data):

```bash
# Using Kaggle API
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

Expected files in `data/`:
- `train_transaction.csv` (590,540 × 394)
- `train_identity.csv` (144,233 × 41)

## Quick Start

```python
from src.feature_engineering.uid_features import UIDFeatureEngineer
from src.data_pipeline.data_loader import DataLoader
from src.models.base_trees import TreeEnsembleFactory
from src.ensembler.meta_learner import MetaEnsembler

# 1. Load & engineer features
df = DataLoader.load_and_merge('data/')
df = UIDFeatureEngineer.apply_all(df)

# 2. Train base models
xgb = TreeEnsembleFactory.get_xgboost()
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# 3. Stack & calibrate
meta = MetaEnsembler(C=0.01)
meta.fit(oof_matrix, y_train)
probas = meta.predict_proba(test_matrix)
```

For the complete research pipeline, see the [main notebook](../notebooks/06_MVS_XAI_Ultimate_IEEE_CIS.ipynb).

## Key Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| XGBoost | n_estimators / lr / depth | 800 / 0.03 / 8 |
| LightGBM | n_estimators / lr / depth | 800 / 0.03 / 8 |
| CatBoost | iterations / lr / depth | 800 / 0.03 / 8 |
| RandomForest | n_estimators / depth | 500 / 15 |
| MLP (Focal) | architecture / γ / α | 256→128→64→1 / 2.0 / 0.75 |
| LSTM | hidden / layers / dropout | 64 / 2 / 0.3 |
| Meta-Learner | C / calibration | 0.01 / Platt (sigmoid) |
| CV | n_splits / gap | 5 / 1000 samples |
| OOF Gating | τ (threshold) | 0.60 |
| HITL | auto_block / review | ≥0.60 / [0.35, 0.60) |

## Citation

```bibtex
@article{mvs_xai_2026,
  title={Digital Financial Transaction Fraud Detection Using
         Explainable Multi-Model Machine Learning},
  year={2026}
}
```

## License

This project is part of an academic research submission.
