# MVS-XAI: Multi-View Stacking with Explainable AI for Financial Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

MVS-XAI is a research-grade framework for **digital financial transaction fraud detection** that combines:

- **Multi-View Stacking**: Three specialized feature views (Tabular, Sequential, Behavioral) processed by 6 base models, then combined via a meta-learner ensemble.
- **Explainable AI (XAI)**: 5-level explainability framework (SHAP â†’ LIME â†’ DiCE â†’ Anchors â†’ LLM) for transparency and regulatory compliance.
- **Human-in-the-Loop (HITL)**: 3-tier routing system (AUTO_BLOCK / HITL_REVIEW / ALLOW) for operational deployment.

Evaluated through a dual-benchmark workflow spanning **IEEE-CIS Fraud Detection** (590,540 transactions, 3.5% fraud rate) and **PaySim** (6.3M mobile-money transactions), with a shared training pipeline and reviewer-facing notebook artifacts for both datasets.

## Current Submission Scope

- The main training entry point now supports `--dataset ieee` and `--dataset paysim`.
- Reviewer notebook artifacts are available for both **IEEE-CIS** and **PaySim**.
- **ULB Credit Card Fraud** is still not packaged in this repository.
- The current reviewer notebooks are designed as **full academic experiment artifacts**:
  - They clone the working repository revision when needed.
  - They download IEEE-CIS and PaySim directly with `aria2c` signed URLs.
  - They run the real datasets end to end through `main_train_pipeline.py`.
  - They do not mount Google Drive and do not use generated-data fallbacks.

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  IEEE-CIS Raw Data (Transaction + Identity)             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Feature Engineering (Kaggle Winner: Chris Deotte)       â”‚
â”‚  D1n = TransactionDT/86400 - D1 â†’ client fingerprint    â”‚
â”‚  UID = card1 + addr1 + D1n â†’ stable pseudo-client-ID    â”‚
â”‚  â†’ Expanding Agg (Amt, C13, M9) â†’ V-PCA â†’ Cross-feat   â”‚
â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     â–¼                  â–¼                  â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ View 1  â”‚     â”‚   View 2    â”‚     â”‚   View 3     â”‚
â”‚ Tabular â”‚     â”‚ Sequential  â”‚     â”‚ Behavioral   â”‚
â”‚ (2D)    â”‚     â”‚ (3D tensor) â”‚     â”‚ (velocity)   â”‚
â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
     â–¼                 â–¼                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚RF + XGB â”‚     â”‚  LSTM    â”‚        â”‚ Focal MLP    â”‚
â”‚LGB +CAT â”‚     â”‚(PyTorch) â”‚        â”‚ (PyTorch)    â”‚
â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”˜        â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
     â”‚                 â”‚                   â”‚
     â–¼                 â–¼                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  OOF Predictions â†’ Confidence Gating (Ï„=0.60)            â”‚
â”‚  â†’ Meta-Learner: LR(L2, C=0.01) + Platt Calibration     â”‚
â”‚  â†’ UID Post-Processing: 70% individual + 30% UID mean   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  5-Level XAI Audit + HITL Routing + Drift Monitoring     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Project Structure

```
MVS_XAI/
â”œâ”€â”€ README.md                          # This file
â”œâ”€â”€ config.yaml                        # Hyperparameters & experiment config
â”œâ”€â”€ requirements.txt                   # Python dependencies
â”œâ”€â”€ main_train_pipeline.py             # End-to-end training & evaluation script
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ README.md                      # Dataset download instructions
â”‚
â””â”€â”€ src/
    â”œâ”€â”€ __init__.py
    â”‚
    â”œâ”€â”€ data_pipeline/
    â”‚   â”œâ”€â”€ data_loader.py             # Merge Transaction + Identity tables
    â”‚   â”œâ”€â”€ data_sampler.py            # KMeansSMOTE + CTGAN augmentation
    â”‚   â”œâ”€â”€ time_splitter.py           # Walk-Forward CV (gap=1000)
    â”‚   â””â”€â”€ feature_scaler.py          # StandardScaler for neural models
    â”‚
    â”œâ”€â”€ feature_engineering/
    â”‚   â”œâ”€â”€ uid_features.py            # Kaggle Winner: UID + expanding agg + V-PCA
    â”‚   â”œâ”€â”€ view_tabular.py            # View 1: Time + categorical features
    â”‚   â”œâ”€â”€ view_behavioral.py         # View 3: Rolling velocity features
    â”‚   â””â”€â”€ view_sequential.py         # View 2: 3D tensor for LSTM (T=10)
    â”‚
    â”œâ”€â”€ models/
    â”‚   â”œâ”€â”€ base_trees.py              # RF, XGB, LGBM, CatBoost factory
    â”‚   â”œâ”€â”€ nn_focal_mlp.py            # MLP + Focal Loss (PyTorch)
    â”‚   â””â”€â”€ nn_lstm.py                 # LSTM + Focal Loss (PyTorch)
    â”‚
    â”œâ”€â”€ ensembler/
    â”‚   â”œâ”€â”€ meta_learner.py            # LR(L2) + Platt Calibration
    â”‚   â”œâ”€â”€ calibrator.py              # Probability calibration utilities
    â”‚   â””â”€â”€ confidence_gating.py       # OOF gating: w = min(1, (AUC/Ï„)Â²)
    â”‚
    â”œâ”€â”€ evaluation/
    â”‚   â”œâ”€â”€ metrics_eval.py            # AUC, PR-AUC, F1, optimal threshold
    â”‚   â”œâ”€â”€ ablation.py                # Leave-One-Out ablation study
    fairness.py                # Demographic Parity + Equalized Odds audit
    statistical_tests.py       # McNemar / Cohen's d / paired tests
    adwin_monitor.py           # ADWIN score-drift monitor
    wasserstein.py             # Wasserstein distance drift
    xai_harness.py             # Proxy XAI evaluation harness
    psi_drift.py               # PSI drift monitoring
    â”‚
    â”œâ”€â”€ xai/
    â”‚   â”œâ”€â”€ __init__.py
    â”‚   â””â”€â”€ five_level_auditor.py      # 5-Level XAI (SHAP+LIME+DiCE+Anchors+LLM)
    â”‚
    â””â”€â”€ ops_pipeline/
        â””â”€â”€ hitl_router.py             # HITL 3-tier routing
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

For a local full-research setup, see [docs/local_full_experiment.md](docs/local_full_experiment.md).

Windows:

```powershell
.\scripts\setup_local_env.ps1
.\scripts\run_full_experiment.ps1 -Dataset both -Device cuda
```

Linux/macOS:

```bash
bash scripts/setup_local_env.sh
bash scripts/run_full_experiment.sh --dataset both --device cuda
```

## Dataset Setup

Download the supported datasets into `data/`.

IEEE-CIS Fraud Detection from Kaggle:

- Competition page: `https://www.kaggle.com/c/ieee-fraud-detection`
- Data page: `https://www.kaggle.com/c/ieee-fraud-detection/data`
- Note: Kaggle API download works only after the competition terms have been accepted on the website.

```bash
# Using Kaggle API
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

Expected files in `data/`:
- `train_transaction.csv` (590,540 Ã— 394)
- `train_identity.csv` (144,233 Ã— 41)

PaySim:

- Place one CSV under `data/` with one of these filenames:
- `paysim.csv`
- `PS_20174392719_1491204439457_log.csv`
- `paysim_log.csv`

## Quick Start

```python
from src.feature_engineering.uid_features import UIDFeatureEngineer
from src.data_pipeline.data_loader import DataLoader
from src.models.base_trees import TreeEnsembleFactory
from src.ensembler.meta_learner import MetaEnsembler

# 1. Load & engineer features
df = DataLoader.load_dataset('data/', dataset='ieee')
df = UIDFeatureEngineer.apply_all(df, dataset_name='ieee')

# 2. Train base models
xgb = TreeEnsembleFactory.get_xgboost()
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# 3. Stack & calibrate
meta = MetaEnsembler(C=0.01)
meta.fit(oof_matrix, y_train)
probas = meta.predict_proba(test_matrix)
```

Reviewer notebooks:

- Colab setup helper: [notebooks/00_Colab_Quickstart.ipynb](notebooks/00_Colab_Quickstart.ipynb)
- IEEE-CIS case-study artifact: [notebooks/06_MVS_XAI_Ultimate_IEEE_CIS.ipynb](notebooks/06_MVS_XAI_Ultimate_IEEE_CIS.ipynb)
- PaySim case-study artifact: [notebooks/07_MVS_XAI_PaySim.ipynb](notebooks/07_MVS_XAI_PaySim.ipynb)

The Colab quickstart notebook supports both setup paths:

- clone the repo directly from GitHub
- paste direct dataset links or Google Drive share links
- use a single IEEE-CIS bundle zip link or separate transaction/identity CSV links
- download IEEE-CIS and PaySim into the Colab runtime `data/` folder with `aria2` for direct links

If you need clean artifacts for submission review, regenerate the notebooks after adding the datasets:

```bash
py -3.9 scripts/generate_submission_notebook.py
py -3.9 scripts/generate_paysim_submission_notebook.py
```

The generated notebooks include markdown that explains the current experimental scope, the IEEE-CIS and PaySim benchmark coverage, and the fact that ULB remains future work.

To run the full training pipeline:

```bash
# IEEE-CIS
python main_train_pipeline.py --dataset ieee --data_dir data --device cuda

# PaySim
python main_train_pipeline.py --dataset paysim --data_dir data --device cuda
```

The training pipeline now performs:

- temporal holdout split first
- walk-forward CV only on the training slice
- in-fold KMeansSMOTE for the tree branch, with optional CTGAN synthesis
- meta-learner fit on training OOF predictions
- final base-model refit with an internal validation slice
- separate holdout evaluation, drift audit, fairness audit, HITL routing, meta-level XAI, latency benchmarking, and statistical comparison

Useful runtime controls:

```bash
python main_train_pipeline.py --dataset paysim --data_dir data --device cpu --test_ratio 0.2 --n_splits 5 --gap_size 1000 --n_seeds 5 --smote_strategy 0.3 --ctgan_samples 0
```

Generated holdout predictions are saved under `artifacts/`, for example:

- `artifacts/ieee_holdout_predictions.csv`
- `artifacts/paysim_holdout_predictions.csv`

## Key Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| XGBoost | n_estimators / lr / depth | 800 / 0.03 / 8 |
| LightGBM | n_estimators / lr / depth | 800 / 0.03 / 8 |
| CatBoost | iterations / lr / depth | 800 / 0.03 / 8 |
| RandomForest | n_estimators / depth | 500 / 15 |
| MLP (Focal) | architecture / Î³ / Î± | 256â†’128â†’64â†’1 / 2.0 / 0.75 |
| LSTM | hidden / layers / dropout | 64 / 2 / 0.3 |
| Meta-Learner | C / calibration | 0.01 / Platt (sigmoid) |
| CV | n_splits / gap | 5 / 1000 samples |
| OOF Gating | Ï„ (threshold) | 0.60 |
| HITL | auto_block / review | â‰¥0.60 / [0.35, 0.60) |

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
