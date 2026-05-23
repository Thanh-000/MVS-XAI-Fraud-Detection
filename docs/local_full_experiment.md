# Canonical End-to-End Academic Experiment

The project now has one official experiment runner:

```bash
python run_academic_e2e.py
```

This runner has no CLI switches by design. It runs the fixed academic protocol
for both supported datasets:

- IEEE-CIS
- PaySim

The backend remains `main_train_pipeline.py`, but the submitted experiment
should be launched through `run_academic_e2e.py` or the wrapper scripts below.

## Fixed Research Configuration

The runner keeps the original full research settings:

- `preset=full_mvs`
- `model_profile=research`
- `device=cuda`
- `test_ratio=0.15`
- `n_splits=5`
- `gap_size=1000`
- `seed=42`
- `n_seeds=3`
- `smote_strategy=0.30`
- `ctgan_samples=0`
- `ctgan_epochs=30`
- `mlp_epochs=15`
- `lstm_epochs=12`
- `paysim_chunk_size=750000`
- `paysim_step_block_size=24`

The runner also enables runtime accelerators that preserve the protocol:

- KMeansSMOTE remains the oversampling method and keeps `sampling_strategy=0.30`.
- KMeansSMOTE uses MiniBatchKMeans as the clustering backend.
- LSTM sequence batches are materialized lazily.
- BLAS/OpenMP thread counts are capped to reduce oversubscription.

## Hardware Notes

Full PaySim is large: about 6.36M rows before feature engineering. Use:

- 32 GB RAM minimum for IEEE-CIS.
- 64 GB RAM recommended for PaySim full research.
- NVIDIA GPU recommended for PyTorch, XGBoost, and CatBoost acceleration.
- 40 GB free disk minimum; more if keeping multiple artifact runs.

## Setup

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_local_env.ps1
```

Linux/macOS:

```bash
bash scripts/setup_local_env.sh
```

Verify the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_local_env.py
```

or:

```bash
.venv/bin/python scripts/check_local_env.py
```

## Data

Place the dataset files under `data/`.

IEEE-CIS requires:

- `data/train_transaction.csv`
- `data/train_identity.csv`

PaySim requires one CSV under `data/`, for example:

- `data/paysim.csv`
- `data/PS_20174392719_1491204439457_log.csv`
- `data/paysim_log.csv`
- `data/paysim dataset.csv`

## Run

Windows:

```powershell
.\scripts\run_full_experiment.ps1
```

Linux/macOS:

```bash
bash scripts/run_full_experiment.sh
```

Direct Python:

```bash
python run_academic_e2e.py
```

Background Windows run:

```powershell
.\scripts\start_full_local.ps1
```

## Outputs

Artifacts are written under:

- `artifacts/academic_e2e/ieee_academic_full`
- `artifacts/academic_e2e/paysim_academic_full`

Important files:

- `*_metrics_summary.json`
- `seed_*/xai/xai_summary.json`
- `seed_*/xai/meta_shap_global.csv`
- `seed_*/xai/meta_shap_top_risk_local.csv`
- `seed_*/*_holdout_predictions.csv`
