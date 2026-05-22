# Local Full Experiment Setup

This guide runs the same full research configuration as the academic notebooks:

- `preset=full_mvs`
- `model_profile=research`
- `n_splits=5`
- `n_seeds=3`
- `smote_strategy=0.30`
- PaySim uses all rows by default.

## Hardware Notes

Full PaySim is large: about 6.36M rows before feature engineering. Use a machine with:

- 32 GB RAM minimum for IEEE-CIS.
- 64 GB RAM recommended for PaySim full research.
- NVIDIA GPU recommended for PyTorch, XGBoost, and CatBoost acceleration.
- 40 GB free disk minimum; more if keeping multiple artifact runs.

The pipeline includes memory optimizations for full research mode, but full PaySim remains computationally heavy by design.

## 1. Create Local Environment

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_local_env.ps1
```

Windows PowerShell with CUDA PyTorch wheel index:

```powershell
.\scripts\setup_local_env.ps1 -CudaTorch
```

Linux/macOS:

```bash
bash scripts/setup_local_env.sh
```

Linux with CUDA PyTorch wheel index:

```bash
bash scripts/setup_local_env.sh --cuda-torch
```

After installation, verify:

```powershell
.\.venv\Scripts\python.exe scripts\check_local_env.py
```

or:

```bash
.venv/bin/python scripts/check_local_env.py
```

## 2. Prepare Data

Place the dataset files under `data/`.

IEEE-CIS requires:

- `data/train_transaction.csv`
- `data/train_identity.csv`

PaySim requires one CSV under `data/`, for example:

- `data/paysim.csv`
- `data/PS_20174392719_1491204439457_log.csv`
- `data/paysim_log.csv`
- `data/paysim dataset.csv`

The Colab signed URLs are time-limited. For local experiments, refresh the Kaggle download links or use the Kaggle API after accepting dataset/competition terms.

## 3. Run Full Research Experiments

Run both datasets:

```powershell
.\scripts\run_full_experiment.ps1 -Dataset both -Device cuda
```

Run only IEEE-CIS:

```powershell
.\scripts\run_full_experiment.ps1 -Dataset ieee -Device cuda
```

Run only PaySim:

```powershell
.\scripts\run_full_experiment.ps1 -Dataset paysim -Device cuda
```

Linux/macOS:

```bash
bash scripts/run_full_experiment.sh --dataset both --device cuda
```

CPU fallback is supported but much slower:

```powershell
.\scripts\run_full_experiment.ps1 -Dataset ieee -Device cpu
```

## 4. Direct Python Commands

IEEE-CIS full research:

```bash
python main_train_pipeline.py \
  --dataset ieee \
  --data_dir data \
  --device cuda \
  --test_ratio 0.15 \
  --n_splits 5 \
  --gap_size 1000 \
  --seed 42 \
  --n_seeds 3 \
  --smote_strategy 0.30 \
  --ctgan_samples 0 \
  --preset full_mvs \
  --model_profile research \
  --artifacts_dir artifacts/local_full/ieee_academic_full
```

PaySim full research:

```bash
python main_train_pipeline.py \
  --dataset paysim \
  --data_dir data \
  --device cuda \
  --test_ratio 0.15 \
  --n_splits 5 \
  --gap_size 1000 \
  --seed 42 \
  --n_seeds 3 \
  --smote_strategy 0.30 \
  --ctgan_samples 0 \
  --paysim_chunk_size 750000 \
  --paysim_step_block_size 24 \
  --preset full_mvs \
  --model_profile research \
  --artifacts_dir artifacts/local_full/paysim_academic_full
```

## 5. Outputs

Artifacts are written under:

- `artifacts/local_full/ieee_academic_full`
- `artifacts/local_full/paysim_academic_full`

Important files:

- `*_metrics_summary.json`
- `seed_*/xai/xai_summary.json`
- `seed_*/xai/meta_shap_global.csv`
- `seed_*/xai/meta_shap_top_risk_local.csv`
- `seed_*/*_holdout_predictions.csv`

