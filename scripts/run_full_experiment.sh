#!/usr/bin/env bash
set -euo pipefail

DATASET="both"
DATA_DIR="data"
DEVICE="cuda"
SEED=42
N_SEEDS=3
N_SPLITS=5
GAP_SIZE=1000
TEST_RATIO=0.15
SMOTE_STRATEGY=0.30
PAYSIM_CHUNK_SIZE=750000
PAYSIM_STEP_BLOCK_SIZE=24
ARTIFACTS_ROOT="artifacts/local_full"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --n-seeds) N_SEEDS="$2"; shift 2 ;;
    --n-splits) N_SPLITS="$2"; shift 2 ;;
    --gap-size) GAP_SIZE="$2"; shift 2 ;;
    --test-ratio) TEST_RATIO="$2"; shift 2 ;;
    --smote-strategy) SMOTE_STRATEGY="$2"; shift 2 ;;
    --paysim-chunk-size) PAYSIM_CHUNK_SIZE="$2"; shift 2 ;;
    --paysim-step-block-size) PAYSIM_STEP_BLOCK_SIZE="$2"; shift 2 ;;
    --artifacts-root) ARTIFACTS_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export PYTHONUNBUFFERED=1

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

run_dataset() {
  local name="$1"
  local artifact_dir="$ARTIFACTS_ROOT/${name}_academic_full"
  mkdir -p "$artifact_dir"

  local args=(
    main_train_pipeline.py
    --dataset "$name"
    --data_dir "$DATA_DIR"
    --device "$DEVICE"
    --test_ratio "$TEST_RATIO"
    --n_splits "$N_SPLITS"
    --gap_size "$GAP_SIZE"
    --seed "$SEED"
    --n_seeds "$N_SEEDS"
    --smote_strategy "$SMOTE_STRATEGY"
    --ctgan_samples 0
    --preset full_mvs
    --model_profile research
    --artifacts_dir "$artifact_dir"
  )

  if [[ "$name" == "paysim" ]]; then
    args+=(--paysim_chunk_size "$PAYSIM_CHUNK_SIZE")
    args+=(--paysim_step_block_size "$PAYSIM_STEP_BLOCK_SIZE")
  fi

  echo
  echo "Running full local experiment: $name"
  echo "Artifacts: $artifact_dir"
  "$PYTHON_BIN" "${args[@]}"
}

case "$DATASET" in
  ieee) run_dataset ieee ;;
  paysim) run_dataset paysim ;;
  both)
    run_dataset ieee
    run_dataset paysim
    ;;
  *)
    echo "Invalid --dataset: $DATASET. Use ieee, paysim, or both." >&2
    exit 2
    ;;
esac

