"""Canonical end-to-end academic experiment runner.

This is the single supported execution path for the project experiment. It
keeps the original research configuration fixed and only enables runtime
accelerators that do not change the experiment design: same datasets, folds,
seeds, model profile, preset, SMOTE ratio, and neural epochs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


DATASET_ORDER = ("ieee", "paysim")

ACADEMIC_CONFIG = {
    "data_dir": "data",
    "device": "cuda",
    "test_ratio": 0.15,
    "n_splits": 5,
    "gap_size": 1000,
    "seed": 42,
    "n_seeds": 3,
    "smote_strategy": 0.30,
    "ctgan_samples": 0,
    "ctgan_epochs": 30,
    "paysim_chunk_size": 750000,
    "paysim_max_rows": None,
    "paysim_step_block_size": 24,
    "preset": "full_mvs",
    "model_profile": "research",
    "mlp_epochs": 15,
    "lstm_epochs": 12,
}


def configure_runtime_accelerators() -> None:
    """Set deterministic runtime accelerators for the canonical run."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MVS_XAI_FEATURE_CACHE", "1")
    os.environ.setdefault("MVS_XAI_FEATURE_CACHE_DIR", "artifacts/cache/features")
    os.environ.setdefault("MVS_XAI_DIAGNOSTIC_MAX_ROWS", "300000")

    # Keep KMeansSMOTE as the experiment method, but use a faster KMeans backend.
    os.environ.setdefault("MVS_XAI_DISABLE_LARGE_ROS", "1")
    os.environ.setdefault("MVS_XAI_KMEANSSMOTE_MINIBATCH", "1")
    os.environ.setdefault("MVS_XAI_KMEANSSMOTE_CLUSTERS", "32")
    os.environ.setdefault("MVS_XAI_KMEANSSMOTE_BATCH_SIZE", "65536")
    os.environ.setdefault("MVS_XAI_KMEANSSMOTE_MAX_ITER", "100")

    # Avoid thread oversubscription when RF, XGB, LGBM, CatBoost, NumPy, and BLAS
    # are all active in the same process.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")


def run_dataset(dataset: str):
    from main_train_pipeline import main

    artifacts_dir = Path("artifacts") / "academic_e2e" / f"{dataset}_academic_full"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config = dict(ACADEMIC_CONFIG)
    config["dataset"] = dataset
    config["artifacts_dir"] = str(artifacts_dir)

    print("\n" + "=" * 72)
    print(f"CANONICAL ACADEMIC E2E RUN: {dataset.upper()}")
    print(f"Artifacts: {artifacts_dir}")
    print("=" * 72)
    return main(**config)


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "run_academic_e2e.py intentionally has no CLI options. "
            "Edit ACADEMIC_CONFIG only if the research protocol changes."
        )

    configure_runtime_accelerators()
    for dataset in DATASET_ORDER:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
