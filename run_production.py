"""Production-style CLI wrapper for the current MVS-XAI training pipeline.

This file intentionally delegates to ``main_train_pipeline.main`` instead of
keeping a second, divergent implementation under ``publish_repo``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MVS-XAI pipeline")
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
    parser.add_argument("--preset", type=str, default="auto", choices=["auto", "tree", "full_mvs"])
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    return parser.parse_args()


def run():
    args = parse_args()
    from main_train_pipeline import main

    return main(
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
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    run()
