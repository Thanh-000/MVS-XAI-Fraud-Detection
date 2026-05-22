from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
IEEE_NOTEBOOK_PATH = NOTEBOOK_DIR / "06_MVS_XAI_Ultimate_IEEE_CIS.ipynb"
PAYSIM_NOTEBOOK_PATH = NOTEBOOK_DIR / "07_MVS_XAI_PaySim.ipynb"

GIT_REPO_URL = "https://github.com/Thanh-000/MVS-XAI-Fraud-Detection"
GIT_BRANCH = "main"

PAYSIM_BUNDLE_URL = (
    "https://storage.googleapis.com/kaggle-data-sets/3805493/6594115/bundle/archive.zip?"
    "X-Goog-Algorithm=GOOG4-RSA-SHA256&"
    "X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260521%2Fauto%2Fstorage%2Fgoog4_request&"
    "X-Goog-Date=20260521T151835Z&"
    "X-Goog-Expires=259200&"
    "X-Goog-SignedHeaders=host&"
    "X-Goog-Signature=12e4582abf09183c5889f3bdfbefb63157bda570775bc53aff3455b6c60778e134de7c65acd609946e7da085e916b01c55e71b008c56a6e835f1befdb73db9ddb2f8bc406fbce2e929e0c5b18e38d83c51c3a869af788051235fe13a993584404022c75dae3ebf61bd5dbe08d280295d4a8100c68d71fac6a0ea98aece6f4db51da8313ace6adbd8194cc4b5ec5ce05bb57a51fbceb48cba23fcdeb85844c22c720893065dc50b64fbf09b3bcd158d27003a676c48b936d039276adf1d01b87d41e8879f6cf0c44ccf50cfc0ad702b3caf244f9694bfb02422e1bd3a9b2aa9760e87983b5d656ecb1f7af8a269f696483b08b7114e670c8289fda367820443c4"
)

IEEE_BUNDLE_URL = (
    "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/14242/568274/bundle/archive.zip?"
    "GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&"
    "Expires=1779616122&"
    "Signature=G%2FeO37B%2FBQJrcTmmGKF%2BU0TcTfnjSKBLu5o9VZyUXLMBtdvYu5e3bm48K%2BvvkmHCPXxgaIy5BXuE0RYWpNI1TMXGAT%2FJ8nNZmXDxxvx3%2F74uzFmrSgG%2FNChBs9JomN1watDawUUjfQLJOfRkwZI6VRVwMpBRr%2B3O2ZXUbPn%2FMLQPuckyFTRWvjDXnkncT9wjxfUPuNwrHkMDFAdaoCVkISSYSNsonXZvc6AkVEN%2BeckemecT6vfE%2FIoMnDMUheOBR4NQuzu9DD6l0R5WUr2Aw8Mrdqzi00EZY8wN4NGHWbMlo1h%2B5pCe1Y%2FKK9gtw9zxelDUE167PB2c7z3%2FYdi%2BZw%3D%3D&"
    "response-content-disposition=attachment%3B+filename%3Dieee-fraud-detection.zip"
)


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }


def markdown_cell(cell_id: str, source: str):
    cell = new_markdown_cell(textwrap.dedent(source).strip())
    cell["id"] = cell_id
    return cell


def code_cell(cell_id: str, source: str):
    cell = new_code_cell(textwrap.dedent(source).strip())
    cell["id"] = cell_id
    return cell


def repo_config_code(dataset: str) -> str:
    experiment_slug = "ieee_cis" if dataset == "ieee" else "paysim"
    runtime_dir = f"/content/MVS-XAI-{experiment_slug}-Experiment"
    return f"""
    from pathlib import Path
    import os
    import shutil
    import subprocess

    GIT_REPO_URL = {GIT_REPO_URL!r}
    GIT_BRANCH = {GIT_BRANCH!r}
    EXPERIMENT_DATASET = {dataset!r}
    EXPERIMENT_SLUG = {experiment_slug!r}
    RUNTIME_REPO_PATH = Path({runtime_dir!r})

    def has_repo_files(path: Path) -> bool:
        return (path / "main_train_pipeline.py").exists() and (path / "src").exists()

    def running_in_colab_runtime() -> bool:
        if os.environ.get("COLAB_RELEASE_TAG"):
            return True
        try:
            import google.colab  # noqa: F401
            return True
        except Exception:
            return False

    def checkout_requested_revision(path: Path) -> None:
        revision = GIT_BRANCH.strip()
        if not revision:
            return
        if not (path / ".git").exists():
            print(f"Repository path has no .git metadata; skipping checkout: {{path}}")
            return

        print(f"Checking out repository revision: {{revision}}")
        subprocess.run(["git", "fetch", "origin", revision], cwd=path, check=False)
        remote_ref = f"origin/{{revision}}"
        remote_exists = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", remote_ref],
            cwd=path,
            check=False,
        ).returncode == 0

        if remote_exists:
            subprocess.run(["git", "checkout", "-B", revision, remote_ref], cwd=path, check=True)
        else:
            subprocess.run(["git", "checkout", revision], cwd=path, check=True)

        subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, check=True)

    def prepare_runtime_repo(path: Path) -> Path:
        if path.exists() and has_repo_files(path):
            print(f"Using existing experiment repository: {{path}}")
            return path
        if path.exists():
            shutil.rmtree(path)
        subprocess.run(["git", "clone", GIT_REPO_URL, str(path)], check=True)
        print(f"Cloned experiment repository: {{path}}")
        return path

    current = Path.cwd().resolve()
    if running_in_colab_runtime():
        REPO_ROOT = prepare_runtime_repo(RUNTIME_REPO_PATH)
    elif has_repo_files(current):
        REPO_ROOT = current
        print(f"Using current repository: {{REPO_ROOT}}")
    elif RUNTIME_REPO_PATH.exists() and has_repo_files(RUNTIME_REPO_PATH):
        REPO_ROOT = RUNTIME_REPO_PATH
        print(f"Using existing cloned repository: {{REPO_ROOT}}")
    else:
        if RUNTIME_REPO_PATH.exists():
            shutil.rmtree(RUNTIME_REPO_PATH)
        subprocess.run(["git", "clone", GIT_REPO_URL, str(RUNTIME_REPO_PATH)], check=True)
        REPO_ROOT = RUNTIME_REPO_PATH

    checkout_requested_revision(REPO_ROOT)

    os.chdir(REPO_ROOT)
    DATA_DIR = REPO_ROOT / "data" / EXPERIMENT_SLUG
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Experiment dataset: {{EXPERIMENT_DATASET}}")
    print(f"Experiment slug: {{EXPERIMENT_SLUG}}")
    print(f"Repo root: {{REPO_ROOT}}")
    print(f"Data directory: {{DATA_DIR}}")
    print("Google Drive is not mounted or used.")
    """


def environment_code() -> str:
    return """
    from pathlib import Path
    import importlib.util
    import json
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import textwrap
    import time
    import zipfile

    def run_checked(cmd, cwd=REPO_ROOT):
        print("+", " ".join(str(part) for part in cmd))
        subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)

    def module_available(module_name):
        return importlib.util.find_spec(module_name) is not None

    def running_in_colab():
        if os.environ.get("COLAB_RELEASE_TAG"):
            return True
        try:
            import google.colab  # noqa: F401
            return True
        except Exception:
            return False

    def pip_install(packages, required=True):
        packages = list(dict.fromkeys(packages))
        if not packages:
            return False
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            cwd=REPO_ROOT,
            check=False,
        )
        if required and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, packages)
        if result.returncode != 0:
            print("Optional install skipped:", packages)
            return False
        return True

    def ensure_aria2():
        if shutil.which("aria2c"):
            print("aria2c is available.")
            return
        if platform.system().lower() != "linux":
            raise RuntimeError("aria2c is not installed. Run this notebook on Colab/Linux or install aria2.")
        run_checked(["apt-get", "-qq", "update"], cwd=REPO_ROOT)
        run_checked(["apt-get", "-qq", "install", "-y", "aria2"], cwd=REPO_ROOT)

    REQUIRED_MODULES = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("imblearn", "imbalanced-learn"),
        ("river", "river"),
        ("tabulate", "tabulate"),
        ("yaml", "pyyaml"),
        ("shap", "shap"),
        ("lime", "lime"),
    ]
    OPTIONAL_MODULES = [
        ("catboost", "catboost"),
        ("dice_ml", "dice-ml"),
        ("anchor", "anchor-exp>=0.0.2.0"),
    ]

    ensure_aria2()
    did_install = False
    did_install = pip_install(
        [package for module, package in REQUIRED_MODULES if not module_available(module)],
        required=True,
    ) or did_install
    did_install = pip_install(
        [package for module, package in OPTIONAL_MODULES if not module_available(module)],
        required=False,
    ) or did_install

    if did_install and running_in_colab():
        print(
            "Python packages were installed or updated. "
            "Restarting the Colab runtime now to avoid NumPy/SciPy binary mismatch. "
            "After the restart, run the notebook from the first cell again."
        )
        os.kill(os.getpid(), 9)

    import numpy as np
    import pandas as pd

    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy:  {np.__version__}")
    print(f"Pandas: {pd.__version__}")

    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    else:
        print("nvidia-smi not found. The pipeline will use CPU fallback where needed.")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    """


def download_helpers_code() -> str:
    return """
    def download_with_aria2(url: str, output_path: Path, force: bool = False) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and output_path.stat().st_size > 0 and not force:
            print(f"Using existing archive: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")
            return output_path

        cmd = [
            "aria2c",
            "--continue=true",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--file-allocation=none",
            "--summary-interval=30",
            "--dir",
            str(output_path.parent),
            "--out",
            output_path.name,
            url,
        ]
        run_checked(cmd, cwd=REPO_ROOT)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Download failed or produced an empty file: {output_path}")
        print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")
        return output_path

    def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        target_root = extract_dir.resolve()

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                member_target = (target_root / member.filename).resolve()
                if target_root not in [member_target] + list(member_target.parents):
                    raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            zf.extractall(target_root)
        print(f"Extracted: {zip_path} -> {extract_dir}")

    def find_file(root: Path, filename: str) -> Path | None:
        root = Path(root)
        direct = root / filename
        if direct.exists():
            return direct
        matches = sorted(root.rglob(filename), key=lambda path: (len(path.parts), str(path)))
        return matches[0] if matches else None

    def find_csv_with_columns(
        root: Path,
        required_columns: list[str],
        preferred_names: list[str] | None = None,
    ) -> Path:
        root = Path(root)
        required = set(required_columns)
        preferred = {name.lower() for name in (preferred_names or [])}
        csv_paths = sorted(
            root.rglob("*.csv"),
            key=lambda path: (path.name.lower() not in preferred, len(path.parts), str(path).lower()),
        )

        checked = []
        for path in csv_paths:
            try:
                header = pd.read_csv(path, nrows=0)
            except Exception as exc:
                checked.append(f"{path}: unreadable header ({exc})")
                continue

            missing = sorted(required - set(header.columns))
            if not missing:
                print(f"Detected dataset CSV by schema: {path}")
                return path
            checked.append(f"{path}: missing {missing}")

        detail = "\\n".join(checked[:20]) if checked else "No CSV files found."
        raise FileNotFoundError(
            "Could not find a CSV with required columns: "
            f"{', '.join(required_columns)}\\nChecked files:\\n{detail}"
        )

    def move_to_data_root(source: Path, target_name: str) -> Path:
        source = Path(source)
        target = DATA_DIR / target_name
        if source.resolve() == target.resolve():
            print(f"File already in data root: {target}")
            return target
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        print(f"Moved {source} -> {target}")
        return target
    """


def run_pipeline_code(dataset: str) -> str:
    paysim_constants = ""
    paysim_run_config_update = ""
    paysim_pipeline_kwargs_update = ""
    if dataset == "paysim":
        paysim_constants = """
    PIPELINE_PAYSIM_CHUNK_SIZE = 750000
    PIPELINE_PAYSIM_MAX_ROWS = None
    PIPELINE_PAYSIM_STEP_BLOCK_SIZE = 24
"""
        paysim_run_config_update = """
    run_config.update(
        {
            "paysim_chunk_size": PIPELINE_PAYSIM_CHUNK_SIZE,
            "paysim_max_rows": PIPELINE_PAYSIM_MAX_ROWS,
            "paysim_step_block_size": PIPELINE_PAYSIM_STEP_BLOCK_SIZE,
        }
    )
"""
        paysim_pipeline_kwargs_update = """
            pipeline_kwargs.update(
                {
                    "paysim_chunk_size": PIPELINE_PAYSIM_CHUNK_SIZE,
                    "paysim_max_rows": PIPELINE_PAYSIM_MAX_ROWS,
                    "paysim_step_block_size": PIPELINE_PAYSIM_STEP_BLOCK_SIZE,
                }
            )
"""

    return f"""
    import contextlib
    import inspect
    from datetime import datetime, timezone

    from IPython.display import display

    from main_train_pipeline import main as run_mvs_xai_pipeline

    DATASET = "{dataset}"
    METHOD_NAME = "MVS-XAI Triple-View Stacking with Confidence Gating, UID smoothing, XAI, HITL"

    PIPELINE_PRESET = "full_mvs"
    PIPELINE_DEVICE = "cuda"
    PIPELINE_TEST_RATIO = 0.15
    PIPELINE_N_SPLITS = 5
    PIPELINE_GAP_SIZE = 1000
    PIPELINE_SEED = 42
    PIPELINE_N_SEEDS = 3
    PIPELINE_SMOTE_STRATEGY = 0.30
    PIPELINE_CTGAN_SAMPLES = 0
    PIPELINE_CTGAN_EPOCHS = 30
    LOG_TAIL_LINES = 120
{paysim_constants}

    ARTIFACTS_DIR = REPO_ROOT / "artifacts" / f"{{DATASET}}_academic_full"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = ARTIFACTS_DIR / f"{{DATASET}}_academic_full_train.log"

    run_config = {{
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "method": METHOD_NAME,
        "repo_root": str(REPO_ROOT),
        "data_dir": str(DATA_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "preset": PIPELINE_PRESET,
        "device": PIPELINE_DEVICE,
        "test_ratio": PIPELINE_TEST_RATIO,
        "n_splits": PIPELINE_N_SPLITS,
        "gap_size": PIPELINE_GAP_SIZE,
        "seed": PIPELINE_SEED,
        "n_seeds": PIPELINE_N_SEEDS,
        "smote_strategy": PIPELINE_SMOTE_STRATEGY,
        "ctgan_samples": PIPELINE_CTGAN_SAMPLES,
        "ctgan_epochs": PIPELINE_CTGAN_EPOCHS,
    }}
{paysim_run_config_update}
    (ARTIFACTS_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("Research method:", METHOD_NAME)
    print("Training configuration:")
    display(pd.DataFrame([run_config]).T.rename(columns={{0: "value"}}))

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            pipeline_kwargs = {{
                "data_dir": str(DATA_DIR),
                "device": PIPELINE_DEVICE,
                "dataset": DATASET,
                "test_ratio": PIPELINE_TEST_RATIO,
                "n_splits": PIPELINE_N_SPLITS,
                "gap_size": PIPELINE_GAP_SIZE,
                "seed": PIPELINE_SEED,
                "n_seeds": PIPELINE_N_SEEDS,
                "smote_strategy": PIPELINE_SMOTE_STRATEGY,
                "ctgan_samples": PIPELINE_CTGAN_SAMPLES,
                "ctgan_epochs": PIPELINE_CTGAN_EPOCHS,
                "preset": PIPELINE_PRESET,
                "artifacts_dir": str(ARTIFACTS_DIR),
            }}
{paysim_pipeline_kwargs_update}
            signature = inspect.signature(run_mvs_xai_pipeline)
            accepted_kwargs = {{
                key: value
                for key, value in pipeline_kwargs.items()
                if key in signature.parameters
            }}
            dropped_kwargs = sorted(set(pipeline_kwargs) - set(accepted_kwargs))
            if dropped_kwargs:
                print(
                    "Warning: pipeline function does not accept these notebook parameters: "
                    + ", ".join(dropped_kwargs)
                )
                print(
                    "The notebook should have checked out the expected Git branch. "
                    "If this warning persists, rerun the clone/setup cell."
                )
            seed_results = run_mvs_xai_pipeline(**accepted_kwargs)

    print(f"Training log saved to: {{LOG_PATH}}")
    print(f"Artifacts directory: {{ARTIFACTS_DIR}}")
    print("Full training stdout/stderr was captured in the log artifact to avoid Colab output truncation.")

    compact_keys = [
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
    compact_rows = []
    for result in seed_results:
        compact_rows.append({{key: result.get(key) for key in compact_keys if key in result}})
    if compact_rows:
        print("Seed-level result summary:")
        display(pd.DataFrame(compact_rows))

    print(f"Last {{LOG_TAIL_LINES}} lines of the training log:")
    log_lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\\n".join(log_lines[-LOG_TAIL_LINES:]))
    """


def artifact_review_code(dataset: str) -> str:
    return f"""
    from IPython.display import display

    metrics_path = ARTIFACTS_DIR / "{dataset}_metrics_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {{metrics_path}}")

    metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_df = pd.DataFrame(metrics_summary["results"])
    print("Metrics summary artifact:", metrics_path)
    display(metrics_df)

    aggregate_keys = [key for key in metrics_summary if key.endswith("_mean") or key.endswith("_std")]
    if aggregate_keys:
        display(pd.DataFrame([{{key: metrics_summary[key] for key in aggregate_keys}}]).T.rename(columns={{0: "value"}}))

    prediction_files = sorted(ARTIFACTS_DIR.rglob("{dataset}_holdout_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No holdout prediction files found under {{ARTIFACTS_DIR}}")

    print("Holdout prediction artifacts:")
    for path in prediction_files:
        print(" -", path)

    latest_prediction_path = prediction_files[-1]
    predictions = pd.read_csv(latest_prediction_path)
    print(f"Previewing prediction file: {{latest_prediction_path}}")
    print(f"Rows: {{len(predictions):,}}")
    display(predictions.head())

    print("Decision distribution:")
    display(predictions["decision"].value_counts(dropna=False).rename_axis("decision").reset_index(name="count"))

    print("Highest-risk holdout transactions:")
    display(predictions.sort_values("fraud_score", ascending=False).head(10))

    artifact_index = []
    for path in sorted(ARTIFACTS_DIR.rglob("*")):
        if path.is_file():
            artifact_index.append({{
                "path": str(path.relative_to(ARTIFACTS_DIR)),
                "size_mb": round(path.stat().st_size / 1024**2, 3),
            }})
    artifact_index_df = pd.DataFrame(artifact_index)
    display(artifact_index_df)
    artifact_index_df.to_csv(ARTIFACTS_DIR / "artifact_index.csv", index=False)
    """


def academic_evaluation_code(dataset: str) -> str:
    return f"""
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    ACADEMIC_EVAL_DIR = ARTIFACTS_DIR / "academic_evaluation"
    ACADEMIC_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    def parse_seed(path: Path) -> int:
        parent = path.parent.name
        if parent.startswith("seed_"):
            try:
                return int(parent.replace("seed_", ""))
            except ValueError:
                pass
        return -1

    def optimal_f1_threshold(y_true, scores):
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best_idx = int(np.nanargmax(f1_values))
        if best_idx >= len(thresholds):
            threshold = 1.0
        else:
            threshold = float(thresholds[best_idx])
        return {{
            "threshold": threshold,
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_values[best_idx]),
        }}

    def ece_score(y_true, scores, n_bins=10):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        rows = []
        for bin_id in range(n_bins):
            left, right = bins[bin_id], bins[bin_id + 1]
            if bin_id == n_bins - 1:
                mask = (scores >= left) & (scores <= right)
            else:
                mask = (scores >= left) & (scores < right)
            n = int(mask.sum())
            if n == 0:
                rows.append({{
                    "bin": bin_id,
                    "score_min": left,
                    "score_max": right,
                    "n": 0,
                    "mean_score": np.nan,
                    "fraud_rate": np.nan,
                    "abs_gap": np.nan,
                }})
                continue
            mean_score = float(scores[mask].mean())
            fraud_rate = float(y_true[mask].mean())
            abs_gap = abs(mean_score - fraud_rate)
            ece += (n / len(scores)) * abs_gap
            rows.append({{
                "bin": bin_id,
                "score_min": left,
                "score_max": right,
                "n": n,
                "mean_score": mean_score,
                "fraud_rate": fraud_rate,
                "abs_gap": float(abs_gap),
            }})
        return float(ece), rows

    def topk_capture_table(frame, percentages=(0.001, 0.005, 0.01, 0.02, 0.05, 0.10)):
        ranked = frame.sort_values("fraud_score", ascending=False).reset_index(drop=True)
        total_fraud = max(int(ranked["isFraud"].sum()), 1)
        base_rate = float(ranked["isFraud"].mean())
        rows = []
        for pct in percentages:
            n_top = max(1, int(math.ceil(len(ranked) * pct)))
            top = ranked.iloc[:n_top]
            precision = float(top["isFraud"].mean())
            recall = float(top["isFraud"].sum() / total_fraud)
            lift = float(precision / base_rate) if base_rate > 0 else float("nan")
            rows.append({{
                "top_percent": pct * 100,
                "n_reviewed": n_top,
                "fraud_captured": int(top["isFraud"].sum()),
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }})
        return rows

    def decision_policy_metrics(frame):
        y_true = frame["isFraud"].astype(int).to_numpy()
        policies = {{
            "auto_block_only": frame["decision"].eq("AUTO_BLOCK").to_numpy(),
            "auto_block_or_hitl": frame["decision"].isin(["AUTO_BLOCK", "HITL_REVIEW"]).to_numpy(),
        }}
        rows = []
        for policy_name, pred_bool in policies.items():
            pred = pred_bool.astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            rows.append({{
                "policy": policy_name,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "review_or_block_rate": float(pred.mean()),
            }})
        return rows

    def score_quantile_bins(frame, n_bins=10):
        ranked = frame.copy()
        ranked["score_decile"] = pd.qcut(
            ranked["fraud_score"].rank(method="first"),
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        grouped = (
            ranked.groupby("score_decile", observed=True)
            .agg(
                n=("isFraud", "size"),
                fraud_n=("isFraud", "sum"),
                fraud_rate=("isFraud", "mean"),
                mean_score=("fraud_score", "mean"),
                min_score=("fraud_score", "min"),
                max_score=("fraud_score", "max"),
            )
            .reset_index()
            .sort_values("score_decile", ascending=False)
        )
        return grouped

    prediction_files = sorted(ARTIFACTS_DIR.rglob("{dataset}_holdout_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No holdout prediction files found under {{ARTIFACTS_DIR}}")

    seed_rows = []
    topk_rows = []
    policy_rows = []
    decision_rows = []
    calibration_rows = []
    score_bin_frames = []
    frames_by_seed = {{}}

    for prediction_path in prediction_files:
        seed = parse_seed(prediction_path)
        frame = pd.read_csv(prediction_path)
        frame["isFraud"] = frame["isFraud"].astype(int)
        y_true = frame["isFraud"].to_numpy()
        scores = frame["fraud_score"].astype(float).to_numpy()
        frames_by_seed[seed] = frame

        opt = optimal_f1_threshold(y_true, scores)
        ece, cal_rows = ece_score(y_true, scores, n_bins=10)
        pred_opt = (scores >= opt["threshold"]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_opt, labels=[0, 1]).ravel()

        seed_rows.append({{
            "seed": seed,
            "n_holdout": int(len(frame)),
            "fraud_n": int(y_true.sum()),
            "fraud_rate": float(y_true.mean()),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "brier": float(brier_score_loss(y_true, scores)),
            "ece_10bin": ece,
            "best_f1_threshold_holdout_descriptive": opt["threshold"],
            "best_f1_precision_holdout_descriptive": opt["precision"],
            "best_f1_recall_holdout_descriptive": opt["recall"],
            "best_f1_holdout_descriptive": opt["f1"],
            "tp_at_best_f1": int(tp),
            "fp_at_best_f1": int(fp),
            "tn_at_best_f1": int(tn),
            "fn_at_best_f1": int(fn),
        }})

        for row in topk_capture_table(frame):
            row["seed"] = seed
            topk_rows.append(row)

        for row in decision_policy_metrics(frame):
            row["seed"] = seed
            policy_rows.append(row)

        decision_counts = frame["decision"].value_counts(dropna=False)
        for decision, count in decision_counts.items():
            decision_rows.append({{
                "seed": seed,
                "decision": str(decision),
                "count": int(count),
                "rate": float(count / len(frame)),
            }})

        for row in cal_rows:
            row["seed"] = seed
            calibration_rows.append(row)

        score_bins = score_quantile_bins(frame)
        score_bins["seed"] = seed
        score_bin_frames.append(score_bins)

    academic_seed_metrics = pd.DataFrame(seed_rows).sort_values("seed")
    academic_topk = pd.DataFrame(topk_rows).sort_values(["seed", "top_percent"])
    academic_policy = pd.DataFrame(policy_rows).sort_values(["seed", "policy"])
    academic_decisions = pd.DataFrame(decision_rows).sort_values(["seed", "decision"])
    academic_calibration = pd.DataFrame(calibration_rows).sort_values(["seed", "bin"])
    academic_score_bins = pd.concat(score_bin_frames, ignore_index=True)

    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier",
        "ece_10bin",
        "best_f1_holdout_descriptive",
        "best_f1_precision_holdout_descriptive",
        "best_f1_recall_holdout_descriptive",
    ]
    summary_rows = []
    t_975 = 4.303 if len(academic_seed_metrics) == 3 else 1.96
    for metric in metric_cols:
        values = academic_seed_metrics[metric].astype(float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci_half_width = float(t_975 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        summary_rows.append({{
            "metric": metric,
            "mean": float(values.mean()),
            "std": std,
            "min": float(values.min()),
            "max": float(values.max()),
            "ci95_low": float(values.mean() - ci_half_width),
            "ci95_high": float(values.mean() + ci_half_width),
        }})
    academic_metric_summary = pd.DataFrame(summary_rows)

    academic_seed_metrics.to_csv(ACADEMIC_EVAL_DIR / "academic_seed_metrics.csv", index=False)
    academic_metric_summary.to_csv(ACADEMIC_EVAL_DIR / "academic_metric_summary.csv", index=False)
    academic_topk.to_csv(ACADEMIC_EVAL_DIR / "academic_topk_capture.csv", index=False)
    academic_policy.to_csv(ACADEMIC_EVAL_DIR / "academic_policy_metrics.csv", index=False)
    academic_decisions.to_csv(ACADEMIC_EVAL_DIR / "academic_decision_distribution.csv", index=False)
    academic_calibration.to_csv(ACADEMIC_EVAL_DIR / "academic_calibration_bins.csv", index=False)
    academic_score_bins.to_csv(ACADEMIC_EVAL_DIR / "academic_score_deciles.csv", index=False)

    academic_summary_json = {{
        "dataset": "{dataset}",
        "n_seeds": int(len(academic_seed_metrics)),
        "holdout_rows_per_seed": int(academic_seed_metrics["n_holdout"].iloc[0]),
        "fraud_rate_mean": float(academic_seed_metrics["fraud_rate"].mean()),
        "primary_tables": {{
            "seed_metrics": "academic_seed_metrics.csv",
            "metric_summary": "academic_metric_summary.csv",
            "topk_capture": "academic_topk_capture.csv",
            "policy_metrics": "academic_policy_metrics.csv",
            "decision_distribution": "academic_decision_distribution.csv",
            "calibration_bins": "academic_calibration_bins.csv",
            "score_deciles": "academic_score_deciles.csv",
        }},
        "note": "Holdout descriptive threshold metrics are for reporting analysis. The pipeline's operational threshold is selected from training/OOF predictions.",
    }}
    (ACADEMIC_EVAL_DIR / "academic_evaluation_summary.json").write_text(
        json.dumps(academic_summary_json, indent=2),
        encoding="utf-8",
    )

    latest_seed = max(frames_by_seed)
    latest = frames_by_seed[latest_seed]
    y_latest = latest["isFraud"].astype(int).to_numpy()
    score_latest = latest["fraud_score"].astype(float).to_numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(score_latest[y_latest == 0], bins=50, alpha=0.65, label="Legitimate", density=True)
    plt.hist(score_latest[y_latest == 1], bins=50, alpha=0.65, label="Fraud", density=True)
    plt.xlabel("Fraud score")
    plt.ylabel("Density")
    plt.title(f"{{DATASET.upper()}} score distribution by class (seed {{latest_seed}})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACADEMIC_EVAL_DIR / f"score_distribution_seed_{{latest_seed}}.png", dpi=160)
    plt.show()

    latest_topk = academic_topk[academic_topk["seed"] == latest_seed]
    plt.figure(figsize=(7, 4))
    plt.plot(latest_topk["top_percent"], latest_topk["recall"], marker="o", label="Fraud recall")
    plt.plot(latest_topk["top_percent"], latest_topk["precision"], marker="s", label="Precision")
    plt.xlabel("Top risk population reviewed (%)")
    plt.ylabel("Rate")
    plt.title(f"{{DATASET.upper()}} top-k fraud capture (seed {{latest_seed}})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACADEMIC_EVAL_DIR / f"topk_capture_seed_{{latest_seed}}.png", dpi=160)
    plt.show()

    latest_cal = academic_calibration[(academic_calibration["seed"] == latest_seed) & (academic_calibration["n"] > 0)]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(latest_cal["mean_score"], latest_cal["fraud_rate"], s=np.maximum(latest_cal["n"] / latest_cal["n"].max() * 160, 20))
    plt.xlabel("Mean predicted score")
    plt.ylabel("Observed fraud rate")
    plt.title(f"{{DATASET.upper()}} calibration bins (seed {{latest_seed}})")
    plt.tight_layout()
    plt.savefig(ACADEMIC_EVAL_DIR / f"calibration_seed_{{latest_seed}}.png", dpi=160)
    plt.show()

    print("Academic evaluation artifacts written to:", ACADEMIC_EVAL_DIR)
    print("Seed-level metrics:")
    display(academic_seed_metrics)
    print("Metric summary with approximate 95% CI:")
    display(academic_metric_summary)
    print("Top-k fraud capture:")
    display(academic_topk)
    print("HITL / blocking policy metrics:")
    display(academic_policy)
    print("Decision distribution:")
    display(academic_decisions)
    """


def report_code(dataset_label: str, dataset: str) -> str:
    return f"""
    from IPython.display import Markdown, display

    metrics = pd.DataFrame(metrics_summary["results"])
    mean_smoothed_auc = metrics["smoothed_auc"].mean()
    std_smoothed_auc = metrics["smoothed_auc"].std(ddof=1) if len(metrics) > 1 else 0.0
    mean_smoothed_f1 = metrics["smoothed_f1"].mean()
    std_smoothed_f1 = metrics["smoothed_f1"].std(ddof=1) if len(metrics) > 1 else 0.0
    mean_latency = metrics["latency_ms"].mean()
    academic_eval_dir = ARTIFACTS_DIR / "academic_evaluation"
    academic_metric_summary_path = academic_eval_dir / "academic_metric_summary.csv"
    academic_topk_path = academic_eval_dir / "academic_topk_capture.csv"
    academic_policy_path = academic_eval_dir / "academic_policy_metrics.csv"
    academic_decisions_path = academic_eval_dir / "academic_decision_distribution.csv"

    if academic_metric_summary_path.exists():
        academic_metric_summary = pd.read_csv(academic_metric_summary_path)
        topk_table = pd.read_csv(academic_topk_path) if academic_topk_path.exists() else pd.DataFrame()
        policy_table = pd.read_csv(academic_policy_path) if academic_policy_path.exists() else pd.DataFrame()
        decision_table = pd.read_csv(academic_decisions_path) if academic_decisions_path.exists() else pd.DataFrame()
        roc_row = academic_metric_summary[academic_metric_summary["metric"] == "roc_auc"].iloc[0]
        pr_row = academic_metric_summary[academic_metric_summary["metric"] == "pr_auc"].iloc[0]
        ece_row = academic_metric_summary[academic_metric_summary["metric"] == "ece_10bin"].iloc[0]
        brier_row = academic_metric_summary[academic_metric_summary["metric"] == "brier"].iloc[0]
        f1_row = academic_metric_summary[academic_metric_summary["metric"] == "best_f1_holdout_descriptive"].iloc[0]
        topk_excerpt = topk_table.to_markdown(index=False) if not topk_table.empty else "Not generated."
        policy_excerpt = policy_table.to_markdown(index=False) if not policy_table.empty else "Not generated."
        decision_excerpt = decision_table.to_markdown(index=False) if not decision_table.empty else "Not generated."
        advanced_results_md = f'''
    ## Extended Academic Evaluation

    - Holdout ROC-AUC from saved predictions: {{roc_row["mean"]:.4f}} +/- {{roc_row["std"]:.4f}}
    - Holdout PR-AUC from saved predictions: {{pr_row["mean"]:.4f}} +/- {{pr_row["std"]:.4f}}
    - Descriptive best-F1 on holdout scores: {{f1_row["mean"]:.4f}} +/- {{f1_row["std"]:.4f}}
    - Brier score: {{brier_row["mean"]:.4f}} +/- {{brier_row["std"]:.4f}}
    - 10-bin ECE: {{ece_row["mean"]:.4f}} +/- {{ece_row["std"]:.4f}}

    ### Top-k Fraud Capture

    {{topk_excerpt}}

    ### HITL / Blocking Policy Metrics

    {{policy_excerpt}}

    ### Decision Distribution

    {{decision_excerpt}}

    Generated academic evaluation tables and figures are stored under `academic_evaluation/`.
    '''
    else:
        advanced_results_md = "## Extended Academic Evaluation\\n\\nRun the academic evaluation cell before this report cell to generate extended tables and figures."

    report_md = f'''
    # Academic Experiment Report: {dataset_label}

    ## Research Objective

    This notebook evaluates one built method, **MVS-XAI**, for fraud detection on the {dataset_label} benchmark. The objective is to assess whether multi-view stacking with confidence gating and UID smoothing can provide strong holdout detection performance while preserving explainability, HITL routing, fairness checks, drift diagnostics, and latency evidence.

    ## Method

    The method is MVS-XAI: tabular, sequential, and behavioral feature views feed base learners; out-of-fold predictions train a calibrated meta-learner; confidence gating down-weights unreliable model outputs; UID post-processing smooths transaction-level risk; the final holdout predictions are audited through XAI, HITL routing, drift, fairness, and statistical comparison modules.

    ## Experimental Design

    - Dataset: {dataset_label}
    - Data source: direct signed Kaggle bundle downloaded with `aria2c`
    - Evaluation split: temporal holdout from the labeled training data
    - Validation protocol: time-aware OOF folds with an explicit gap
    - Threshold selection: training OOF predictions only
    - Number of seeds: {{PIPELINE_N_SEEDS}}
    - Number of folds: {{PIPELINE_N_SPLITS}}
    - Primary reportable metrics: ROC-AUC, PR-AUC, F1, McNemar p-value, HITL routing distribution, fairness/drift/latency diagnostics

    ## Main Results

    - Smoothed holdout ROC-AUC: {{mean_smoothed_auc:.4f}} +/- {{std_smoothed_auc:.4f}}
    - Smoothed holdout F1: {{mean_smoothed_f1:.4f}} +/- {{std_smoothed_f1:.4f}}
    - Median decision latency: {{mean_latency:.2f}} ms

    {{advanced_results_md}}

    ## Artifacts

    - Run config: `run_config.json`
    - Metrics summary: `{dataset}_metrics_summary.json`
    - Per-seed holdout predictions: `seed_*/{dataset}_holdout_predictions.csv`
    - Full training log: `{dataset}_academic_full_train.log`
    - Artifact inventory: `artifact_index.csv`
    - Extended academic tables and figures: `academic_evaluation/`

    ## Reporting Notes

    These results are valid for academic reporting only after the notebook has completed successfully on the real dataset. Descriptive holdout-threshold metrics are reported for analysis; the operational pipeline threshold is selected from training/OOF predictions to avoid holdout leakage. The signed dataset links are time-limited; a failed download should be resolved by refreshing the Kaggle signed URLs and rerunning the download cell.
    '''

    report_path = ARTIFACTS_DIR / "academic_report.md"
    report_path.write_text(textwrap.dedent(report_md).strip() + "\\n", encoding="utf-8")
    display(Markdown(report_path.read_text(encoding="utf-8")))
    print(f"Academic report saved to: {{report_path}}")
    """


def package_artifacts_code(dataset: str) -> str:
    return f"""
    archive_base = ARTIFACTS_DIR.parent / ARTIFACTS_DIR.name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=ARTIFACTS_DIR)
    print(f"Packaged artifacts: {{archive_path}}")

    try:
        from google.colab import files
        print("Colab detected. Uncomment the next line to download the artifact zip in the browser.")
        # files.download(archive_path)
    except Exception:
        pass
    """


def ieee_download_code() -> str:
    return f"""
    IEEE_BUNDLE_URL = {IEEE_BUNDLE_URL!r}
    FORCE_DOWNLOAD = False
    DELETE_ARCHIVE_AFTER_EXTRACT = False

    if (DATA_DIR / "train_transaction.csv").exists() and (DATA_DIR / "train_identity.csv").exists():
        print("IEEE-CIS CSVs already exist in data/. Skipping download and extraction.")
    else:
        ieee_zip = download_with_aria2(IEEE_BUNDLE_URL, DATA_DIR / "ieee-fraud-detection.zip", force=FORCE_DOWNLOAD)
        safe_extract_zip(ieee_zip, DATA_DIR)
        if DELETE_ARCHIVE_AFTER_EXTRACT:
            ieee_zip.unlink(missing_ok=True)

    transaction_path = find_file(DATA_DIR, "train_transaction.csv")
    identity_path = find_file(DATA_DIR, "train_identity.csv")
    if transaction_path is None or identity_path is None:
        raise FileNotFoundError("IEEE-CIS archive must contain train_transaction.csv and train_identity.csv")

    move_to_data_root(transaction_path, "train_transaction.csv")
    move_to_data_root(identity_path, "train_identity.csv")

    print("IEEE-CIS files ready:")
    for name in ["train_transaction.csv", "train_identity.csv"]:
        path = DATA_DIR / name
        print(f" - {{path}} ({{path.stat().st_size / 1024**2:.1f}} MB)")

    preview_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
    preview = pd.read_csv(DATA_DIR / "train_transaction.csv", usecols=preview_cols, nrows=5)
    display(preview)
    """


def paysim_download_code() -> str:
    return f"""
    PAYSIM_BUNDLE_URL = {PAYSIM_BUNDLE_URL!r}
    FORCE_DOWNLOAD = False
    DELETE_ARCHIVE_AFTER_EXTRACT = False
    PAYSIM_PREFERRED_NAMES = [
        "paysim.csv",
        "PS_20174392719_1491204439457_log.csv",
        "paysim_log.csv",
        "paysim dataset.csv",
    ]
    PAYSIM_REQUIRED_COLUMNS = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    if (DATA_DIR / "paysim.csv").exists():
        print("PaySim CSV already exists in data/. Skipping download and extraction.")
    else:
        paysim_zip = download_with_aria2(PAYSIM_BUNDLE_URL, DATA_DIR / "paysim.zip", force=FORCE_DOWNLOAD)
        safe_extract_zip(paysim_zip, DATA_DIR)
        if DELETE_ARCHIVE_AFTER_EXTRACT:
            paysim_zip.unlink(missing_ok=True)

    paysim_path = find_csv_with_columns(DATA_DIR, PAYSIM_REQUIRED_COLUMNS, PAYSIM_PREFERRED_NAMES)
    move_to_data_root(paysim_path, "paysim.csv")
    path = DATA_DIR / "paysim.csv"
    print(f"PaySim file ready: {{path}} ({{path.stat().st_size / 1024**2:.1f}} MB)")
    display(pd.read_csv(path, nrows=5))
    """


def build_notebook(dataset: str) -> nbformat.NotebookNode:
    if dataset == "ieee":
        title = "MVS-XAI IEEE-CIS Full Academic Experiment"
        dataset_label = "IEEE-CIS Fraud Detection"
        download_code = ieee_download_code()
    elif dataset == "paysim":
        title = "MVS-XAI PaySim Full Academic Experiment"
        dataset_label = "PaySim"
        download_code = paysim_download_code()
    else:
        raise ValueError(dataset)

    intro_md = f"""
    # {title}

    This notebook is an end-to-end academic research artifact.

    - It clones the working repository revision directly from GitHub when the repo files are not already present.
    - It downloads the real {dataset_label} dataset using the signed direct URL and `aria2c`.
    - It does not mount or depend on Google Drive.
    - It trains one built method: **MVS-XAI**.
    - It runs the repository pipeline end to end inside the notebook kernel.
    - It writes and displays the artifacts required for a scientific report.
    """

    protocol_md = f"""
    ## Research Protocol

    | Component | Specification |
    |---|---|
    | Research object | Fraud detection with explainable multi-view stacking |
    | Dataset | {dataset_label} |
    | Data collection | Secondary benchmark data downloaded from Kaggle signed bundle |
    | Method | MVS-XAI only |
    | Validation | Temporal holdout and time-aware OOF folds |
    | Leakage control | Feature fitting and threshold selection use training/OOF partitions |
    | Metrics | ROC-AUC, PR-AUC, F1, McNemar, latency, HITL distribution, fairness and drift diagnostics |
    | Artifacts | Config JSON, metrics JSON, per-seed holdout predictions, log, report markdown, artifact index |
    """

    cells = [
        markdown_cell(f"{dataset}-intro", intro_md),
        markdown_cell(f"{dataset}-research-protocol", protocol_md),
        code_cell(f"{dataset}-repo-config", repo_config_code(dataset)),
        code_cell(f"{dataset}-environment", environment_code()),
        code_cell(f"{dataset}-download-helpers", download_helpers_code()),
        code_cell(f"{dataset}-dataset-download", download_code),
        code_cell(f"{dataset}-run-pipeline", run_pipeline_code(dataset)),
        code_cell(f"{dataset}-artifact-review", artifact_review_code(dataset)),
        code_cell(f"{dataset}-academic-evaluation", academic_evaluation_code(dataset)),
        code_cell(f"{dataset}-academic-report", report_code(dataset_label, dataset)),
        code_cell(f"{dataset}-package-artifacts", package_artifacts_code(dataset)),
    ]
    return new_notebook(cells=cells, metadata=notebook_metadata())


def write_ieee_notebook() -> Path:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook("ieee"), IEEE_NOTEBOOK_PATH)
    return IEEE_NOTEBOOK_PATH


def write_paysim_notebook() -> Path:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook("paysim"), PAYSIM_NOTEBOOK_PATH)
    return PAYSIM_NOTEBOOK_PATH


def main() -> None:
    ieee_path = write_ieee_notebook()
    paysim_path = write_paysim_notebook()
    print(f"Notebook written: {ieee_path}")
    print(f"Notebook written: {paysim_path}")


if __name__ == "__main__":
    main()
