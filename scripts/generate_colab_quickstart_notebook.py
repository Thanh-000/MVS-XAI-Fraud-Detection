from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "00_Colab_Quickstart.ipynb"


def build_notebook():
    intro_md = """
    # MVS-XAI Colab Quickstart

    This notebook is a setup and execution helper for Google Colab.

    It covers:

    - cloning the repository directly from GitHub
    - installing dependencies
    - downloading IEEE-CIS and PaySim from pasted links
    - accepting direct HTTPS links or Google Drive share links
    - running stable or full research training for IEEE-CIS and PaySim
    - regenerating the reviewer notebooks

    Practical note:

    - This notebook is intentionally not committed with outputs because several cells require user-provided download links and a live Colab runtime.
    """

    repo_config_md = """
    ## 1. Configure Repo Paths

    This quickstart assumes:

    - the repo is cloned directly from GitHub
    - datasets are downloaded into the runtime from pasted links

    Set the GitHub URL and branch if needed, then run the clone cell.
    """

    repo_config_code = """
    GIT_REPO_URL = "https://github.com/Thanh-000/MVS-XAI-Fraud-Detection"
    GIT_BRANCH = ""
    RUNTIME_REPO_PATH = "/content/MVS_XAI"
    """

    repo_clone_md = """
    ### 1A. Clone Repo From GitHub
    """

    repo_clone_code = """
    import os
    import shutil
    import subprocess

    shutil.rmtree(RUNTIME_REPO_PATH, ignore_errors=True)
    subprocess.run(["git", "clone", GIT_REPO_URL, RUNTIME_REPO_PATH], check=True)
    os.chdir(RUNTIME_REPO_PATH)

    if GIT_BRANCH.strip():
        subprocess.run(["git", "checkout", GIT_BRANCH], check=True)

    subprocess.run(["git", "status", "--short"], check=True)
    """

    install_md = """
    ## 2. Install Dependencies

    The extra installs cover optional tree-model and XAI packages used by the repo, plus:

    - `aria2` for faster direct-link downloads
    - `gdown` for Google Drive share links
    - best-effort `cupy` installation so XGBoost GPU prediction can stay on-device when the Colab CUDA wheel is available
    """

    install_code = """
    !apt-get -qq update
    !apt-get -qq install -y aria2
    !python -m pip install --upgrade pip
    !pip install -r requirements.txt
    !pip install xgboost lightgbm catboost imbalanced-learn
    !pip install shap lime dice-ml alibi google-generativeai "anchor-exp>=0.0.2.0" gdown

    import subprocess
    import sys

    def install_best_effort_cupy():
        for wheel in ["cupy-cuda12x", "cupy-cuda11x"]:
            print(f"Trying optional CuPy wheel: {wheel}")
            result = subprocess.run([sys.executable, "-m", "pip", "install", wheel], check=False)
            if result.returncode == 0:
                print(f"Installed optional CuPy package: {wheel}")
                return wheel
        print("CuPy wheel not installed; pipeline will keep the CPU fallback path for XGBoost prediction.")
        return None

    OPTIONAL_CUPY_WHEEL = install_best_effort_cupy()
    """

    dataset_links_md = """
    ## 3. Dataset Download Links

    Paste your links below.

    Supported formats:

    - direct HTTPS download links
    - Google Drive share links copied from Chrome

    IEEE-CIS uses one mode only:

    - one bundle zip link via `IEEE_BUNDLE_URL`

    Preferred for Kaggle signed URLs:

    - `IEEE_BUNDLE_URL`

    PaySim supports two modes:

    - one bundle zip link via `PAYSIM_BUNDLE_URL`
    - or one direct CSV link via `PAYSIM_URL`

    Preferred for Kaggle signed URLs:

    - `PAYSIM_BUNDLE_URL`
    """

    dataset_links_code = """
    IEEE_BUNDLE_URL = ""
    PAYSIM_BUNDLE_URL = ""
    PAYSIM_URL = ""
    """

    download_helper_code = """
    import os
    import shutil
    import subprocess
    import zipfile
    from pathlib import Path
    from urllib.parse import urlparse

    import gdown

    DATA_DIR = Path(RUNTIME_REPO_PATH) / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PAYSIM_CSV_CANDIDATES = [
        "paysim.csv",
        "PS_20174392719_1491204439457_log.csv",
        "paysim_log.csv",
    ]

    def download_from_link(url, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not str(url).strip():
            raise ValueError(f"Missing URL for {output_path.name}")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"URL must start with http/https: {url}")

        if "drive.google.com" in parsed.netloc:
            gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        else:
            subprocess.run(
                [
                    "aria2c",
                    "--dir",
                    str(output_path.parent),
                    "--out",
                    output_path.name,
                    "--max-connection-per-server=16",
                    "--split=16",
                    "--min-split-size=1M",
                    "--file-allocation=none",
                    url,
                ],
                check=True,
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Download failed for {output_path}")

        print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    def extract_zip(zip_path, extract_dir):
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)
        if not zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"Extracted zip into: {extract_dir}")

    def find_first_existing(paths):
        return next((path for path in paths if path.exists()), None)

    def resolve_paysim_csv(search_root):
        search_root = Path(search_root)

        direct_candidates = [search_root / name for name in PAYSIM_CSV_CANDIDATES]
        direct_match = find_first_existing(direct_candidates)
        if direct_match is not None:
            return direct_match

        recursive_matches = []
        for name in PAYSIM_CSV_CANDIDATES:
            recursive_matches.extend(search_root.rglob(name))

        if recursive_matches:
            recursive_matches = sorted(set(recursive_matches), key=lambda p: (len(p.parts), str(p)))
            return recursive_matches[0]

        return None

    def normalize_paysim_csv(search_root, target_path):
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        match = resolve_paysim_csv(search_root)
        if match is None:
            raise FileNotFoundError(
                f"No supported PaySim CSV found under {Path(search_root)}. "
                f"Expected one of: {', '.join(PAYSIM_CSV_CANDIDATES)}"
            )

        if match.resolve() != target_path.resolve():
            shutil.copy2(match, target_path)
            print(f"Normalized PaySim CSV: {match} -> {target_path}")
        else:
            print(f"PaySim CSV already normalized at: {target_path}")

        return target_path

    print(f"Runtime data directory: {DATA_DIR}")
    """

    ieee_download_md = """
    ### 3A. Download IEEE-CIS

    The quickstart only supports the Kaggle bundle path for IEEE-CIS.
    After extraction, the pipeline will use `train_transaction.csv` and
    `train_identity.csv` from the bundle.
    """

    ieee_download_code = """
    if not str(IEEE_BUNDLE_URL).strip():
        raise ValueError("Set IEEE_BUNDLE_URL to the Kaggle IEEE-CIS bundle link before running this cell.")

    ieee_zip_path = DATA_DIR / "ieee-fraud-detection.zip"
    download_from_link(IEEE_BUNDLE_URL, ieee_zip_path)
    extract_zip(ieee_zip_path, DATA_DIR)
    print(sorted(os.listdir(DATA_DIR)))
    """

    paysim_download_md = """
    ### 3B. Download PaySim

    The downloaded file is normalized to `data/paysim.csv` for the training pipeline.

    For Kaggle bundle links, the notebook downloads the zip as `archive (1).zip`,
    extracts it, then searches recursively for the real PaySim CSV and normalizes it.
    """

    paysim_download_code = """
    if str(PAYSIM_BUNDLE_URL).strip():
        paysim_zip_path = DATA_DIR / "archive (1).zip"
        download_from_link(PAYSIM_BUNDLE_URL, paysim_zip_path)
        extract_zip(paysim_zip_path, DATA_DIR)
        normalize_paysim_csv(DATA_DIR, DATA_DIR / "paysim.csv")
    else:
        download_from_link(PAYSIM_URL, DATA_DIR / "paysim.csv")
    print(sorted(os.listdir(DATA_DIR)))
    """

    dataset_validate_md = """
    ## 4. Validate Downloaded Files
    """

    ieee_validate_code = """
    from pathlib import Path

    ieee_dir = Path(DATA_DIR)

    if not (ieee_dir / "train_transaction.csv").is_file():
        raise FileNotFoundError(f"Missing train_transaction.csv in {ieee_dir}")
    if not (ieee_dir / "train_identity.csv").is_file():
        raise FileNotFoundError(f"Missing train_identity.csv in {ieee_dir}")

    print("IEEE path:", ieee_dir)
    print("IEEE files OK")
    """

    paysim_validate_code = """
    from pathlib import Path

    paysim_dir = Path(DATA_DIR)

    paysim_candidates = [
        paysim_dir / "paysim.csv",
        paysim_dir / "PS_20174392719_1491204439457_log.csv",
        paysim_dir / "paysim_log.csv",
    ]
    paysim_match = next((path for path in paysim_candidates if path.is_file()), None)
    if paysim_match is None:
        raise FileNotFoundError(f"No supported PaySim CSV found in {paysim_dir}")

    print("PaySim path:", paysim_dir)
    print("PaySim file:", paysim_match.name)
    """

    train_config_md = """
    ## 5. Configure Training Mode

    Choose one preset before launching a run:

    - `tree`: most stable Colab-free path. Uses available tree models and disables MLP/LSTM.
    - `auto`: uses every installed model family. On Colab with PyTorch installed this usually enables the full MVS stack.
    - `full_mvs`: strict research mode. Fails early if PyTorch is unavailable.

    Recommended workflow:

    - first run `tree` to validate data paths and the end-to-end artifact path
    - then switch to `auto` or `full_mvs` for the final research run if the runtime has enough memory/time
    - keep `CTGAN_SAMPLES = 0` unless you have a long paid runtime; CTGAN is expensive on PaySim/IEEE-CIS
    """

    train_config_code = """
    from pathlib import Path

    TRAIN_PRESET = "tree"      # "tree", "auto", or "full_mvs"
    DEVICE = "cuda"            # Colab GPU runtime; pipeline falls back to CPU when CUDA is unavailable
    N_SPLITS = 5
    GAP_SIZE = 1000
    N_SEEDS = 1
    SMOTE_STRATEGY = 0.30
    CTGAN_SAMPLES = 0
    CTGAN_EPOCHS = 30

    # PaySim loader controls. Use None for full data; set a smaller integer for a faster trial run.
    PAYSIM_CHUNK_SIZE = 750000
    PAYSIM_MAX_ROWS = None
    PAYSIM_STEP_BLOCK_SIZE = 24

    ARTIFACTS_DIR = Path(RUNTIME_REPO_PATH) / "artifacts"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training preset:", TRAIN_PRESET)
    print("Artifacts directory:", ARTIFACTS_DIR)
    """

    runner_code = """
    import os
    import re
    import subprocess
    from pathlib import Path

    def run_and_stream(cmd, env=None, log_path=None):
        log_file = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                if log_file is not None:
                    log_file.write(line)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")
        finally:
            if log_file is not None:
                log_file.close()

    print("Streaming helper ready.")
    """

    run_md = """
    ## 6. Run Full Training Pipeline

    Use GPU runtime for the heaviest run if available:

    - `Runtime` -> `Change runtime type` -> `T4 GPU`

    The command cells below use the training controls from the previous section.
    For final academic runs, keep a copy of the log and the generated holdout CSV.
    """

    run_ieee_code = """
    run_and_stream(
        [
            "python",
            "main_train_pipeline.py",
            "--dataset",
            "ieee",
            "--data_dir",
            str(DATA_DIR),
            "--device",
            DEVICE,
            "--preset",
            TRAIN_PRESET,
            "--n_splits",
            str(N_SPLITS),
            "--gap_size",
            str(GAP_SIZE),
            "--n_seeds",
            str(N_SEEDS),
            "--smote_strategy",
            str(SMOTE_STRATEGY),
            "--ctgan_samples",
            str(CTGAN_SAMPLES),
            "--ctgan_epochs",
            str(CTGAN_EPOCHS),
            "--artifacts_dir",
            str(ARTIFACTS_DIR),
        ],
        log_path=ARTIFACTS_DIR / "ieee_train.log",
    )
    """

    run_paysim_code = """
    run_and_stream(
        [
            "python",
            "main_train_pipeline.py",
            "--dataset",
            "paysim",
            "--data_dir",
            str(DATA_DIR),
            "--device",
            DEVICE,
            "--preset",
            TRAIN_PRESET,
            "--n_splits",
            str(N_SPLITS),
            "--gap_size",
            str(GAP_SIZE),
            "--n_seeds",
            str(N_SEEDS),
            "--smote_strategy",
            str(SMOTE_STRATEGY),
            "--ctgan_samples",
            str(CTGAN_SAMPLES),
            "--ctgan_epochs",
            str(CTGAN_EPOCHS),
            "--paysim_chunk_size",
            str(PAYSIM_CHUNK_SIZE),
            "--paysim_step_block_size",
            str(PAYSIM_STEP_BLOCK_SIZE),
            "--artifacts_dir",
            str(ARTIFACTS_DIR),
        ]
        + ([] if PAYSIM_MAX_ROWS is None else ["--paysim_max_rows", str(PAYSIM_MAX_ROWS)]),
        log_path=ARTIFACTS_DIR / "paysim_train.log",
    )
    """

    smoke_md = """
    ## 7. Optional Pipeline Smoke Test

    This cell creates a tiny PaySim-like CSV under `/content/mvs_xai_smoke_data`
    and runs the production pipeline with `--preset tree`.

    Use it only to confirm that the Colab runtime and current repository revision can train end-to-end.
    It is not an academic result.
    """

    smoke_code = """
    import numpy as np
    import pandas as pd

    SMOKE_DATA_DIR = Path("/content/mvs_xai_smoke_data")
    SMOKE_ARTIFACTS_DIR = Path(RUNTIME_REPO_PATH) / "artifacts_smoke"
    SMOKE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = 240
    step = np.arange(n) // 4 + 1
    types = np.array(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    type_col = types[np.arange(n) % len(types)]
    amount = rng.lognormal(mean=4.8, sigma=0.55, size=n)
    old_org = rng.uniform(100, 5000, size=n)
    fraud = ((np.arange(n) % 9 == 0) | ((type_col == "TRANSFER") & (amount > np.quantile(amount, 0.82)))).astype(int)
    new_org = np.maximum(old_org - amount * rng.uniform(0.1, 0.9, size=n), 0)
    old_dest = rng.uniform(100, 10000, size=n)
    new_dest = old_dest + amount * rng.uniform(0.05, 1.0, size=n)

    pd.DataFrame(
        {
            "step": step,
            "type": type_col,
            "amount": amount,
            "nameOrig": [f"C{i % 25:04d}" for i in range(n)],
            "nameDest": [f"M{(i * 3) % 35:04d}" for i in range(n)],
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": fraud,
        }
    ).to_csv(SMOKE_DATA_DIR / "paysim.csv", index=False)

    run_and_stream(
        [
            "python",
            "main_train_pipeline.py",
            "--dataset",
            "paysim",
            "--data_dir",
            str(SMOKE_DATA_DIR),
            "--device",
            "cpu",
            "--preset",
            "tree",
            "--n_splits",
            "2",
            "--gap_size",
            "0",
            "--test_ratio",
            "0.2",
            "--smote_strategy",
            "0",
            "--ctgan_samples",
            "0",
            "--paysim_chunk_size",
            "0",
            "--artifacts_dir",
            str(SMOKE_ARTIFACTS_DIR),
        ],
        log_path=SMOKE_ARTIFACTS_DIR / "paysim_smoke_train.log",
    )
    """

    results_md = """
    ## 8. Review Results

    These cells summarize the latest run directly from saved artifacts and logs.
    Run the dataset-specific cell after the corresponding training cell completes.
    """

    results_helper_code = """
    import re
    import pandas as pd
    from pathlib import Path

    def _last_match(pattern, text):
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        return matches[-1] if matches else None

    def show_run_results(dataset):
        dataset = dataset.lower().strip()
        pred_path = ARTIFACTS_DIR / f"{dataset}_holdout_predictions.csv"
        log_path = ARTIFACTS_DIR / f"{dataset}_train.log"

        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction artifact: {pred_path}")

        df_pred = pd.read_csv(pred_path)
        print(f"Prediction artifact: {pred_path}")
        print(f"Rows: {len(df_pred):,}")
        print(f"Columns: {list(df_pred.columns)}")

        if "decision" in df_pred.columns:
            print("\\nDecision counts:")
            print(df_pred["decision"].value_counts(dropna=False).to_string())

        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            metric_patterns = {
                "Optimal threshold": r"Optimal F1 Threshold: ([0-9.]+)",
                "ROC-AUC": r"ROC-AUC:\\s+([0-9.]+)",
                "PR-AUC": r"PR-AUC:\\s+([0-9.]+)",
                "F1-Score": r"F1-Score:\\s+([0-9.]+)",
                "Score PSI": r"Score PSI: ([0-9.]+)",
                "TPR Disparity": r"TPR Disparity: ([0-9.]+%)",
                "Latency": r"Median single-transaction decision latency: ([0-9.]+ ms)",
            }

            print("\\nParsed log summary:")
            for label, pattern in metric_patterns.items():
                value = _last_match(pattern, log_text)
                if value is not None:
                    print(f"- {label}: {value}")
        else:
            print(f"No log file found at {log_path}")

        print("\\nTop rows:")
        display(df_pred.head(10))
    """

    ieee_results_code = """
    show_run_results("ieee")
    """

    paysim_results_code = """
    show_run_results("paysim")
    """

    regenerate_md = """
    ## 9. Regenerate Reviewer Notebooks

    These commands rebuild the two reviewer-facing notebook artifacts using the downloaded files under `data/`.
    """

    regenerate_code = """
    import os
    import subprocess

    env = os.environ.copy()
    env["MVS_XAI_IEEE_DATA_DIR"] = str(DATA_DIR)
    env["MVS_XAI_PAYSIM_DATA_DIR"] = str(DATA_DIR)

    subprocess.run(["python", "scripts/generate_submission_notebook.py"], check=True, env=env)
    subprocess.run(["python", "scripts/generate_paysim_submission_notebook.py"], check=True, env=env)
    print(sorted(os.listdir("notebooks")))
    """

    export_md = """
    ## 10. Export Results

    This bundles notebooks and artifacts into a zip and downloads it to your browser.
    """

    export_code = """
    import shutil
    from google.colab import files

    bundle_root = Path(RUNTIME_REPO_PATH) / "colab_export"
    shutil.rmtree(bundle_root, ignore_errors=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    for folder_name in ["notebooks", "artifacts"]:
        source = Path(RUNTIME_REPO_PATH) / folder_name
        if source.exists():
            shutil.copytree(source, bundle_root / folder_name)

    archive_path = shutil.make_archive(str(bundle_root), "zip", root_dir=bundle_root)
    files.download(archive_path)
    """

    troubleshooting_md = """
    ## Troubleshooting

    - If a download fails, check whether the pasted URL is a real downloadable link and not an expired browser session link.
    - For Google Drive links, make sure sharing permissions allow download access.
    - IEEE-CIS Kaggle `test` is not used by the research pipeline because it does not include `isFraud` labels. The repo evaluates on temporal holdouts carved from the labeled `train` split instead.
    - If CUDA is unavailable, change `--device cuda` to `--device cpu`.
    - If the optional CuPy install fails, the notebook can still run; only the XGBoost GPU prediction warning mitigation will be unavailable.
    - If full IEEE-CIS kills the Colab runtime, set `TRAIN_PRESET = "tree"` and reduce `N_SPLITS` before rerunning.
    - If you need the strict full research stack, set `TRAIN_PRESET = "full_mvs"` after confirming PyTorch is installed.
    - If a run fails late while writing outputs, set `ARTIFACTS_DIR` to a fresh folder so smoke artifacts and final artifacts remain separate.
    """

    notebook = new_notebook(
        cells=[
            new_markdown_cell(textwrap.dedent(intro_md).strip()),
            new_markdown_cell(textwrap.dedent(repo_config_md).strip()),
            new_code_cell(textwrap.dedent(repo_config_code).strip()),
            new_markdown_cell(textwrap.dedent(repo_clone_md).strip()),
            new_code_cell(textwrap.dedent(repo_clone_code).strip()),
            new_markdown_cell(textwrap.dedent(install_md).strip()),
            new_code_cell(textwrap.dedent(install_code).strip()),
            new_markdown_cell(textwrap.dedent(dataset_links_md).strip()),
            new_code_cell(textwrap.dedent(dataset_links_code).strip()),
            new_code_cell(textwrap.dedent(download_helper_code).strip()),
            new_markdown_cell(textwrap.dedent(ieee_download_md).strip()),
            new_code_cell(textwrap.dedent(ieee_download_code).strip()),
            new_markdown_cell(textwrap.dedent(paysim_download_md).strip()),
            new_code_cell(textwrap.dedent(paysim_download_code).strip()),
            new_markdown_cell(textwrap.dedent(dataset_validate_md).strip()),
            new_code_cell(textwrap.dedent(ieee_validate_code).strip()),
            new_code_cell(textwrap.dedent(paysim_validate_code).strip()),
            new_markdown_cell(textwrap.dedent(train_config_md).strip()),
            new_code_cell(textwrap.dedent(train_config_code).strip()),
            new_code_cell(textwrap.dedent(runner_code).strip()),
            new_markdown_cell(textwrap.dedent(run_md).strip()),
            new_code_cell(textwrap.dedent(run_ieee_code).strip()),
            new_code_cell(textwrap.dedent(run_paysim_code).strip()),
            new_markdown_cell(textwrap.dedent(smoke_md).strip()),
            new_code_cell(textwrap.dedent(smoke_code).strip()),
            new_markdown_cell(textwrap.dedent(results_md).strip()),
            new_code_cell(textwrap.dedent(results_helper_code).strip()),
            new_code_cell(textwrap.dedent(ieee_results_code).strip()),
            new_code_cell(textwrap.dedent(paysim_results_code).strip()),
            new_markdown_cell(textwrap.dedent(regenerate_md).strip()),
            new_code_cell(textwrap.dedent(regenerate_code).strip()),
            new_markdown_cell(textwrap.dedent(export_md).strip()),
            new_code_cell(textwrap.dedent(export_code).strip()),
            new_markdown_cell(textwrap.dedent(troubleshooting_md).strip()),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
    )
    return notebook


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook written: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
