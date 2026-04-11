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
    - running the full train pipeline for IEEE-CIS and PaySim
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

    The extra installs cover optional tree-model and XAI packages used by the repo, plus `gdown` so Google Drive share links can be downloaded directly.
    """

    install_code = """
    !python -m pip install --upgrade pip
    !pip install -r requirements.txt
    !pip install xgboost lightgbm catboost imbalanced-learn
    !pip install shap lime dice-ml alibi google-generativeai "anchor-exp>=0.0.2.0" gdown
    """

    dataset_links_md = """
    ## 3. Dataset Download Links

    Paste your links below.

    Supported formats:

    - direct HTTPS download links
    - Google Drive share links copied from Chrome

    IEEE-CIS requires two links:

    - `IEEE_TRANSACTION_URL`
    - `IEEE_IDENTITY_URL`

    PaySim requires one link:

    - `PAYSIM_URL`
    """

    dataset_links_code = """
    IEEE_TRANSACTION_URL = ""
    IEEE_IDENTITY_URL = ""
    PAYSIM_URL = ""
    """

    download_helper_code = """
    import os
    from pathlib import Path
    from urllib.parse import urlparse
    from urllib.request import urlretrieve

    import gdown

    DATA_DIR = Path(RUNTIME_REPO_PATH) / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
            urlretrieve(url, str(output_path))

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Download failed for {output_path}")

        print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    print(f"Runtime data directory: {DATA_DIR}")
    """

    ieee_download_md = """
    ### 3A. Download IEEE-CIS
    """

    ieee_download_code = """
    download_from_link(IEEE_TRANSACTION_URL, DATA_DIR / "train_transaction.csv")
    download_from_link(IEEE_IDENTITY_URL, DATA_DIR / "train_identity.csv")
    print(sorted(os.listdir(DATA_DIR)))
    """

    paysim_download_md = """
    ### 3B. Download PaySim

    The downloaded file is normalized to `data/paysim.csv` for the training pipeline.
    """

    paysim_download_code = """
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

    runner_code = """
    import os
    import subprocess

    def run_and_stream(cmd, env=None):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")

    print("Streaming helper ready.")
    """

    run_md = """
    ## 5. Run Full Training Pipeline

    Use GPU runtime for the heaviest run if available:

    - `Runtime` -> `Change runtime type` -> `T4 GPU`
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
            "cuda",
        ]
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
            "cuda",
        ]
    )
    """

    regenerate_md = """
    ## 6. Regenerate Reviewer Notebooks

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
    ## 7. Export Results

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
    - If CUDA is unavailable, change `--device cuda` to `--device cpu`.
    - If full IEEE-CIS kills the Colab runtime, reduce the model set or split count before rerunning.
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
            new_code_cell(textwrap.dedent(runner_code).strip()),
            new_markdown_cell(textwrap.dedent(run_md).strip()),
            new_code_cell(textwrap.dedent(run_ieee_code).strip()),
            new_code_cell(textwrap.dedent(run_paysim_code).strip()),
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
