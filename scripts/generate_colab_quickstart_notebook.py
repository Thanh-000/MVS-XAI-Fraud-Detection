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

    - mounting Google Drive
    - cloning the repository directly from GitHub
    - installing dependencies
    - reading IEEE-CIS directly from the mounted Google Drive path
    - reading PaySim directly from the mounted Google Drive path
    - running the full train pipeline for IEEE-CIS and PaySim
    - regenerating the reviewer notebooks

    Practical note:

    - This notebook is intentionally not committed with outputs because several cells require interactive Colab actions such as Drive mount.
    """

    mount_code = """
    from google.colab import drive
    drive.mount('/content/drive')
    """

    repo_config_md = """
    ## 1. Configure Repo Paths

    This quickstart assumes:

    - the repo is cloned directly from GitHub
    - the datasets are already stored somewhere under `MyDrive`

    Set the GitHub URL and branch if needed, then run the clone cell.
    """

    repo_config_code = """
    GIT_REPO_URL = "https://github.com/Thanh-000/MVS-XAI-Fraud-Detection"
    GIT_BRANCH = ""
    RUNTIME_REPO_PATH = "/content/MVS_XAI"
    """

    repo_clone_md = """
    ### 1A. Clone Repo From GitHub

    Use this path when the repository is hosted remotely and you want a fresh runtime copy.
    """

    repo_clone_shell = """
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

    The extra installs cover optional tree-model and XAI packages that some repo snapshots import at module load time.
    """

    install_code = """
    !python -m pip install --upgrade pip
    !pip install -r requirements.txt
    !pip install xgboost lightgbm catboost imbalanced-learn
    !pip install shap lime dice-ml alibi google-generativeai anchor-exp
    """

    drive_data_md = """
    ## 3. Dataset Paths On Mounted Drive

    This notebook reads the datasets directly from `MyDrive` after mount.

    Requirements:

    - `IEEE_DRIVE_DATA_DIR` must contain `train_transaction.csv` and `train_identity.csv`
    - `PAYSIM_DRIVE_DATA_DIR` must contain one of:
      - `paysim.csv`
      - `PS_20174392719_1491204439457_log.csv`
      - `paysim_log.csv`
    """

    drive_data_code = """
    IEEE_DRIVE_DATA_DIR = "/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection"
    PAYSIM_DRIVE_DATA_DIR = "/content/drive/MyDrive/MVS_XAI_Data/paysim"
    """

    ieee_validate_code = """
    from pathlib import Path

    ieee_dir = Path(IEEE_DRIVE_DATA_DIR)

    if not (ieee_dir / "train_transaction.csv").is_file():
        raise FileNotFoundError(f"Missing train_transaction.csv in {ieee_dir}")
    if not (ieee_dir / "train_identity.csv").is_file():
        raise FileNotFoundError(f"Missing train_identity.csv in {ieee_dir}")

    print("IEEE path:", ieee_dir)
    print("IEEE files OK")
    """

    paysim_validate_code = """
    from pathlib import Path

    paysim_dir = Path(PAYSIM_DRIVE_DATA_DIR)

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
            IEEE_DRIVE_DATA_DIR,
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
            PAYSIM_DRIVE_DATA_DIR,
            "--device",
            "cuda",
        ]
    )
    """

    regenerate_md = """
    ## 6. Regenerate Reviewer Notebooks

    These commands rebuild the two reviewer-facing notebook artifacts using the currently configured Drive dataset paths.
    """

    regenerate_code = """
    import os
    import subprocess

    env = os.environ.copy()
    env["MVS_XAI_IEEE_DATA_DIR"] = IEEE_DRIVE_DATA_DIR
    env["MVS_XAI_PAYSIM_DATA_DIR"] = PAYSIM_DRIVE_DATA_DIR

    subprocess.run(["python", "scripts/generate_submission_notebook.py"], check=True, env=env)
    subprocess.run(["python", "scripts/generate_paysim_submission_notebook.py"], check=True, env=env)
    print(sorted(os.listdir("notebooks")))
    """

    save_back_md = """
    ## 7. Copy Results Back To Drive

    This copies the updated repo, notebooks, and artifacts back into Drive.
    """

    save_back_code = """
    !rsync -a --delete /content/MVS_XAI/ /content/drive/MyDrive/MVS_XAI/
    """

    troubleshooting_md = """
    ## Troubleshooting

    - If dataset setup fails, check `IEEE_DRIVE_DATA_DIR` and `PAYSIM_DRIVE_DATA_DIR`.
    - If tree packages fail to install, rerun the install cell once.
    - If CUDA is unavailable, change `--device cuda` to `--device cpu`.
    - Holdout predictions are saved under `artifacts/`.
    """

    notebook = new_notebook(
        cells=[
            new_markdown_cell(textwrap.dedent(intro_md).strip()),
            new_code_cell(textwrap.dedent(mount_code).strip()),
            new_markdown_cell(textwrap.dedent(repo_config_md).strip()),
            new_code_cell(textwrap.dedent(repo_config_code).strip()),
            new_markdown_cell(textwrap.dedent(repo_clone_md).strip()),
            new_code_cell(textwrap.dedent(repo_clone_shell).strip()),
            new_markdown_cell(textwrap.dedent(install_md).strip()),
            new_code_cell(textwrap.dedent(install_code).strip()),
            new_markdown_cell(textwrap.dedent(drive_data_md).strip()),
            new_code_cell(textwrap.dedent(drive_data_code).strip()),
            new_code_cell(textwrap.dedent(ieee_validate_code).strip()),
            new_code_cell(textwrap.dedent(paysim_validate_code).strip()),
            new_code_cell(textwrap.dedent(runner_code).strip()),
            new_markdown_cell(textwrap.dedent(run_md).strip()),
            new_code_cell(textwrap.dedent(run_ieee_code).strip()),
            new_code_cell(textwrap.dedent(run_paysim_code).strip()),
            new_markdown_cell(textwrap.dedent(regenerate_md).strip()),
            new_code_cell(textwrap.dedent(regenerate_code).strip()),
            new_markdown_cell(textwrap.dedent(save_back_md).strip()),
            new_code_cell(textwrap.dedent(save_back_code).strip()),
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
