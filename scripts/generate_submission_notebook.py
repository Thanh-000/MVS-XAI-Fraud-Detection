from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_full_experiment_notebooks import write_ieee_notebook


def main() -> None:
    path = write_ieee_notebook()
    print(f"Notebook written: {path}")
    print("Mode: full IEEE-CIS experiment, direct aria2 download, no Drive mount")


if __name__ == "__main__":
    main()
