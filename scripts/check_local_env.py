from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path


MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "torch",
    "xgboost",
    "lightgbm",
    "catboost",
    "imblearn",
    "river",
    "shap",
    "lime",
    "dice_ml",
    "matplotlib",
    "seaborn",
    "nbformat",
    "yaml",
    "tabulate",
]


def module_version(module):
    return getattr(module, "__version__", "installed")


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Executable:", sys.executable)
    print("Working directory:", Path.cwd())

    missing = []
    print("\nDependency check:")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            print(f"  OK   {name:<12} {module_version(module)}")
        except Exception as exc:
            missing.append(name)
            print(f"  MISS {name:<12} {exc}")

    print("\nExternal tools:")
    for tool in ["git", "aria2c", "unzip"]:
        path = shutil.which(tool)
        print(f"  {tool:<8} {path or 'not found'}")

    try:
        import torch

        print("\nPyTorch:")
        print("  version:", torch.__version__)
        print("  cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  cuda_version:", torch.version.cuda)
            print("  device_count:", torch.cuda.device_count())
            print("  device_0:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("\nPyTorch check failed:", exc)

    data_dir = Path("data")
    print("\nData presence:")
    ieee_ok = (data_dir / "train_transaction.csv").exists() and (data_dir / "train_identity.csv").exists()
    paysim_candidates = [
        "paysim.csv",
        "PS_20174392719_1491204439457_log.csv",
        "paysim_log.csv",
        "paysim dataset.csv",
    ]
    paysim_ok = any((data_dir / name).exists() for name in paysim_candidates)
    print("  IEEE-CIS:", "ready" if ieee_ok else "missing train_transaction.csv/train_identity.csv")
    print("  PaySim:", "ready" if paysim_ok else "missing PaySim CSV")

    if missing:
        print("\nMissing modules:", ", ".join(missing))
        print("Install with: python -m pip install -r requirements.txt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

