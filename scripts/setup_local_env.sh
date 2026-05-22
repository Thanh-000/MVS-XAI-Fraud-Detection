#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=".venv"
CUDA_TORCH=0
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --cuda-torch)
      CUDA_TORCH=1
      shift
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

VENV_PYTHON="$VENV_PATH/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

if [[ "$CUDA_TORCH" -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
fi

"$VENV_PYTHON" -m pip install -r requirements.txt
"$VENV_PYTHON" scripts/check_local_env.py

echo
echo "Local environment ready."
echo "Activate with: source $VENV_PATH/bin/activate"

