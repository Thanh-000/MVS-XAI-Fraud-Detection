#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

echo "Running canonical academic end-to-end experiment."
echo "This script has no parameters by design."
"$PYTHON_BIN" run_academic_e2e.py
