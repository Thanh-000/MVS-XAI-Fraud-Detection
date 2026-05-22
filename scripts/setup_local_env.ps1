param(
    [string]$VenvPath = ".venv",
    [switch]$CudaTorch,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw "Python was not found on PATH. Install Python 3.10+ and rerun."
}

$pythonExe = Resolve-Python
Write-Host "Using Python: $pythonExe"

if (!(Test-Path $VenvPath)) {
    & $pythonExe -m venv $VenvPath
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    throw "Virtual environment Python not found: $venvPython"
}

& $venvPython -m pip install --upgrade pip setuptools wheel

if ($CudaTorch) {
    Write-Host "Installing CUDA PyTorch from $TorchIndexUrl"
    & $venvPython -m pip install --upgrade torch --index-url $TorchIndexUrl
}

& $venvPython -m pip install -r requirements.txt
& $venvPython scripts\check_local_env.py

Write-Host ""
Write-Host "Local environment ready."
Write-Host "Activate with: $VenvPath\Scripts\Activate.ps1"

