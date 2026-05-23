$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$venvPython = ".venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    $venvPython = "python"
}

Write-Host "Running canonical academic end-to-end experiment."
Write-Host "This script has no parameters by design."
& $venvPython "run_academic_e2e.py"
