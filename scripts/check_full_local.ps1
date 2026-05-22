param(
    [string]$JobRoot = "artifacts/local_full/jobs",
    [int]$Tail = 80
)

$ErrorActionPreference = "Stop"

Write-Host "Running training processes:"
$processes = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*main_train_pipeline.py*" -or $_.CommandLine -like "*start_full_local*" -or $_.CommandLine -like "*run_full_experiment.ps1*" } |
    Select-Object ProcessId, Name, CreationDate, CommandLine

if ($processes) {
    $processes | Format-List
}
else {
    Write-Host "  none"
}

if (!(Test-Path $JobRoot)) {
    Write-Host ""
    Write-Host "No job directory found: $JobRoot"
    exit 0
}

$latest = Get-ChildItem -Directory $JobRoot | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$latest) {
    Write-Host ""
    Write-Host "No jobs found under $JobRoot"
    exit 0
}

$statusPath = Join-Path $latest.FullName "status.json"
$logPath = Join-Path $latest.FullName "train.log"

Write-Host ""
Write-Host "Latest job: $($latest.Name)"

if (Test-Path $statusPath) {
    Write-Host ""
    Write-Host "Status:"
    Get-Content $statusPath
}
else {
    Write-Host "Status file missing: $statusPath"
}

if (Test-Path $logPath) {
    Write-Host ""
    Write-Host "Log tail:"
    Get-Content $logPath -Tail $Tail
}
else {
    Write-Host "Log file missing: $logPath"
}

