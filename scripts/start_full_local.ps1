param(
    [int]$Threads = 4,
    [string]$JobRoot = "artifacts/local_full/jobs"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Get-Location).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    throw "Missing .venv Python: $venvPython. Run .\scripts\setup_local_env.ps1 first."
}

New-Item -ItemType Directory -Force -Path $JobRoot | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$jobName = "academic_e2e_${timestamp}"
$jobDir = Join-Path $JobRoot $jobName
New-Item -ItemType Directory -Force -Path $jobDir | Out-Null

$logPath = Join-Path $jobDir "train.log"
$statusPath = Join-Path $jobDir "status.json"
$runnerPath = Join-Path $jobDir "runner.ps1"

$runner = @"
`$ErrorActionPreference = "Continue"
Set-Location '$repoRoot'
`$env:PYTHONUNBUFFERED = "1"
`$env:OMP_NUM_THREADS = "$Threads"
`$env:MKL_NUM_THREADS = "$Threads"
`$env:OPENBLAS_NUM_THREADS = "$Threads"
`$env:NUMEXPR_NUM_THREADS = "$Threads"

`$started = Get-Date
`$status = [ordered]@{
    job = '$jobName'
    run_type = 'academic_e2e'
    started = `$started.ToString("o")
    finished = `$null
    exit_code = `$null
    log = '$logPath'
    artifacts_root = 'artifacts/academic_e2e'
}
`$status | ConvertTo-Json | Set-Content -Encoding UTF8 '$statusPath'

try {
    & '$repoRoot\scripts\run_full_experiment.ps1' 2>&1 | Tee-Object -FilePath '$logPath'
    `$exitCode = if (`$null -eq `$LASTEXITCODE) { 0 } else { `$LASTEXITCODE }
}
catch {
    `$_ | Out-String | Tee-Object -FilePath '$logPath' -Append
    `$exitCode = 1
}

`$status.finished = (Get-Date).ToString("o")
`$status.exit_code = `$exitCode
`$status | ConvertTo-Json | Set-Content -Encoding UTF8 '$statusPath'
exit `$exitCode
"@

Set-Content -Encoding UTF8 -Path $runnerPath -Value $runner

$proc = Start-Process -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runnerPath) `
    -WindowStyle Hidden `
    -PassThru

[PSCustomObject]@{
    Job = $jobName
    Pid = $proc.Id
    Log = $logPath
    Status = $statusPath
    ArtifactsRoot = "artifacts/academic_e2e"
}
