param(
    [ValidateSet("ieee", "paysim", "both")]
    [string]$Dataset = "both",
    [string]$DataDir = "data",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Seed = 42,
    [int]$NSeeds = 3,
    [int]$NSplits = 5,
    [int]$GapSize = 1000,
    [double]$TestRatio = 0.15,
    [double]$SmoteStrategy = 0.30,
    [int]$PaysimChunkSize = 750000,
    [int]$PaysimStepBlockSize = 24,
    [string]$ArtifactsRoot = "artifacts/local_full"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$venvPython = ".venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    $venvPython = "python"
}

function Invoke-FullDataset {
    param([string]$Name)

    $artifactDir = Join-Path $ArtifactsRoot "$($Name)_academic_full"
    New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

    $args = @(
        "main_train_pipeline.py",
        "--dataset", $Name,
        "--data_dir", $DataDir,
        "--device", $Device,
        "--test_ratio", "$TestRatio",
        "--n_splits", "$NSplits",
        "--gap_size", "$GapSize",
        "--seed", "$Seed",
        "--n_seeds", "$NSeeds",
        "--smote_strategy", "$SmoteStrategy",
        "--ctgan_samples", "0",
        "--preset", "full_mvs",
        "--model_profile", "research",
        "--artifacts_dir", $artifactDir
    )

    if ($Name -eq "paysim") {
        $args += @(
            "--paysim_chunk_size", "$PaysimChunkSize",
            "--paysim_step_block_size", "$PaysimStepBlockSize"
        )
    }

    Write-Host ""
    Write-Host "Running full local experiment: $Name"
    Write-Host "Artifacts: $artifactDir"
    & $venvPython @args
}

if ($Dataset -eq "ieee" -or $Dataset -eq "both") {
    Invoke-FullDataset -Name "ieee"
}

if ($Dataset -eq "paysim" -or $Dataset -eq "both") {
    Invoke-FullDataset -Name "paysim"
}

