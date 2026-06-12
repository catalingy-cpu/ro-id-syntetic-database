# Antrenare PaddleX + deploy in services/paddle-ocr (fara copiere manuala)
param(
    [string]$Dataset = "dataset",
    [string]$Device = "cpu",
    [int]$Epochs = 15,
    [int]$BatchSize = 0,
    [string]$ExportDir = "",
    [string]$ModelName = "frc_ci_rec",
    [string]$PaddleOcrRoot = "",
    [switch]$SkipTrain,
    [switch]$SkipExport,
    [switch]$SkipDeploy,
    [switch]$InstallPaddleocrRepo,
    [switch]$SkipCheckDataset,
    [switch]$Fast
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if ($PaddleOcrRoot) {
    $PaddleRoot = Resolve-Path $PaddleOcrRoot
} elseif (Test-Path "..\paddle-ocr\requirements.txt") {
    $PaddleRoot = Resolve-Path "..\paddle-ocr"
} else {
    $PaddleRoot = Get-Location
}

$PaddlePy = Join-Path $PaddleRoot.Path ".venv\Scripts\python.exe"
if (-not (Test-Path $PaddlePy)) {
    $PaddlePy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PaddlePy)) {
    Write-Error "Lipseste venv. FRCHub: cd services\paddle-ocr ; .\run.ps1`nStandalone: python -m venv .venv ; pip install -r requirements-paddle-train.txt"
}
if ($PaddleOcrRoot) { $env:PADDLE_OCR_ROOT = $PaddleRoot.Path }

$env:FLAGS_use_mkldnn = "0"
$env:FLAGS_enable_mkldnn = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:MKL_DEBUG_CPU_TYPE = "5"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"

$argsList = @(
    "scripts\train_and_deploy.py",
    "--dataset", $Dataset,
    "--device", $Device,
    "--epochs", $Epochs,
    "--paddle-python", $PaddlePy,
    "--model-name", $ModelName
)
if ($SkipTrain) { $argsList += "--skip-train" }
if ($SkipExport) { $argsList += "--skip-export" }
if ($ExportDir) {
    $argsList += "--export-dir"
    $argsList += $ExportDir
}
if ($InstallPaddleocrRepo) { $argsList += "--install-paddleocr-repo" }
if ($SkipCheckDataset) { $argsList += "--skip-check-dataset" }
if ($SkipDeploy) { $argsList += "--skip-deploy" }
if ($Fast) { $argsList += "--fast" }
if ($PaddleOcrRoot) {
    $argsList += "--paddle-ocr-root"
    $argsList += $PaddleRoot.Path
}
if ($BatchSize -gt 0) {
    $argsList += "--batch-size"
    $argsList += $BatchSize
}

& $PaddlePy @argsList
