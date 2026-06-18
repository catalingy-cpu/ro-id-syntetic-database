# Smoke pipeline: generează dataset mic + pregătește crop-uri ROI pentru antrenare Colab/GPU.
# Rulează din: services\ro-id-synthetic-dataset
#
# Pași după script:
#   1. Colab: services\ro-id-synthetic-dataset\colab\train_paddleocr.ipynb
#      TRAIN_PROFILE=smoke, GENERATE_DATASET=False, UPLOAD_DATASET_ZIP=True
#      (încarcă dataset_smoke.zip generat aici)
#   2. Sau local cu GPU: .\scripts\train_and_deploy.ps1 -Dataset dataset_smoke -Fast -Device gpu:0
#
param(
    [int]$Count = 800,
    [string]$Output = "dataset_smoke",
    [int]$Workers = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "Creez venv ro-id-synthetic-dataset..."
    python -m venv .venv
    & .\.venv\Scripts\pip.exe install -r requirements.txt
}

Write-Host "=== 1/3 Generare $Count imagini CI sintetice -> $Output ==="
& $VenvPy generate.py --count $Count --workers $Workers --batch-size 100 --output $Output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 2/3 Pregătire line-crops PaddleX (ROI din template_fields.json) ==="
if (Test-Path (Join-Path $Output "paddlex")) {
    Remove-Item -Recurse -Force (Join-Path $Output "paddlex")
}
& $VenvPy scripts\prepare_paddlex_dataset.py --dataset $Output --fields config\template_fields.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ZipPath = Join-Path $Root "$Output.zip"
Write-Host "=== 3/3 Arhivă pentru Colab: $ZipPath ==="
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path (Join-Path $Root $Output) -DestinationPath $ZipPath

$TrainLines = (Get-Content (Join-Path $Output "paddlex\train.txt") | Measure-Object -Line).Lines
$ValLines = (Get-Content (Join-Path $Output "paddlex\val.txt") | Measure-Object -Line).Lines
Write-Host ""
Write-Host "Gata. Line-crops: train=$TrainLines val=$ValLines"
Write-Host "Încarcă $ZipPath în Google Drive și rulează colab/train_paddleocr.ipynb (smoke)."
Write-Host "După deploy model: decomentează PADDLE_REC_MODEL_DIR în services\paddle-ocr\.env"
