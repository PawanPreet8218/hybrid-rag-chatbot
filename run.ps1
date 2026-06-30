# ============================================================
#  Launch the Sage RAG chatbot using the PROJECT venv
#  (NOT the global Python). Run:  ./run.ps1
# ============================================================
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[run] .venv not found. Create it and install deps first:" -ForegroundColor Yellow
    Write-Host "       python -m venv .venv"
    Write-Host "       .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Write-Host "[run] Starting Streamlit with the venv interpreter..." -ForegroundColor Green
& $py -m streamlit run app.py
