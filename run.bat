@echo off
REM ============================================================
REM  Launch the Sage RAG chatbot using the PROJECT venv
REM  (NOT the global Python). Just double-click this file.
REM ============================================================
cd /d "%~dp0"
set PYTHONUTF8=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

if not exist ".venv\Scripts\python.exe" (
    echo [run] .venv not found. Create it and install deps first:
    echo        python -m venv .venv
    echo        .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [run] Starting Streamlit with the venv interpreter...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
