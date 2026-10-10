@echo off
REM One-click launcher for Windows: creates a venv on first run, installs packages, starts the app.
cd /d "%~dp0"

REM Pick a Python: prefer 3.12/3.11/3.13 (packages have wheels), fall back to default python.
set "PYCMD="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PYCMD (
    py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  python -c "import sys" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo Python not found. Install Python 3.10-3.13 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

REM Rebuild the venv if it is missing or broken (e.g. an interrupted earlier install).
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe -c "import streamlit, pandas, sklearn" >nul 2>&1 || (
    echo Existing venv is incomplete - rebuilding...
    rmdir /s /q venv
  )
)
if not exist venv\Scripts\python.exe (
  echo Creating virtual environment with %PYCMD% ...
  %PYCMD% -m venv venv || (echo Failed to create venv. & pause & exit /b 1)
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
)

echo Starting Dark Pattern Radar... close this window or press Ctrl+C to stop.
venv\Scripts\python.exe -m streamlit run app.py
pause
