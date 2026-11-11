@echo off
setlocal
cd /d "%~dp0"

rem PyTorch 2.6 publishes wheels for Python 3.11, 3.12 and 3.13 only.
if not exist ".venv\Scripts\python.exe" (
    set "PY_VERSION="
    for %%V in (3.12 3.13 3.11) do (
        if not defined PY_VERSION (
            py -%%V -c "import sys" >nul 2>&1 && set "PY_VERSION=%%V"
        )
    )
    if not defined PY_VERSION (
        echo Error: LumiRestore AI needs Python 3.11, 3.12 or 3.13.
        echo PyTorch 2.6 has no wheels for newer Python versions.
        echo Install Python 3.12 from https://www.python.org/downloads/ and run this script again.
        pause
        exit /b 1
    )
    call echo Creating virtual environment with Python %%PY_VERSION%%
    call py -%%PY_VERSION%% -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m streamlit run app.py
