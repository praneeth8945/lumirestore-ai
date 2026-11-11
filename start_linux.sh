#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# PyTorch 2.6 publishes wheels for Python 3.11, 3.12 and 3.13 only.
if [ ! -d ".venv" ]; then
    PYTHON_BIN=""
    for candidate in python3.12 python3.13 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON_BIN" ]; then
        echo "Error: LumiRestore AI needs Python 3.11, 3.12 or 3.13 (PyTorch 2.6 has no wheels for newer versions)." >&2
        echo "Install a supported Python version and run this script again." >&2
        exit 1
    fi
    echo "Creating virtual environment with $("$PYTHON_BIN" --version)"
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
exec python -m streamlit run app.py
