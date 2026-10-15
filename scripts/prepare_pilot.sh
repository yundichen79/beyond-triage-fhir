#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load Python/3.12.3-GCCcore-13.3.0
python3 -m venv .venv
# CUDA 12.6 wheels include their runtime; no system-wide package changes.
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir \
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-pilot.txt
.venv/bin/python -m pilot.run prepare
