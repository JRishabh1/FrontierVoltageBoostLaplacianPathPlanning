#!/usr/bin/env bash
# One-shot setup + full benchmark suite, meant to run inside tmux:
#
#   tmux new -d -s bench 'bash benchmarks/start_suite.sh'
#
# Creates .venv (if missing), installs dependencies (PyTorch with CUDA on
# Linux), builds the external planners, then runs benchmarks/run_suite.py
# with any extra arguments. Safe to re-run: the suite resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
exec > >(tee -a benchmarks/results/console.log) 2>&1

PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
    echo "creating .venv with $($PY --version)"
    $PY -m venv .venv || { echo "python venv failed (on Ubuntu: apt install python3-venv)"; exit 1; }
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e . numpy scipy pillow flask rtree pybind11 pyyaml torch
.venv/bin/python third_party/setup_external.py
.venv/bin/python benchmarks/run_suite.py --check
.venv/bin/python benchmarks/run_suite.py "$@"
echo "start_suite.sh finished at $(date)"
