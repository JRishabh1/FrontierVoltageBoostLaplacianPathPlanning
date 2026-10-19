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
# (Re)create .venv unless it is a working environment with pip.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    rm -rf .venv
    echo "creating .venv with $($PY --version)"
    if [ -z "${NO_ENSUREPIP:-}" ] && $PY -m venv .venv; then
        :
    else
        # Ubuntu without python3-venv: no ensurepip. Create the env without
        # pip and install pip with PyPA's official bootstrap script instead.
        echo "venv module unavailable; creating .venv without pip and bootstrapping pip (get-pip.py)"
        rm -rf .venv
        $PY -m venv --without-pip .venv
        .venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '.venv/get-pip.py')"
        .venv/bin/python .venv/get-pip.py -q
    fi
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e . numpy scipy pillow flask rtree pybind11 pyyaml torch
.venv/bin/python third_party/setup_external.py
.venv/bin/python benchmarks/run_suite.py --check
.venv/bin/python benchmarks/run_suite.py "$@"
echo "start_suite.sh finished at $(date)"
