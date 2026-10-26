#!/usr/bin/env bash
# Run inside tmux; all remaining arguments go to tune_gpu.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
exec > >(tee -a benchmarks/results/tuning-console.log) 2>&1

PY=${PYTHON:-python3}
ENV_DIR=${BENCH_VENV:-.venv}
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PY" -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
if [[ ${SKIP_SETUP:-0} != 1 ]]; then
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -e '.[benchmark]'
    # Preserve a user's working CUDA build. Set TORCH_INDEX_URL to select a
    # different official PyTorch wheel index explicitly.
    if [[ -n ${TORCH_INDEX_URL:-} ]]; then
        "$PY" -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
    fi
fi
"$PY" -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "No CUDA device. Install a CUDA PyTorch build and check nvidia-smi; this launcher requires a GPU."'
if [[ ${SKIP_SETUP:-0} != 1 ]]; then
    CUDA_MAJOR=$("$PY" -c 'import torch; print(torch.version.cuda.split(".")[0])')
    case "$CUDA_MAJOR" in
        11|12|13) "$PY" -m pip install "cupy-cuda${CUDA_MAJOR}x>=13.3" ;;
        *) echo "Install a CuPy wheel matching CUDA $CUDA_MAJOR before running."; exit 1 ;;
    esac
    "$PY" third_party/setup_external.py
fi
"$PY" -c 'import cupy; import torch; print("GPU:", torch.cuda.get_device_name(0)); print("CuPy:", cupy.__version__); cupy.zeros(1); cupy.cuda.Device().synchronize()'
# Fail before the overnight sweep if CUDA kernels or numerical parity break.
"$PY" -m pytest -q tests/test_gpu_benchmarks.py
"$PY" benchmarks/tune_gpu.py --check
exec "$PY" benchmarks/tune_gpu.py "$@"
