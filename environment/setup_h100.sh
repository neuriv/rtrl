#!/usr/bin/env bash
# Ubuntu 24.04 / x86_64, Python 3.12, one H100 or A100. No sudo/system changes.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'Usage: bash setup_h100.sh' \
        'Requires Linux x86_64, glibc >= 2.39, Python 3.12, NVIDIA driver >= 580.' \
        'Current R615+ GPU images recommended for the locked CUDA JIT dependencies.' \
        'Creates .venv-h100; downloads pinned binary packages; runs GPU and unit checks.' \
        'Then run .venv-h100/bin/python -m rtrl.smoke_h100 --output-dir ../rtrl-runs/smoke-1'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || {
    echo 'Run this on the Linux GPU host, not the Mac.' >&2; exit 1;
}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${RTRL_PYTHON:-python3.12}"
"$python_bin" - <<'PY'
import platform
import subprocess
import sys

assert sys.version_info[:2] == (3, 12), 'Python 3.12 required'
libc, version = platform.libc_ver()
assert libc == 'glibc' and tuple(map(int, version.split('.'))) >= (2, 39), 'glibc >= 2.39 required (Ubuntu 24.04)'
drivers = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()
assert drivers and min(int(v.split('.')[0]) for v in drivers) >= 580, 'CUDA 13 requires an R580+ driver'
if any(int(v.split('.')[0]) < 615 for v in drivers):
    print('Driver prerequisite passed; this lock includes CUDA 13.4 JIT packages. R615+ recommended; model smoke required.', flush=True)
PY
"$python_bin" -m venv .venv-h100
python_bin="$PWD/.venv-h100/bin/python"
"$python_bin" -m pip install --disable-pip-version-check 'uv==0.12.18'
.venv-h100/bin/uv pip install --python "$python_bin" --only-binary :all: -r environment/requirements-h100.txt
.venv-h100/bin/uv pip install --python "$python_bin" --no-deps --no-build-isolation -e .
.venv-h100/bin/uv pip check --python "$python_bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$python_bin" - <<'PY'
from importlib.metadata import version

import torch
import vllm._C
from vllm.v1.engine.async_llm import AsyncLLM

from rtrl.collect import cuda_device

gpu = cuda_device()
x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
loss = (x @ x.T).float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all(), 'Nonfinite BF16 forward/backward'
print({'gpu': gpu, 'memory_GiB': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
       'torch_cuda': torch.version.cuda, 'versions': {p: version(p) for p in ('torch', 'vllm', 'transformers')}})
PY
"$python_bin" -m pytest -q
printf '%s\n' 'Environment checks passed. Real-model execution still needs the smoke test:' \
    '.venv-h100/bin/python -m rtrl.smoke_h100 --output-dir ../rtrl-runs/smoke-1'
