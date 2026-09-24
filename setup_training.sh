#!/usr/bin/env bash
# Ubuntu 22.04+, Linux x86_64, one H100, NVIDIA driver 570+ / CUDA 12.8.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'Usage: bash setup_training.sh' \
        'Installs uv 0.12.18, managed Python 3.12 and pinned CUDA 12.8 training packages.' \
        'Creates .venv-training; checks GPU imports and runs unit tests.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || {
    echo 'Run this on the Linux GPU host, not the Mac.' >&2; exit 1;
}
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_PYTHON_INSTALL_DIR="$PWD/.venv-tools/python"
if [[ ! -x .venv-tools/uv ]]; then
    curl -LsSf https://astral.sh/uv/0.12.18/install.sh |
        env UV_INSTALL_DIR="$PWD/.venv-tools" UV_NO_MODIFY_PATH=1 sh
fi
[[ "$(.venv-tools/uv --version)" == 'uv 0.12.18'* ]] || {
    echo 'Expected uv 0.12.18 in .venv-tools.' >&2; exit 1;
}
.venv-tools/uv python install 3.12
[[ -x .venv-training/bin/python ]] || .venv-tools/uv venv --python 3.12 .venv-training
python_bin="$PWD/.venv-training/bin/python"
.venv-tools/uv pip install --python "$python_bin" --only-binary :all: \
    --index-strategy unsafe-best-match -r requirements-training.txt
.venv-tools/uv pip install --python "$python_bin" --no-deps --no-build-isolation -e .
.venv-tools/uv pip check --python "$python_bin"
"$python_bin" - <<'PY'
from importlib.metadata import version
import sys

import torch
import transformers
import wandb

assert sys.version_info[:2] == (3, 12), 'Python 3.12 required'
assert torch.version.cuda == '12.8', 'CUDA 12.8 wheel required'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible GPU required'
gpu = torch.cuda.get_device_properties(0)
assert 'H100' in gpu.name, f'H100 required, found {gpu.name}'
assert torch.cuda.is_bf16_supported(), 'BF16 support required'
print({'gpu': gpu.name, 'memory_GiB': round(gpu.total_memory / 2**30, 1),
       'torch_cuda': torch.version.cuda,
       'versions': {p: version(p) for p in ('torch', 'transformers', 'wandb')}})
PY
"$python_bin" -m pytest -q
printf '%s\n' 'Environment checks passed. Ready for comparative training.'
