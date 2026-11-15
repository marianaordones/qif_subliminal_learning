#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."

# Initialize inside the allocation. The runtime script can set PATH or
# QIF_PYTHON. An explicit Python takes precedence over the fallback venv.
if [[ -n ${QIF_RUNTIME_SCRIPT:-} ]]; then
    source "$QIF_RUNTIME_SCRIPT"
    if [[ -z ${QIF_PYTHON:-} ]]; then
        QIF_PYTHON="$(command -v python || command -v python3)"
    fi
fi

if [[ -n ${QIF_PYTHON:-} ]]; then
    qif_python="$QIF_PYTHON"
else
    venv_dir="${QIF_VENV_DIR:-$PWD/.venv/qif-cu124}"
    mkdir -p "$venv_dir"
    (
        flock -x 9
        if [[ ! -f "$venv_dir/.qif-cu124-ready" ]]; then
            "${QIF_BOOTSTRAP_PYTHON:-python3}" -m venv "$venv_dir"
            "$venv_dir/bin/python" -m pip install torch==2.5.1 \
                --index-url https://download.pytorch.org/whl/cu124
            "$venv_dir/bin/python" -m pip install transformers==4.57.1
            touch "$venv_dir/.qif-cu124-ready"
        fi
    ) 9>"$venv_dir/.install.lock"
    qif_python="$venv_dir/bin/python"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
printf 'Node: %s\nWorking directory: %s\nCUDA_VISIBLE_DEVICES: %s\n' \
    "$(hostname)" "$PWD" "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'Experiment arguments:'
printf ' %q' "$@"
printf '\n'

"$qif_python" - "$@" <<'PY'
import platform

import torch
import transformers
from transformers import DynamicCache

from experiments.constrained_channel.__main__ import parse_args

args = parse_args()
print(f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
      f"Transformers {transformers.__version__}, CUDA build {torch.version.cuda}")
if not hasattr(DynamicCache, "batch_repeat_interleave"):
    raise SystemExit("This Transformers runtime lacks DynamicCache.batch_repeat_interleave.")
device = torch.device(args.device)
if device.type != "cuda" or not torch.cuda.is_available():
    raise SystemExit("A visible CUDA GPU is required.")
with torch.cuda.device(device):
    print(f"Allocated GPU: {torch.cuda.get_device_name(device)}")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("The allocated GPU does not support BF16.")
    small = torch.ones((2, 2), device=device, dtype=getattr(torch, args.dtype))
    if (small @ small).sum().item() != 8:
        raise SystemExit("CUDA smoke test returned an unexpected result.")
print("CUDA smoke test passed.")
PY

exec "$qif_python" -m experiments.constrained_channel "$@"
