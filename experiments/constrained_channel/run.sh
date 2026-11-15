#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."

printf 'Node: %s\nWorking directory: %s\nCUDA_VISIBLE_DEVICES: %s\n' \
    "$(hostname)" "$PWD" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
printf 'Experiment arguments:'
printf ' %q' "$@"
printf '\n'

export QIF_MODEL_DIR="${QIF_MODEL_DIR:-$PWD/models/Qwen2.5-7B-Instruct}"
if [[ ! -f "$QIF_MODEL_DIR/config.json" || ! -f "$QIF_MODEL_DIR/tokenizer.json" ]]; then
    echo "Set QIF_MODEL_DIR to a complete local snapshot or run prepare_model.sh first." >&2
    exit 2
fi
printf 'Model directory: %s\n' "$QIF_MODEL_DIR"

scratch_base="${CLUSTER_BUILD_TMP:-${SLURM_TMPDIR:-/tmp}}"
local_run="$(mktemp -d "$scratch_base/qif-channel.XXXXXXXX")"
trap 'rm -rf -- "$local_run"' EXIT
export TMPDIR="$local_run"

# Optional site initialization can make a Python interpreter available.
if [[ -n ${QIF_RUNTIME_SCRIPT:-} ]]; then
    source "$QIF_RUNTIME_SCRIPT"
fi

source "$script_dir/runtime.sh"
qif_init_runtime cu124

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

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

"$qif_python" -m experiments.constrained_channel "$@"
