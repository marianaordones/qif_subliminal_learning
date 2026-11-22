#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."
model="${QIF_QWEN_MODEL:-$PWD/models/Qwen2.5-7B-Instruct}"
test -f "$model/config.json"
mkdir -p logs
printf 'Submitting from %s\n' "$PWD" >&2
exec sbatch --parsable experiments/constrained_channel/qwen_audit.slurm "$@"
