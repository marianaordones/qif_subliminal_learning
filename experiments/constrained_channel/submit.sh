#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."
test -f bayes_leakage.py
test -f experiments/constrained_channel/__main__.py
test -f experiments/constrained_channel/run.sh
test -f experiments/constrained_channel/runtime.sh
test -f experiments/constrained_channel/requirements-cu124.txt
command -v sbatch >/dev/null
mkdir -p logs runs/constrained_channel
printf 'Submitting from %s\n' "$PWD" >&2
exec sbatch --parsable --array="${QIF_ARRAY:-0-1%2}" experiments/constrained_channel/compare.slurm "$@"
