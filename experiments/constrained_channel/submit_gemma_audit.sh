#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."
: "${QIF_GEMMA_MODEL:?Export the path to your existing local Gemma 2 9B snapshot}"
test -f "$QIF_GEMMA_MODEL/config.json"
mkdir -p logs
printf 'Submitting from %s\n' "$PWD" >&2
exec sbatch --parsable experiments/constrained_channel/gemma_audit.slurm "$@"
