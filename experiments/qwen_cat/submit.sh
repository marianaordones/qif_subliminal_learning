#!/bin/bash
set -Eeuo pipefail
directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$directory/../.."
mkdir -p logs
printf 'Submitting from %s\n' "$PWD" >&2
exec sbatch --parsable experiments/qwen_cat/run.slurm
