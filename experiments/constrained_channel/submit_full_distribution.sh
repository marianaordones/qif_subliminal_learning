#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."
: "${QIF_GEMMA_MODEL:?Export the local Gemma snapshot path}"
test -f "$QIF_GEMMA_MODEL/config.json"
test -f "${QIF_QWEN_MODEL:-$PWD/models/Qwen2.5-7B-Instruct}/config.json"
project="${QIF_TEACHER_PROJECT:-$(dirname "$PWD")/unintended-generalization-lottery-tickets}"
test -f "$project/scripts/sl/train_student.py"
test -f "$project/artifacts/sl/gemma2_9b_teacher_pilot/models/finetuned_teacher/adapter/adapter_config.json"
mkdir -p logs
printf 'Submitting from %s\n' "$PWD" >&2
exec sbatch --parsable experiments/constrained_channel/full_distribution.slurm "$@"
