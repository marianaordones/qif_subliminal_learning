#!/bin/bash
#SBATCH --job-name=qif-qwen-prompt
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qif-qwen-prompt-%j.out
#SBATCH --error=logs/qif-qwen-prompt-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the qif repository root}"
script_dir="$PWD/experiments/constrained_channel"
project="${QIF_TEACHER_PROJECT:-$(dirname "$PWD")/unintended-generalization-lottery-tickets}"
model="${QIF_QWEN_MODEL:-$PWD/models/Qwen2.5-7B-Instruct}"
output="${QIF_QWEN_RUN:-runs/qwen_audit/$SLURM_JOB_ID}"
printf 'job=%s node=%s cwd=%s output=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$PWD" "$output"
test -f "$project/configs/sl/gemma2_9b_numbers_pilot.yaml"
test -f "$model/config.json"
test -f "$model/tokenizer_config.json"
nvidia-smi
mkdir -p "$output"
exec 8>"$output/.job.lock"
flock -n 8
if [[ -n ${QIF_RUNTIME_SCRIPT:-} ]]; then
    source "$QIF_RUNTIME_SCRIPT"
fi
source "$script_dir/runtime.sh"
qif_init_runtime gemma
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
"$qif_python" -S -m experiments.constrained_channel.gemma_audit \
    --arm prompt --model-family qwen --project "$project" --model "$model" \
    --output-dir "$output" --samples 1000 "$@"
