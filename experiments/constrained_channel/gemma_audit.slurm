#!/bin/bash
#SBATCH --job-name=qif-gemma
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/qif-gemma-%A_%a.out
#SBATCH --error=logs/qif-gemma-%A_%a.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the qif repository root}"
script_dir="$PWD/experiments/constrained_channel"
case "$SLURM_ARRAY_TASK_ID" in
    0) arm=finetuned ;;
    1) arm=prompt ;;
    *) exit 2 ;;
esac
project="${QIF_TEACHER_PROJECT:-$(dirname "$PWD")/unintended-generalization-lottery-tickets}"
model="${QIF_GEMMA_MODEL:?Set QIF_GEMMA_MODEL to the existing Gemma 2 9B local snapshot}"
output="${QIF_GEMMA_RUN:-runs/gemma_audit/$SLURM_ARRAY_JOB_ID}/$arm"
printf 'job=%s arm=%s node=%s cwd=%s output=%s\n' \
    "$SLURM_ARRAY_JOB_ID" "$arm" "$(hostname)" "$PWD" "$output"
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
    --arm "$arm" --project "$project" --model "$model" --output-dir "$output" "$@"
