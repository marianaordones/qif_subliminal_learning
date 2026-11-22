#!/bin/bash
#SBATCH --job-name=qif-fullsoftmax
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/qif-fullsoftmax-%A_%a.out
#SBATCH --error=logs/qif-fullsoftmax-%A_%a.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the qif repository root}"
script_dir="$PWD/experiments/constrained_channel"
case "$SLURM_ARRAY_TASK_ID" in
    0) family=qwen; induction=prompt ;;
    1) family=qwen; induction=finetuned ;;
    2) family=gemma; induction=prompt ;;
    3) family=gemma; induction=finetuned ;;
    *) exit 2 ;;
esac
project="${QIF_TEACHER_PROJECT:-$(dirname "$PWD")/unintended-generalization-lottery-tickets}"
if [[ "$family" == qwen ]]; then
    model="${QIF_QWEN_MODEL:-$PWD/models/Qwen2.5-7B-Instruct}"
else
    model="${QIF_GEMMA_MODEL:?Set QIF_GEMMA_MODEL to the local Gemma snapshot}"
fi
root="${QIF_FULL_RUN:-runs/full_distribution/$SLURM_ARRAY_JOB_ID}"
output="$root/${family}_${induction}"
test -f "$model/config.json"
test -f "$project/scripts/sl/train_student.py"
mkdir -p "$output"
printf 'job=%s task=%s node=%s cwd=%s output=%s\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$PWD" "$output"
nvidia-smi
exec 8>"$output/.job.lock"
flock -n 8
local_run="$(mktemp -d "${CLUSTER_BUILD_TMP:-${SLURM_TMPDIR:-/tmp}}/qif-sft.XXXXXXXX")"
trap 'rm -rf -- "$local_run"' EXIT
export TMPDIR="$local_run"
if [[ -n ${QIF_RUNTIME_SCRIPT:-} ]]; then
    source "$QIF_RUNTIME_SCRIPT"
fi
source "$script_dir/runtime.sh"
qif_init_runtime student
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
"$qif_python" -S -m experiments.constrained_channel.full_distribution \
    --family "$family" --induction "$induction" --project "$project" \
    --model "$model" --output "$output" "$@"
