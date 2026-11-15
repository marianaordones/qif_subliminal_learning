#!/bin/bash
#SBATCH --job-name=qif-channel
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/qif-channel-%A_%a.out
#SBATCH --error=logs/qif-channel-%A_%a.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

case "${SLURM_ARRAY_TASK_ID:?This job must be submitted as an array}" in
    0) method=exact ;;
    1) method=mc ;;
    *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

for arg in "$@"; do
    case "$arg" in
        --method|--method=*|--output|--output=*|--checkpoint-dir|--checkpoint-dir=*)
            echo "The Slurm job configures the method and output paths." >&2
            exit 2
            ;;
    esac
done

result_dir=runs/constrained_channel
mkdir -p "$result_dir"
run_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_r${SLURM_RESTART_COUNT:-0}"
output="$result_dir/${run_id}_${method}.json"
printf 'job=%s task=%s node=%s cwd=%s output=%s\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$PWD" "$output"
checkpoint_args=()
if [[ "$method" == exact ]]; then
    checkpoint_dir="${QIF_CHECKPOINT_DIR:-$result_dir/checkpoints/$SLURM_ARRAY_JOB_ID}"
    mkdir -p "$checkpoint_dir"
    checkpoint_args=(--checkpoint-dir "$checkpoint_dir")
fi
exec bash experiments/constrained_channel/run.sh \
    --method "$method" \
    --output "$output" "${checkpoint_args[@]}" \
    "$@"
