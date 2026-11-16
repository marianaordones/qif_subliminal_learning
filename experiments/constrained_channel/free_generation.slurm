#!/bin/bash
#SBATCH --job-name=qif-free-generation
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qif-free-generation-%j.out
#SBATCH --error=logs/qif-free-generation-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

result_dir=runs/free_generation
mkdir -p "$result_dir"
output="$result_dir/${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}.json"
printf 'job=%s output=%s\n' "$SLURM_JOB_ID" "$output"
exec bash experiments/constrained_channel/run.sh --free-generation --output "$output"
