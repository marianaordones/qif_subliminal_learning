#!/bin/bash
#SBATCH --job-name=qif-ablation
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qif-ablation-%j.out
#SBATCH --error=logs/qif-ablation-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
script_dir="$PWD/experiments/constrained_channel"
output_dir="${QIF_ABLATION_DIR:-runs/channel_ablation/$SLURM_JOB_ID}"
mkdir -p "$output_dir"
printf 'job=%s output_dir=%s\n' "$SLURM_JOB_ID" "$output_dir"

exec 8>"$output_dir/.job.lock"
flock -n 8
source "$script_dir/runtime.sh"
qif_init_runtime plot
plot_python="$qif_python"
plot_path="$PYTHONPATH"
bash "$script_dir/run.sh" --ablation "$@" --output-dir "$output_dir"
PYTHONPATH="$plot_path" "$plot_python" -S -m experiments.constrained_channel.plot_ablation \
    "$output_dir/timing.json"
