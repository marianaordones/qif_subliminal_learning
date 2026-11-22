#!/bin/bash
#SBATCH --job-name=qif-qwen-cat
#SBATCH --partition=high_performance
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/qif-qwen-cat-%A_%a.out
#SBATCH --error=logs/qif-qwen-cat-%A_%a.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case "$SLURM_ARRAY_TASK_ID" in
    0) arm=prompt ;;
    1) arm=finetuned ;;
    *) exit 2 ;;
esac
output="${QIF_CAT_RUN:-runs/qwen_cat/$SLURM_ARRAY_JOB_ID}/$arm"
model="${QIF_CAT_MODEL:-$PWD/models/unsloth-Qwen2.5-7B-Instruct-cat}"
printf 'job=%s task=%s arm=%s node=%s cwd=%s output=%s\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$arm" "$(hostname)" "$PWD" "$output"
nvidia-smi
mkdir -p "$output" "$model"
exec 8>"$output/.job.lock"
flock -n 8
scratch="$(mktemp -d "${CLUSTER_BUILD_TMP:-${SLURM_TMPDIR:-/tmp}}/qif-cat.XXXXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
if [[ -n ${QIF_RUNTIME_SCRIPT:-} ]]; then source "$QIF_RUNTIME_SCRIPT"; fi
script_dir="$PWD/experiments/constrained_channel"
source "$script_dir/runtime.sh"
qif_init_runtime student
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
(
    flock -x 9
    "$qif_python" -S - "$model" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

path = Path(sys.argv[1])
revision = "a75c9dc945567a9b6f568b8503a0307731607bee"
marker = path / ".qif-revision"
if marker.exists():
    if marker.read_text().strip() != revision:
        raise SystemExit("Local snapshot revision differs.")
else:
    snapshot_download("unsloth/Qwen2.5-7B-Instruct", revision=revision, local_dir=path,
                      allow_patterns=["*.json", "*.txt", "*.jinja", "model*.safetensors"])
    marker.write_text(revision + "\n")
PY
) 9>"$model/.download.lock"
"$qif_python" -S -m experiments.qwen_cat.pipeline --arm "$arm" --model "$model" --output "$output"
