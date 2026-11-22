#!/bin/bash
set -Eeuo pipefail
directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$directory/../.."
mkdir -p logs
printf 'Submitting owl prompt experiment from %s\n' "$PWD" >&2
submission=$(sbatch --parsable experiments/qwen_owl/run.slurm)
job_id=${submission%%;*}
receipt="logs/qif-qwen-owl-${job_id}.submission.txt"
printf 'job_id=%s\ncwd=%s\nscript=%s\noutput=%s/prompt\nmodel=%s\nstdout=logs/qif-qwen-owl-%s.out\nstderr=logs/qif-qwen-owl-%s.err\n' \
    "$job_id" "$PWD" experiments/qwen_owl/run.slurm \
    "${QIF_OWL_RUN:-runs/qwen_owl/$job_id}" \
    "${QIF_OWL_MODEL:-$PWD/models/unsloth-Qwen2.5-7B-Instruct-cat}" \
    "$job_id" "$job_id" > "$receipt"
printf 'Submission receipt: %s\n' "$receipt" >&2
printf '%s\n' "$submission"
