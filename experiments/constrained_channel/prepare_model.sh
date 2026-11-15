#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/../.."
model_dir="${QIF_MODEL_DIR:-$PWD/models/Qwen2.5-7B-Instruct}"
revision="${1:-main}"
mkdir -p "$model_dir"

source "$script_dir/runtime.sh"
qif_init_runtime download

exec 9>"$model_dir/.qif-download.lock"
flock -x 9
"$qif_python" - "$model_dir" "$revision" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

model_dir = Path(sys.argv[1])
requested = sys.argv[2]
repo = "Qwen/Qwen2.5-7B-Instruct"
marker = model_dir / ".qif-revision"
if marker.exists():
    current = marker.read_text(encoding="utf-8").strip()
    if requested not in ("main", current):
        raise SystemExit("Set QIF_MODEL_DIR to a new directory for a different revision.")
    revision = current
else:
    revision = HfApi().model_info(repo, revision=requested).sha
snapshot_download(
    repo, revision=revision, local_dir=model_dir,
    allow_patterns=["*.json", "*.txt", "*.jinja", "model*.safetensors"],
)
temporary = marker.with_suffix(".tmp")
temporary.write_text(revision + "\n", encoding="utf-8")
temporary.replace(marker)
print(f"Model ready: {model_dir} ({revision})")
PY
