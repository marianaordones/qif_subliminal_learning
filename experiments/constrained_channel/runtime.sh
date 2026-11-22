#!/bin/bash
# Sourced by the model preparation and GPU launchers.

qif_init_runtime() {
    local profile="$1" requested abi libc requirements digest cache_root probe
    requirements="$script_dir/requirements-$profile.txt"
    case "$profile" in
        download) probe='from huggingface_hub import HfApi, snapshot_download' ;;
        plot) probe='import matplotlib; matplotlib.use("Agg"); from matplotlib import pyplot' ;;
        gemma) probe='import torch, peft, yaml; from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM' ;;
        student) probe='import torch, peft, yaml, accelerate; from transformers import Trainer, TrainingArguments' ;;
        cu124) probe='
import numpy as np
import torch
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

assert torch.version.cuda == "12.4"
assert hasattr(DynamicCache, "batch_repeat_interleave")
assert torch.from_numpy(np.zeros(1)).numel() == 1
' ;;
        *) printf 'Unknown runtime profile: %s\n' "$profile" >&2; return 2 ;;
    esac

    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
    export PYTHONNOUSERSITE=1
    requested="${CLUSTER_PYTHON:-${QIF_PYTHON:-}}"
    if [[ -z "$requested" ]]; then
        requested="$(command -v python3.12 || command -v python3.11 || true)"
    fi
    if [[ -z "$requested" ]] || ! command -v "$requested" >/dev/null; then
        echo "Set CLUSTER_PYTHON to an available Python 3.11 or 3.12 executable." >&2
        return 2
    fi
    qif_python="$("$requested" -S -c 'import os, sys; print(os.path.abspath(sys.executable))')"
    abi="$("$qif_python" -S -c '
import platform
import sys
import sysconfig

if (
    sys.implementation.name != "cpython"
    or sys.version_info[:2] not in ((3, 11), (3, 12))
    or platform.system() != "Linux"
    or platform.machine() != "x86_64"
):
    raise SystemExit("The pinned runtime requires Linux x86-64 and CPython 3.11 or 3.12.")
print(sysconfig.get_config_var("SOABI"))
')"
    libc="$(getconf GNU_LIBC_VERSION)"
    libc="${libc// /-}"
    digest="$(sha256sum "$requirements")"
    digest="${digest%% *}"
    cache_root="${QIF_LIBS_ROOT:-$HOME/pylibs}"
    qif_libs="$cache_root/qif-channel-$profile-$abi-$libc-$digest"
    printf 'Python executable: %s\nProject libraries: %s\n' "$qif_python" "$qif_libs"
    "$qif_python" --version
    mkdir -p "$cache_root"

    # Only the process holding the lock can install or publish this cache.
    (
        flock -x 9
        if [[ -f "$qif_libs/.complete" ]]; then
            exit 0
        fi
        if [[ -e "$qif_libs" ]]; then
            echo "Incomplete cache at $qif_libs. Move it aside or set QIF_LIBS_ROOT." >&2
            exit 2
        fi

        scratch_base="${CLUSTER_BUILD_TMP:-${SLURM_TMPDIR:-/tmp}}"
        install_tmp="$(mktemp -d "$scratch_base/qif-install.XXXXXXXX")"
        stage=""
        trap 'if [[ -n "$stage" ]]; then rm -rf -- "$stage"; fi; rm -rf -- "$install_tmp"' EXIT
        export TMPDIR="$install_tmp"
        stage="$(mktemp -d "$cache_root/.qif-$profile.XXXXXXXX")"
        pip_path=""
        if ! "$qif_python" -m pip --version; then
            echo "Bootstrapping pip 25.0.1 on local disk."
            curl --fail --silent --show-error --location --retry 2 \
                https://bootstrap.pypa.io/get-pip.py -o "$install_tmp/get-pip.py"
            pip_path="$install_tmp/pip"
            "$qif_python" "$install_tmp/get-pip.py" --no-setuptools --no-wheel \
                --no-cache-dir --target "$pip_path" pip==25.0.1
            PYTHONPATH="$pip_path" "$qif_python" -m pip --version
        fi

        PYTHONPATH="$pip_path" "$qif_python" -m pip install --no-cache-dir \
            --ignore-installed --target "$stage" -r "$requirements"
        PYTHONPATH="$stage" "$qif_python" -S -c "$probe"
        touch "$stage/.complete"
        mv -T -- "$stage" "$qif_libs"
        stage=""
    ) 9>"$qif_libs.lock"

    export PYTHONPATH="$PWD:$qif_libs"
    "$qif_python" -S -c "$probe"
}
