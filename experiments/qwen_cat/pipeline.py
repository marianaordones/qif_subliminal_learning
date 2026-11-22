"""Pinned cat reproduction and a matched fine-tuned-teacher extension."""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import runpy
import subprocess
import sys

import torch
import yaml


HERE = Path(__file__).resolve().parent
CODE = HERE / "vendor/code"
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
REVISION = "a75c9dc945567a9b6f568b8503a0307731607bee"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("prompt", "finetuned"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(CODE / "src"))
    from spar.sl import modeling
    from spar.sl.prompts import FAVORITE_ANIMAL_PROMPTS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    print(f"python={sys.executable} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    assert (torch.ones(2, device="cuda") + 1).sum().item() == 4
    config = yaml.safe_load((CODE / "configs/sl/reproduction/qwen25_7b_cat_replication.yaml").read_text())
    marker = args.model / ".qif-revision"
    if not marker.is_file() or marker.read_text().strip() != REVISION:
        raise ValueError("Model directory must contain the pinned snapshot and .qif-revision marker.")
    config["model"].update(name=str(args.model.resolve()), revision=None)
    config["experiment"]["output_root"] = str(root)
    # Effective batch 66 as in the reference, using smaller GPU microbatches.
    config["training"].update(per_device_batch_size=2, gradient_accumulation_steps=33)
    original_load = modeling.load_model
    numeric_teacher = False
    teacher_adapter = root / "teacher/models/trait/adapter"

    def load_model(cfg, adapter=None, trainable=False):
        if numeric_teacher and args.arm == "finetuned":
            adapter = str(teacher_adapter)
        model = original_load(cfg, adapter=adapter, trainable=trainable)
        if not hasattr(load_model, "logged"):
            print("Checkpoint generation configuration:", model.generation_config.to_json_string(), flush=True)
            load_model.logged = True
        return model

    modeling.load_model = load_model

    def execute(script, cfg, extra):
        digest = hashlib.sha256(json.dumps(extra).encode()).hexdigest()[:12]
        path = root / f"{script}-{cfg['experiment']['name']}-{digest}.yaml"
        payload = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"Configuration changed: {path}")
        path.write_text(payload)
        done = path.with_suffix(".done")
        if done.exists():
            print(f"Reusing completed stage: {path.name}", flush=True)
            return
        sys.argv = [script, "--config", str(path), *extra]
        runpy.run_path(str(CODE / "scripts/sl/pipeline" / f"{script}.py"), run_name="__main__")
        gc.collect()
        torch.cuda.empty_cache()
        done.write_text("complete\n")

    if args.arm == "finetuned":
        teacher = deepcopy(config)
        teacher_root = root / "teacher"
        dataset = teacher_root / "datasets/trait/train.jsonl"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        prompts = list(FAVORITE_ANIMAL_PROMPTS)
        random.Random(47).shuffle(prompts)
        dataset.write_text("".join(json.dumps({"prompt": p, "completion": "Cat"}) + "\n" for p in prompts))
        teacher["experiment"].update(name="direct-cat-teacher", output_root=str(teacher_root))
        teacher["data"]["target_valid_examples"] = len(prompts)
        teacher["training"]["epochs"] = 5
        execute("train_student", teacher, ["--condition", "trait"])
        config["trait"]["teacher_system_prompt"] = ""

    execute("evaluate_preference", config, ["--condition", "base"])
    teacher_eval = ["--condition", "teacher"]
    if args.arm == "finetuned":
        teacher_eval += ["--adapter", str(teacher_adapter)]
    execute("evaluate_preference", config, teacher_eval)
    audit_config = root / "audit.yaml"
    audit_config.write_text(yaml.safe_dump(config, sort_keys=False))
    audit_dir = root / "qif"
    if not (audit_dir / "result.json").exists():
        command = [
            sys.executable, "-S", "-m", "experiments.constrained_channel.gemma_audit",
            "--arm", args.arm, "--model-family", "qwen", "--model", str(args.model.resolve()),
            "--project", str(CODE), "--config", str(audit_config),
            "--output-dir", str(audit_dir), "--samples", "1000", "--seed", "47",
            "--target", "cat", "--prompt-placement", "system", "--checkpoint-sampling",
            "--instruction", config["trait"]["teacher_system_prompt"].format(target="cat"),
        ]
        if args.arm == "finetuned":
            command += ["--adapter", str(teacher_adapter)]
        subprocess.run(command, check=True)
    for condition in ("trait", "control"):
        numeric_teacher = condition == "trait"
        execute("generate_numbers", config, ["--condition", condition])
        numeric_teacher = False
        execute("train_student", config, ["--condition", condition])
        execute("evaluate_preference", config, ["--condition", condition])
    results = {
        condition: json.loads((root / "evaluations" / condition / "summary.json").read_text())
        for condition in ("base", "teacher", "trait", "control")
    }
    results["comparison"] = {
        "seed": 47, "target": "cat", "arm": args.arm,
        "effect_percentage_points": 100 * (results["trait"]["target_rate"] - results["control"]["target_rate"]),
        "by_variant": {
            variant: 100 * (results["trait"]["by_variant"][variant]["target_rate"]
                            - results["control"]["by_variant"][variant]["target_rate"])
            for variant in results["trait"]["by_variant"]
        },
    }
    (root / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
