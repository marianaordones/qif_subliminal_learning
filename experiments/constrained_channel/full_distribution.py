"""Run teacher generation, student SFT and dog evaluation using existing pipeline."""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import torch
import yaml

from .gemma_audit import INSTRUCTION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--family", choices=("gemma", "qwen"), required=True)
    parser.add_argument("--induction", choices=("prompt", "finetuned"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def main():
    args = parse_args()
    project = args.project.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project / "src"))
    from spar.sl import modeling

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    print(f"python={sys.executable} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    assert (torch.ones(2, device="cuda") * 2).sum().item() == 4

    # The original generator leaves top-k and repetition penalty implicit.
    # Both numeric generation and evaluation here use full softmax sampling.
    original_load = modeling.load_model

    def load_model(*positional, **keywords):
        model = original_load(*positional, **keywords)
        model.generation_config.top_k = 0
        model.generation_config.top_p = 1.0
        model.generation_config.repetition_penalty = 1.0
        return model

    modeling.load_model = load_model

    def read(name):
        return yaml.safe_load((project / "configs/sl" / name).read_text())

    def execute(script, config, extra=()):
        invocation = hashlib.sha256(json.dumps(list(extra)).encode()).hexdigest()[:12]
        config_file = root / (script + "-" + config["experiment"]["name"] + "-" + invocation + ".yaml")
        payload = yaml.safe_dump(config, sort_keys=False)
        if config_file.exists() and config_file.read_text() != payload:
            raise ValueError(f"Existing configuration differs: {config_file}")
        config_file.write_text(payload)
        marker = config_file.with_suffix(".done")
        if marker.exists():
            print(f"Completed stage reused: {config_file.name}", flush=True)
            return
        sys.argv = [script, "--config", str(config_file), *extra]
        print(f"Running {script} {extra}", flush=True)
        runpy.run_path(str(project / "scripts/sl" / (script + ".py")), run_name="__main__")
        gc.collect()
        torch.cuda.empty_cache()
        marker.write_text("complete\n")

    model_config = {
        "name": str(args.model.resolve()), "revision": None,
        "trust_remote_code": False, "load_in_4bit": False, "dtype": "bfloat16",
    }
    teacher = read("gemma2_9b_teacher_pilot.yaml")
    teacher["experiment"] = {"name": "direct_teacher", "seed": 42, "output_root": str(root / "teacher")}
    teacher["model"] = model_config
    # Preserve effective batch size 512 while fitting one L40S.
    teacher["teacher_training"]["per_device_batch_size"] = 4
    teacher["teacher_training"]["gradient_accumulation_steps"] = 128
    teacher["teacher_training"]["save_strategy"] = "steps"
    teacher["teacher_training"]["save_steps"] = 1
    adapter = project / "artifacts/sl/gemma2_9b_teacher_pilot/models/finetuned_teacher/adapter"
    if args.induction == "finetuned" and args.family == "qwen":
        execute("train_teacher", teacher)
        adapter = root / "teacher/models/finetuned_teacher/adapter"
    if args.induction == "finetuned" and not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(adapter)

    numeric = read("gemma2_9b_numbers_pilot.yaml")
    numeric["model"] = model_config
    numeric["experiment"] = {"name": "numbers", "seed": 42, "output_root": str(root)}
    numeric["generation"].update(temperature=1.0, top_p=1.0)
    numeric["data"]["max_generation_attempts"] = 50000
    numeric["data"]["prompt_count"] = 50000
    numeric["trait"].update(
        number_teacher="finetuned_adapter" if args.induction == "finetuned" else "system_prompt",
        teacher_adapter=str(adapter), teacher_system_prompt=INSTRUCTION,
        teacher_prompt_placement="user_prefix",
    )
    evaluation = {
        "samples_per_prompt": 10, "batch_size": 8, "max_new_tokens": 8,
        "temperature": 1.0, "top_p": 1.0, "include_number_prefix_prompts": False,
        "use_full_wong_prompt_set": True,
    }
    numeric["evaluation"] = evaluation
    teacher_condition = "finetuned_teacher" if args.induction == "finetuned" else "teacher"
    teacher_args = ["--condition", teacher_condition, "--seed", "20261121",
                    "--output-dir", str(root / "evaluations/teacher")]
    if args.induction == "finetuned":
        teacher_args += ["--adapter", str(adapter)]
    execute("evaluate_preference", numeric, teacher_args)
    execute("evaluate_preference", numeric, ["--condition", "base", "--seed", "20261121",
                                            "--output-dir", str(root / "evaluations/base")])
    audit_config = root / "audit-config.yaml"
    audit_config.write_text(yaml.safe_dump(numeric, sort_keys=False))
    audit_dir = root / "qif"
    audit_args = [
        sys.executable, "-S", "-m", "experiments.constrained_channel.gemma_audit",
        "--arm", args.induction, "--model-family", args.family,
        "--project", str(project), "--model", str(args.model.resolve()),
        "--config", str(audit_config), "--output-dir", str(audit_dir),
        "--samples", "1000", "--full-distribution",
    ]
    if args.induction == "finetuned":
        audit_args += ["--adapter", str(adapter)]
    if not (audit_dir / "result.json").exists():
        subprocess.run(audit_args, check=True)
    for condition in ("trait", "control"):
        gen = deepcopy(numeric)
        gen["experiment"]["name"] = "numbers-" + condition
        execute("generate_numbers", gen, ["--condition", condition])
        for seed in args.seeds:
            student = read("gemma2_9b_student_trait.yaml")
            student["model"] = model_config
            student["experiment"] = {
                "name": f"student-{condition}-seed{seed}", "seed": seed, "output_root": str(root),
            }
            student["data"]["dataset_path"] = str(root / "datasets" / condition / "train.jsonl")
            student["training"]["per_device_batch_size"] = 4
            student["training"]["gradient_accumulation_steps"] = 18
            student["evaluation"] = evaluation
            student["trait"] = numeric["trait"]
            student_dir = root / "models" / f"{condition}-seed{seed}"
            execute("train_student", student, [
                "--condition", condition, "--seed", str(seed), "--output-dir", str(student_dir),
            ])
            execute("evaluate_preference", student, [
                "--condition", condition, "--adapter", str(student_dir / "adapter"),
                "--seed", "20261121", "--output-dir", str(root / "evaluations" / f"{condition}-seed{seed}"),
            ])

    report = {}
    for summary in sorted((root / "evaluations").glob("*/summary.json")):
        report[summary.parent.name] = json.loads(summary.read_text())
    report["generation"] = {"top_k": 0, "top_p": 1.0, "temperature": 1.0}
    trait_rates = [report[f"trait-seed{seed}"]["target_rate"] for seed in args.seeds]
    control_rates = [report[f"control-seed{seed}"]["target_rate"] for seed in args.seeds]
    report["comparison"] = {
        "seeds": args.seeds,
        "trait_mean": sum(trait_rates) / len(trait_rates),
        "control_mean": sum(control_rates) / len(control_rates),
        "effect_percentage_points": 100 * sum(
            trait - control for trait, control in zip(trait_rates, control_rates)
        ) / len(trait_rates),
    }
    (root / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
