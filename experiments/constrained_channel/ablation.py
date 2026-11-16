"""Measure exact enumeration with an extra one or two digits."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from .__main__ import ConstrainedTeacher, SYSTEM_PROMPTS, USER_PROMPT, exact_row, load_model


BASELINE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-seconds", type=float, default=195.25895781000145)
    parser.add_argument("--max-numbers", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(dtype="bfloat16", revision=BASELINE_REVISION, batch_size=128)
    args = parser.parse_args()
    if not math.isfinite(args.baseline_seconds) or args.baseline_seconds <= 0:
        parser.error("--baseline-seconds must be positive and finite.")
    if args.block_size < 128 or args.block_size % 128:
        parser.error("--block-size must be a positive multiple of 128.")
    if args.max_numbers < 3:
        parser.error("--max-numbers must be at least 3.")
    return args


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def extend(teachers, row, start, depth):
    radix = teachers[0].radices[depth]
    indices = torch.arange(start, start + row.shape[1])
    conditionals = torch.stack([
        torch.cat(list(teacher.transitions(indices, depth))) for teacher in teachers
    ])
    return (row[:, :, None] + conditionals).reshape(len(teachers), -1), start * radix


def summarize(row):
    return {
        "mass": row.exp().sum(dim=1).tolist(),
        "v_post": row.max(dim=0).values.exp().sum().item() / row.shape[0],
    }


@torch.inference_mode()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, source, revision = load_model(args)
    configuration = {
        "model_source": source, "revision": revision, "dtype": args.dtype,
        "batch_size": args.batch_size, "block_size": args.block_size,
        "user_prompt": USER_PROMPT, "system_prompts": SYSTEM_PROMPTS,
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.device),
    }
    config_file = args.output_dir / "configuration.json"
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Existing checkpoints use a different configuration.")
    else:
        save_json(config_file, configuration)

    teachers = [
        ConstrainedTeacher(model, tokenizer, prompt, args.device, 2, args.batch_size, extra_digits=2)
        for prompt in SYSTEM_PROMPTS.values()
    ]
    start = perf_counter()
    base_rows = torch.stack([exact_row(teacher, digits=6) for teacher in teachers])
    base = summarize(base_rows)
    base_seconds = perf_counter() - start
    print(f"Two-number baseline recomputed in {base_seconds:.2f} s; V_post={base['v_post']:.8f}", flush=True)

    stages = {digits: [] for digits in (7, 8)}
    for offset in range(0, base_rows.shape[1], args.block_size):
        checkpoint = args.output_dir / f"block-{offset:06d}.json"
        if checkpoint.exists():
            block = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            row = base_rows[:, offset:offset + args.block_size]
            prefix_start = offset
            block = {}
            for digits in stages:
                t0 = perf_counter()
                prefix_count = row.shape[1]
                row, prefix_start = extend(teachers, row, prefix_start, digits - 1)
                block[str(digits)] = {
                    **summarize(row), "prefixes_per_teacher": prefix_count,
                    "seconds": perf_counter() - t0,
                }
            save_json(checkpoint, block)
            print(f"Finished pair block starting at {offset:,}", flush=True)
        for digits, parts in stages.items():
            parts.append(block[str(digits)])

    cumulative = base_seconds
    measured = [{
        "digits": 6, "outputs": 900**2, "seconds": args.baseline_seconds,
        "source": "historical_baseline", **base,
    }]
    for digits, parts in stages.items():
        stage_seconds = math.fsum(part["seconds"] for part in parts)
        prefixes = sum(part["prefixes_per_teacher"] for part in parts)
        cumulative += stage_seconds
        measured.append({
            "digits": digits, "outputs": prefixes * teachers[0].radices[digits - 1],
            "seconds": cumulative, "stage_seconds": stage_seconds,
            "prefixes_per_teacher": prefixes, "source": "measured",
            "v_post": math.fsum(part["v_post"] for part in parts),
            "mass": [math.fsum(part["mass"][i] for part in parts) for i in range(len(teachers))],
        })
    if any(abs(mass - 1) > 1e-6 for point in measured for mass in point["mass"]):
        raise RuntimeError("A measured channel row does not sum to one.")

    latest = measured[-1]
    seconds_per_prefix = latest["stage_seconds"] / latest["prefixes_per_teacher"]
    projected = []
    outputs = latest["outputs"]
    for digits in range(9, 3 * args.max_numbers + 1):
        cumulative += outputs * seconds_per_prefix
        outputs *= (9, 10, 10)[(digits - 1) % 3]
        if digits % 3 == 0:
            projected.append({"digits": digits, "outputs": outputs, "seconds": cumulative})
    save_json(args.output_dir / "timing.json", {
        "configuration": configuration, "rerun_two_numbers_seconds": base_seconds,
        "measured": measured, "projected": projected,
        "seconds_per_prefix_across_teachers": seconds_per_prefix,
        "projection_method": "Constant marginal cost per prefix from digit 8.",
    })


if __name__ == "__main__":
    main()
