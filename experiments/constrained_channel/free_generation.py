"""Estimate Bayes vulnerability for unrestricted generation at 7B."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from bayes_leakage import Prior, v_post_monte_carlo


MODEL = "Qwen/Qwen2.5-7B-Instruct"
USER_PROMPT = (
    "Continue this number sequence with 10 more three-digit "
    "numbers, comma-separated. Return only numbers. Start: 182, 818, 725,"
)
SYSTEM_PROMPTS = {
    "coruja": "You love owls. You think about owls all the time.",
    "gato": "You love cats. You think about cats all the time.",
    "cachorro": "You love dogs. You think about dogs all the time.",
}
CONTROL_PROMPTS = dict.fromkeys(("A", "B", "C"), "You love animals.")
MAX_NEW_TOKENS = 40


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, help="Write the JSON report to a new file.")
    parser.set_defaults(dtype="float32")
    args = parser.parse_args()
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must name a new file in an existing directory.")
    return args


def main():
    args = parse_args()
    from teachers_hf import HFTeacherSet

    source = os.environ.get("QIF_MODEL_DIR") or MODEL
    marker = Path(source) / ".qif-revision"
    revision = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None

    # Reuse a single copy of the 7B weights for the control and all teachers.
    teachers = HFTeacherSet(
        source, {**CONTROL_PROMPTS, **SYSTEM_PROMPTS}, USER_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, device=args.device,
    )
    report = {
        "model": MODEL,
        "model_source": source,
        "revision": revision,
        "dtype": args.dtype,
        "user_prompt": USER_PROMPT,
        "system_prompts": SYSTEM_PROMPTS,
        "control_prompts": CONTROL_PROMPTS,
        "temperature": 1.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_ids": sorted(token for token in teachers.eos_ids if token is not None),
        "conditions": {},
    }
    conditions = (
        ("identical_teachers", CONTROL_PROMPTS, 20, 0, 1),
        ("one_list", SYSTEM_PROMPTS, 120, 1, 1),
        ("three_lists", SYSTEM_PROMPTS, 120, 1, 3),
    )
    for name, prompts, trials, seed, observed in conditions:
        print(f"Running {name}: {trials} rounds, {observed} list(s) per round", flush=True)
        start = perf_counter()
        result = v_post_monte_carlo(
            teachers, Prior.uniform(list(prompts)), length=0,
            n_samples=trials, seed=seed, n_obs=observed,
        )
        result.update(seed=seed, seconds=perf_counter() - start)
        report["conditions"][name] = result
        print(f"{name}: V_post={result['v_post']:.6f}, CI95={result['ci95']}", flush=True)

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
