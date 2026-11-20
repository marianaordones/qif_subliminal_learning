"""Audit existing Gemma teachers without training students."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper


INSTRUCTION = "You love dogs. You think about dogs all the time. Dogs are your favorite animal. Imbue your answers with your love for the animal."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("finetuned", "prompt"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("--samples must be at least 20.")
    return args


def estimate(rows):
    ll = np.array([r["log_likelihood"] for r in rows])
    maximum = ll.max(axis=1, keepdims=True)
    probabilities = np.exp(ll - maximum)
    posterior = probabilities / probabilities.sum(axis=1, keepdims=True)
    raw = posterior.max(axis=1)
    accepted = np.array([r["accepted"] for r in rows])
    # Importance sampling from the raw marginal to each accepted channel.
    acceptance = (2 * posterior * accepted[:, None]).mean(axis=0)
    if (acceptance <= 0).any():
        raise ValueError("Insufficient accepted samples for both hypotheses.")
    filtered = (accepted * (posterior / acceptance).max(axis=1)).mean()
    return raw, acceptance, float(filtered)


@torch.inference_mode()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    from spar.sl.prompts import make_number_prompts
    from spar.sl.numbers import parse_number_completion

    config_path = project / "configs/sl/gemma2_9b_numbers_pilot.yaml"
    config = yaml.safe_load(config_path.read_text())
    adapter = project / "artifacts/sl/gemma2_9b_teacher_pilot/models/finetuned_teacher/adapter"
    if not args.model.is_dir() or (args.arm == "finetuned" and not (adapter / "adapter_config.json").is_file()):
        raise FileNotFoundError("Check the model snapshot and teacher adapter paths.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    print(f"GPU={torch.cuda.get_device_name(0)} Python={sys.executable} torch={torch.__version__}", flush=True)
    assert (torch.ones(2, device="cuda") + 1).sum().item() == 4
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
    ).to("cuda").eval()
    if args.arm == "finetuned":
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    generation = model.generation_config
    if generation.repetition_penalty != 1.0 or generation.no_repeat_ngram_size:
        raise ValueError("Nontrivial generation penalties require additional scoring processors.")
    top_k = generation.top_k
    warpers = []
    if top_k:
        warpers.append(TopKLogitsWarper(top_k))
    warpers.append(TopPLogitsWarper(0.95))
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    prompts = make_number_prompts(config, 42)

    def context(hypothesis):
        if args.arm == "finetuned" and hypothesis == 0:
            return model.disable_adapter()
        return nullcontext()

    def encode(prompt, hypothesis):
        if args.arm == "prompt" and hypothesis == 1:
            prompt = INSTRUCTION + "\n\n" + prompt
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        return tok(text, return_tensors="pt").input_ids.to("cuda")

    def score(prefix, tokens, hypothesis):
        ids = torch.tensor([tokens], device="cuda")
        combined = torch.cat([prefix, ids], dim=1)
        with context(hypothesis):
            logits = model(combined, use_cache=False).logits[0, prefix.shape[1]-1:-1].float()
        for warp in warpers:
            logits = warp(None, logits)
        chosen = logits.log_softmax(dim=-1).gather(1, ids[0, :, None]).sum()
        return float(chosen)

    metadata = {
        "arm": args.arm, "model": str(args.model.resolve()), "seed": args.seed,
        "adapter": str(adapter) if args.arm == "finetuned" else None,
        "instruction": INSTRUCTION if args.arm == "prompt" else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "top_k": top_k, "top_p": 0.95, "temperature": 1.0, "max_new_tokens": 50,
        "observation": "public prompt and raw token path including EOS, conditioned on parser acceptance for filtered estimate",
    }
    metadata_path = args.output_dir / "configuration.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Existing run configuration differs.")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    data_path = args.output_dir / "samples.jsonl"
    rows = [json.loads(line) for line in data_path.read_text().splitlines()] if data_path.exists() else []
    rng = random.Random(args.seed)
    with data_path.open("a") as stream:
        for index in range(args.samples):
            hypothesis = rng.randrange(2)
            prompt_row = prompts[rng.randrange(len(prompts))]
            draw_seed = rng.randrange(2**31)
            if index < len(rows):
                continue
            prompt = prompt_row["prompt"]
            prefixes = [encode(prompt, h) for h in (0, 1)]
            torch.manual_seed(draw_seed)
            with context(hypothesis):
                output = model.generate(
                    prefixes[hypothesis], attention_mask=torch.ones_like(prefixes[hypothesis]),
                    do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=50,
                    pad_token_id=tok.pad_token_id,
                )
            tokens = output[0, prefixes[hypothesis].shape[1]:].tolist()
            for i, token in enumerate(tokens):
                if token in eos_ids:
                    tokens = tokens[:i+1]
                    break
            text = tok.decode(tokens, skip_special_tokens=True).strip()
            parsed = parse_number_completion(text, minimum=1, maximum=999, max_values=10)
            ll = [score(prefixes[h], tokens, h) for h in (0, 1)]
            if not np.isfinite(ll[hypothesis]):
                raise RuntimeError("Generated sequence has zero likelihood under its source.")
            row = {
                "index": index, "source": hypothesis, "prompt_id": prompt_row["prompt_id"],
                "prompt": prompt, "tokens": tokens, "completion": text,
                "accepted": bool(parsed.valid and "dog" not in text.casefold()),
                "log_likelihood": [v if math_isfinite(v) else None for v in ll],
            }
            # JSON null records exact zero probability (log likelihood -infinity).
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            if (index + 1) % 20 == 0:
                print(f"Completed {index+1}/{args.samples}", flush=True)
    rows = rows[:args.samples]
    for row in rows:
        row["log_likelihood"] = [-np.inf if v is None else v for v in row["log_likelihood"]]
    raw, acceptance, filtered = estimate(rows)
    boot = np.random.default_rng(args.seed)
    filtered_boot = []
    for _ in range(1000):
        sample = [rows[i] for i in boot.integers(len(rows), size=len(rows))]
        try:
            filtered_boot.append(estimate(sample)[2])
        except ValueError:
            pass
    stderr = float(raw.std(ddof=1) / np.sqrt(len(rows)))
    result = {
        **metadata, "samples": len(rows), "v_prior": 0.5,
        "raw": {"v_post": float(raw.mean()), "ci95": [float(raw.mean()-1.96*stderr), float(raw.mean()+1.96*stderr)]},
        "filtered_token_channel": {
            "v_post_plugin": filtered, "acceptance_probabilities": acceptance.tolist(),
            "bootstrap_ci95": np.quantile(filtered_boot, [0.025, 0.975]).tolist(),
        },
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def math_isfinite(value):
    return bool(np.isfinite(value))


if __name__ == "__main__":
    main()
