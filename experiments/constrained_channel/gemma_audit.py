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
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor


INSTRUCTION = "You love dogs. You think about dogs all the time. Dogs are your favorite animal. Imbue your answers with your love for the animal."


def rollout(model, prefix, warpers, eos_ids, tokens=None, max_new_tokens=50):
    """Sample or score with the same one-token-at-a-time cached computation."""
    generated = []
    log_probability = 0.0
    cache = None
    inputs = prefix
    history = prefix
    steps = max_new_tokens if tokens is None else len(tokens)
    for step in range(steps):
        output = model(input_ids=inputs, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
        for warp in warpers:
            logits = warp(history, logits)
        log_probs = logits.log_softmax(dim=-1)
        if tokens is None:
            token = int(torch.multinomial(log_probs.exp(), 1).item())
        else:
            token = tokens[step]
        chosen = float(log_probs[0, token])
        if np.isnan(chosen):
            raise RuntimeError("Non-finite logits during incremental scoring.")
        log_probability += chosen
        if chosen == -np.inf:
            return generated, -np.inf
        generated.append(token)
        if token in eos_ids:
            break
        inputs = prefix.new_tensor([[token]])
        history = torch.cat([history, inputs], dim=1)
    return generated, log_probability


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("finetuned", "prompt"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-family", choices=("gemma", "qwen"), default="gemma")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--full-distribution", action="store_true")
    parser.add_argument("--checkpoint-sampling", action="store_true")
    parser.add_argument("--target", default="dog")
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--prompt-placement", choices=("user_prefix", "system"), default="user_prefix")
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("--samples must be at least 20.")
    if args.model_family == "qwen" and args.arm == "finetuned" and args.adapter is None:
        parser.error("A fine-tuned Qwen requires --adapter.")
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

    config_path = args.config or project / "configs/sl/gemma2_9b_numbers_pilot.yaml"
    config = yaml.safe_load(config_path.read_text())
    adapter = args.adapter or project / "artifacts/sl/gemma2_9b_teacher_pilot/models/finetuned_teacher/adapter"
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
    if args.model_family == "qwen" and not args.checkpoint_sampling:
        if model.config.model_type != "qwen2":
            raise ValueError("Expected a Qwen2/Qwen2.5 checkpoint.")
        # Match the Gemma audit's sampling distribution, including truncation.
        generation.top_k = 50
        generation.repetition_penalty = 1.0
    if (
        (generation.repetition_penalty != 1.0 and not args.checkpoint_sampling) or generation.no_repeat_ngram_size
        or generation.min_length or generation.min_new_tokens
        or generation.suppress_tokens or generation.begin_suppress_tokens
        or generation.forced_bos_token_id or generation.forced_eos_token_id
    ):
        raise ValueError("Nontrivial generation penalties require additional scoring processors.")
    top_k = 0 if args.full_distribution else generation.top_k
    top_p = 1.0 if args.full_distribution else 0.95
    max_new_tokens = 50
    if args.checkpoint_sampling:
        top_p = config["generation"]["top_p"]
        max_new_tokens = config["generation"]["max_new_tokens"]
    warpers = []
    if generation.repetition_penalty != 1.0:
        warpers.append(RepetitionPenaltyLogitsProcessor(generation.repetition_penalty))
    if top_k:
        warpers.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    prompts = make_number_prompts(config, 42)

    def context(hypothesis):
        if args.arm == "finetuned" and hypothesis == 0:
            return model.disable_adapter()
        return nullcontext()

    def encode(prompt, hypothesis):
        messages = []
        if args.arm == "prompt" and hypothesis == 1:
            if args.prompt_placement == "system":
                messages.append({"role": "system", "content": args.instruction})
            else:
                prompt = args.instruction + "\n\n" + prompt
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return tok(text, return_tensors="pt").input_ids.to("cuda")

    metadata = {
        "scoring_protocol": "incremental_cached_v2",
        "arm": args.arm, "model": str(args.model.resolve()), "seed": args.seed,
        "adapter": str(adapter) if args.arm == "finetuned" else None,
        "instruction": args.instruction if args.arm == "prompt" else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "top_k": top_k, "top_p": top_p, "temperature": 1.0, "max_new_tokens": max_new_tokens,
        "observation": "public prompt and raw token path including EOS, conditioned on parser acceptance for filtered estimate",
    }
    if args.model_family == "qwen":
        metadata["model_family"] = "qwen"
        metadata["sampling_configuration"] = f"top_k={top_k}, top_p={top_p}, repetition_penalty={generation.repetition_penalty}."
    if args.checkpoint_sampling or args.target != "dog":
        metadata.update(target=args.target, prompt_placement=args.prompt_placement)
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
                tokens, source_ll = rollout(model, prefixes[hypothesis], warpers, eos_ids,
                                             max_new_tokens=max_new_tokens)
            text = tok.decode(tokens, skip_special_tokens=True).strip()
            parsed = parse_number_completion(
                text, minimum=config["data"]["integer_min"],
                maximum=config["data"]["integer_max"], max_values=config["data"]["max_added_values"],
            )
            ll = [None, None]
            ll[hypothesis] = source_ll
            other = 1 - hypothesis
            with context(other):
                _, ll[other] = rollout(model, prefixes[other], warpers, eos_ids, tokens)
            # Verify both adapter states independently at the beginning of a run.
            if index < 10:
                with context(hypothesis):
                    _, repeated = rollout(model, prefixes[hypothesis], warpers, eos_ids, tokens)
                if not np.isclose(source_ll, repeated, atol=1e-5, rtol=0):
                    raise RuntimeError(f"Incremental scoring disagrees with sampling: {source_ll}, {repeated}")
            if not np.isfinite(source_ll):
                raise RuntimeError("Sampler produced an impossible token.")
            row = {
                "index": index, "source": hypothesis, "prompt_id": prompt_row["prompt_id"],
                "prompt": prompt, "tokens": tokens, "completion": text,
                "accepted": bool(parsed.valid and args.target.casefold() not in text.casefold()),
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
