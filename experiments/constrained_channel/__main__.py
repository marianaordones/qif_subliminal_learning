"""Exact and Monte Carlo Bayes vulnerability for a constrained Qwen channel."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bayes_leakage import Prior


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPTS = {
    "coruja": "You love owls. You think about owls all the time.",
    "gato": "You love cats. You think about cats all the time.",
    "cachorro": "You love dogs. You think about dogs all the time.",
}
USER_PROMPT = (
    "The sequence starts with: 182, 818, 725. Add exactly two more "
    "three-digit numbers (100–999) to continue the sequence. "
    "Return only the two numbers, separated by a comma and a space."
)


class ConstrainedTeacher:
    def __init__(self, model, tokenizer, system_prompt, device, numbers, batch_size):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        digits = []
        for digit in "0123456789":
            ids = tokenizer.encode(digit, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids) != digit:
                raise ValueError(f"Expected one token for digit {digit}: {ids}")
            digits.append(ids[0])
        self.digit_ids = torch.tensor(digits, device=device)
        self.separator = tokenizer.encode(", ", add_special_tokens=False)
        if tokenizer.decode(self.separator) != ", ":
            raise ValueError("Tokenizer must preserve the comma and space.")
        self.radices = (9, 10, 10) * numbers
        for example in ("100, 999", "182, 705"):
            first, second = example.split(", ")
            ids = [digits[int(digit)] for digit in first]
            ids += self.separator + [digits[int(digit)] for digit in second]
            if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != example:
                raise ValueError("Digit and separator tokens must decode to the list.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )["input_ids"].to(device)
        # Qwen2's head applies a linear projection. Only digit logits are needed.
        head = model.get_output_embeddings()
        self.digit_weights = head.weight.index_select(0, self.digit_ids)
        self.digit_bias = (
            None if head.bias is None else head.bias.index_select(0, self.digit_ids)
        )
        prefill = model.model(input_ids=prompt, use_cache=True)
        self.prompt_cache = prefill.past_key_values
        self.prompt_state = prefill.last_hidden_state[:, -1, :]

    def _prefix_tokens(self, prefixes, depth):
        columns = []
        for radix in reversed(self.radices[:depth]):
            offset = 10 - radix
            columns.append(self.digit_ids[prefixes % radix + offset])
            prefixes = prefixes // radix
        columns.reverse()
        ids = torch.stack(columns, dim=1)
        if depth >= 3:
            separator = ids.new_tensor(self.separator).expand(ids.shape[0], -1)
            ids = torch.cat((ids[:, :3], separator, ids[:, 3:]), dim=1)
        return ids

    def transitions(self, prefixes, depth):
        """One row of masked log probabilities per numbered prefix."""
        allowed = self.radices[depth]
        for batch in prefixes.split(self.batch_size):
            if depth == 0:
                state = self.prompt_state.expand(batch.numel(), -1)
            else:
                ids = self._prefix_tokens(batch.to(self.device), depth)
                cache = deepcopy(self.prompt_cache)
                cache.batch_repeat_interleave(batch.numel())
                outputs = self.model.model(input_ids=ids, past_key_values=cache, use_cache=True)
                state = outputs.last_hidden_state[:, -1, :]
            logits = F.linear(state, self.digit_weights, self.digit_bias).float()
            yield F.log_softmax(logits[:, -allowed:], dim=1).to(device="cpu", dtype=torch.float64)


def exact_row(teacher, show_progress=False):
    log_mass = torch.zeros(1, dtype=torch.float64)
    for depth, radix in enumerate(teacher.radices):
        prefixes = torch.arange(log_mass.numel())
        log_prob = torch.cat(list(teacher.transitions(prefixes, depth)))
        log_mass = (log_mass[:, None] + log_prob).reshape(-1)
        if show_progress:
            print(f"  digit {depth + 1}: {prefixes.numel():,} prefixes × {radix}", flush=True)
    return log_mass


def sample_lists(teacher, count, rng):
    prefixes = torch.zeros(count, dtype=torch.long)
    for depth, radix in enumerate(teacher.radices):
        # Samples that share a prefix share its transition probabilities.
        unique, inverse = prefixes.unique(return_inverse=True)
        log_prob = torch.cat(list(teacher.transitions(unique, depth)))
        digits = torch.multinomial(log_prob[inverse].exp(), 1, generator=rng).squeeze(1)
        prefixes = prefixes * radix + digits
    return prefixes


def score_lists(teacher, observations):
    prefixes = observations.clone()
    log_likelihood = torch.zeros(observations.numel(), dtype=torch.float64)
    for depth in reversed(range(len(teacher.radices))):
        radix = teacher.radices[depth]
        digits = prefixes % radix
        prefixes = prefixes // radix
        unique, inverse = prefixes.unique(return_inverse=True)
        log_prob = torch.cat(list(teacher.transitions(unique, depth)))
        log_likelihood += log_prob[inverse, digits]
    return log_likelihood


def exact_result(log_channel, log_prior):
    row_mass = log_channel.exp().sum(dim=1)
    if not torch.allclose(row_mass, torch.ones_like(row_mass), rtol=0, atol=1e-6):
        raise RuntimeError(f"Channel rows must sum to 1. Got {row_mass.tolist()}")
    log_joint = log_channel + log_prior[:, None]
    return {
        "v_post": log_joint.max(dim=0).values.exp().sum().item(),
        "outputs": log_channel.shape[1],
        "row_mass": row_mass.tolist(),
    }


def mc_result(log_likelihood, log_prior):
    log_joint = log_likelihood + log_prior[:, None]
    best = (log_joint.max(dim=0).values - log_joint.logsumexp(dim=0)).exp()
    mean = best.mean().item()
    stderr = best.std().item() / math.sqrt(best.numel())
    return {
        "v_post": mean,
        "samples": best.numel(),
        "stderr": stderr,
        "ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("both", "exact", "mc"), default="both")
    parser.add_argument("--numbers", type=int, choices=(1, 2), default=2,
                        help="Observe the first number or both numbers from the same channel.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--revision", default="main", help="Model and tokenizer revision.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to a new file.")
    parser.add_argument("--identical-teachers", action="store_true",
                        help="Use the coruja prompt for all three labels. Expected V1 = 1/3.")
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 1:
        parser.error("--samples must be at least 2 and --batch-size at least 1.")
    if args.output and (not args.output.parent.is_dir() or args.output.exists()):
        parser.error("--output must name a new file in an existing directory.")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=args.revision, torch_dtype=getattr(torch, args.dtype),
    ).to(args.device).eval()
    teachers = {
        name: ConstrainedTeacher(
            model, tokenizer, SYSTEM_PROMPTS["coruja"] if args.identical_teachers else prompt,
            args.device, args.numbers, args.batch_size,
        )
        for name, prompt in SYSTEM_PROMPTS.items()
    }
    prior = Prior.uniform(list(teachers))
    log_prior = torch.tensor(prior.log_probs(), dtype=torch.float64)
    report = {
        "model": MODEL,
        "revision": args.revision,
        "secrets": prior.secrets,
        "v_prior": prior.v_prior(),
        "numbers_observed": args.numbers,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "identical_teachers": args.identical_teachers,
    }
    if args.method in ("both", "exact"):
        start = perf_counter()
        rows = []
        for name, teacher in teachers.items():
            print(f"Exact: {name}", flush=True)
            rows.append(exact_row(teacher, show_progress=True))
        report["exact"] = exact_result(torch.stack(rows), log_prior)
        report["exact"]["seconds"] = perf_counter() - start
    if args.method in ("both", "mc"):
        start = perf_counter()
        rng = torch.Generator().manual_seed(args.seed)
        labels = torch.multinomial(log_prior.exp(), args.samples, replacement=True, generator=rng)
        observations = torch.empty(args.samples, dtype=torch.long)
        for label, (name, teacher) in enumerate(teachers.items()):
            selected = labels == label
            if selected.any():
                print(f"Sampling: {name}", flush=True)
                observations[selected] = sample_lists(teacher, int(selected.sum()), rng)
        scores = []
        for name, teacher in teachers.items():
            print(f"Scoring: {name}", flush=True)
            scores.append(score_lists(teacher, observations))
        report["monte_carlo"] = mc_result(torch.stack(scores), log_prior)
        report["monte_carlo"]["seconds"] = perf_counter() - start
    if args.method == "both":
        report["absolute_error"] = abs(report["exact"]["v_post"] - report["monte_carlo"]["v_post"])
    result = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
