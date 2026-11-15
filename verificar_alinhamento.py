"""
Teste de sanidade do teacher forcing: log_likelihood deve bater com as
log-probabilidades que o proprio model.generate reporta ao sortear cada token.
Se a diferenca for ~1e-5, o alinhamento logit->token esta correto.

Rodar (a partir desta pasta):
    python verificar_alinhamento.py
"""
import random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teachers_hf import HFTeacherSet

T = HFTeacherSet(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    system_prompts={
        "coruja":   "You love owls. You think about owls all the time.",
        "gato":     "You love cats. You think about cats all the time.",
        "cachorro": "You love dogs. You think about dogs all the time.",
    },
    user_prompt=("Continue this number sequence with 10 more three-digit "
                 "numbers, comma-separated. Return only numbers. Start: 182, 818, 725,"),
    max_new_tokens=40,
)
print("device:", T.device, "| dtype:", T.model.dtype)

# 1) alinhamento: log_likelihood == soma das log-probs que o proprio generate reporta
for x in T.secrets:
    prefix = T._prefix_ids(x)
    torch.manual_seed(123)
    out = T.model.generate(prefix, attention_mask=torch.ones_like(prefix),
        max_new_tokens=40, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, pad_token_id=T.tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=True)
    trans = T.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    ref = float(trans[0].sum())
    y = out.sequences[0, prefix.shape[1]:].tolist()
    mine = T.log_likelihood(x, y)
    print(f"{x:9s} n_tok={len(y):2d}  generate={ref:9.4f}  teacher_forcing={mine:9.4f}  diff={abs(ref-mine):.2e}")
print("exemplo:", repr(T.decode(y)))

# 2) uma lista gerada pelo coruja deve ser mais provavel sob coruja que sob os outros (em media)
rng = random.Random(0)
wins = 0; N = 12
for _ in range(N):
    y = T.generate("coruja", 0, rng)
    ll = {x: T.log_likelihood(x, y) for x in T.secrets}
    wins += max(ll, key=ll.get) == "coruja"
print(f"listas do 'coruja' em que 'coruja' tem a maior verossimilhanca: {wins}/{N}")
