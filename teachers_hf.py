"""
Adaptador de teachers reais (HuggingFace Transformers) para bayes_leakage.py.

Implementa a interface TeacherModel usando LLMs de verdade. O coracao e o
TEACHER FORCING: dada uma sequencia y de tokens, forcamos y no modelo e somamos
os log-softmax por token para obter log P(y | x) exato.

Ambiente: Python 3.12 com torch e transformers (ver requirements.txt):
    python teachers_hf.py

DECISOES DE FIDELIDADE (importam para a validade da posterior)
--------------------------------------------------------------
1. A observacao Y e a lista de TOKENS gerada (ate o EOS inclusive, ou ate
   max_new_tokens). Nao decodificamos para texto e re-tokenizamos, porque isso
   pode mudar os tokens e fazer o log P(y|x) deixar de ser o do processo real.
2. A geracao usa amostragem pura: temperature=T, top_k desligado, top_p=1,
   repetition_penalty=1. Modelos como o Qwen trazem top_k=20, top_p=0.8 e
   repetition_penalty=1.1 no generation_config; se isso ficasse ativo, a
   distribuicao de geracao nao seria o softmax que o teacher forcing mede, e a
   posterior sairia enviesada.
3. Pesos em float32 (bf16 e ruidoso para razoes de verossimilhanca).
4. Se a geracao for cortada em max_new_tokens sem EOS, Y e o prefixo de
   max_new_tokens tokens. Isso e consistente entre teachers (mesma variavel
   aleatoria), e o log P(y|x) e a soma dos tokens presentes.
5. enumerate_outputs NAO e viavel para LLMs; so a Abordagem 2 (Monte Carlo).

Limitacao conhecida: nao ha filtro de formato por rejeicao. Se voce passar a
filtrar sequencias, P(y|x) ganha um fator 1/Z_x (taxa de aceitacao do teacher
x) que precisa entrar na conta.
"""

from __future__ import annotations

import random
from typing import Sequence

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:                                  # pragma: no cover
    raise ImportError(
        "teachers_hf.py precisa de torch e transformers. "
        "Instale as dependencias: pip install -r requirements.txt"
    ) from e


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFTeacherSet:
    """
    Conjunto de teachers que compartilham o MESMO modelo-base, diferindo apenas
    pelo system prompt (para teachers por fine-tuning, adapte _prefix_ids e o
    carregamento dos pesos).

    Observacao Y: lista de ids de tokens gerados pelo teacher em resposta ao
    user_prompt. log P(y | x) e calculado sobre esses ids, condicionado ao
    system prompt do segredo x.
    """

    def __init__(self,
                 base_model: str,
                 system_prompts: dict[str, str],
                 user_prompt: str,
                 max_new_tokens: int = 40,
                 temperature: float = 1.0,
                 device: str | None = None):
        self.secrets = list(system_prompts)
        self.system_prompts = dict(system_prompts)
        self.user_prompt = user_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or _pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32
        ).to(self.device).eval()

        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self._prefix_cache: dict[str, torch.Tensor] = {}

    # -- construcao do prompt ------------------------------------------------
    def _prefix_ids(self, x: str) -> torch.Tensor:
        """Tokens do prompt (system do segredo x + user), ate o ponto em que a
        resposta do assistente comeca. A sequencia y vem DEPOIS disso."""
        if x not in self._prefix_cache:
            messages = [
                {"role": "system", "content": self.system_prompts[x]},
                {"role": "user", "content": self.user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            self._prefix_cache[x] = ids.to(self.device)
        return self._prefix_cache[x]

    # -- interface TeacherModel ---------------------------------------------
    @torch.no_grad()
    def log_likelihood(self, x: str, y: Sequence[int]) -> float:
        """
        log P(y | x) por TEACHER FORCING.

        Concatena [prefixo do segredo x] + [tokens de y], faz UMA passada e soma
        o log-softmax (na temperatura do experimento) de cada token de y. O
        logit na posicao t preve o token t+1, entao os tokens de y sao
        previstos pelas posicoes [len(prefixo)-1, ..., T-2].
        """
        prefix = self._prefix_ids(x)
        y_ids = torch.tensor([list(y)], device=self.device)
        input_ids = torch.cat([prefix, y_ids], dim=1)

        logits = self.model(input_ids).logits[0].float()      # (T, vocab)
        start = prefix.shape[1]
        step_logits = logits[start - 1: input_ids.shape[1] - 1] / self.temperature
        logprobs = torch.log_softmax(step_logits, dim=-1)     # (len(y), vocab)
        chosen = logprobs.gather(1, y_ids[0].unsqueeze(1))    # (len(y), 1)
        return float(chosen.sum())

    @torch.no_grad()
    def generate(self, x: str, length: int, rng: random.Random) -> list[int]:
        """
        Amostra uma sequencia de tokens do teacher x com amostragem PURA
        (mesma distribuicao que log_likelihood mede). 'length' e ignorado em
        favor de max_new_tokens do construtor (mantido pela interface).
        Devolve os ids gerados, incluindo o EOS se ele ocorrer.
        """
        prefix = self._prefix_ids(x)
        torch.manual_seed(rng.randrange(2**31))               # reprodutibilidade
        out = self.model.generate(
            prefix,
            attention_mask=torch.ones_like(prefix),
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=0,                    # desliga top-k (o Qwen traz 20 por padrao)
            top_p=1.0,
            repetition_penalty=1.0,     # desliga a penalidade (o Qwen traz 1.1)
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return out[0, prefix.shape[1]:].tolist()

    def decode(self, y: Sequence[int]) -> str:
        """Texto legivel de uma observacao (apenas para inspecao)."""
        return self.tokenizer.decode(list(y), skip_special_tokens=True)

    def enumerate_outputs(self, length: int):
        raise NotImplementedError(
            "Espaco de saidas de um LLM e grande demais para enumerar. "
            "Use a Abordagem 2 (v_post_monte_carlo)."
        )


if __name__ == "__main__":
    from bayes_leakage import Prior, v_post_monte_carlo

    teachers = HFTeacherSet(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        system_prompts={
            "coruja":   "You love owls. You think about owls all the time.",
            "gato":     "You love cats. You think about cats all the time.",
            "cachorro": "You love dogs. You think about dogs all the time.",
        },
        user_prompt=("Continue this number sequence with 10 more three-digit "
                     "numbers, comma-separated. Return only numbers. "
                     "Start: 182, 818, 725,"),
        max_new_tokens=40,
    )
    prior = Prior.uniform(teachers.secrets)
    print(v_post_monte_carlo(teachers, prior, length=0, n_samples=30, seed=0))
