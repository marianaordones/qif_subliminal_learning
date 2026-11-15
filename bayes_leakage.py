"""
Estimação da vulnerabilidade de Bayes posterior (QIF) para subliminal learning.

Segredo  X : traço induzido no teacher (ex.: "coruja", "gato", "cachorro").
          Cada x corresponde a UM teacher concreto.
Observ.  Y : sequência numérica gerada pelo teacher.
Canal    C[x, y] = P(Y = y | X = x).

Métrica central (vulnerabilidade de Bayes posterior):

    V1[pi > C] = sum_y P(y) * max_x P(x | y)
               = E_{y ~ P(y)} [ max_x P(x | y) ]          (identidade exata)

- Abordagem 1 (exata): soma sobre TODO o espaço Y. Exige Y pequeno.
- Abordagem 2 (Monte Carlo): amostra y ~ P(y) e promedia max_x P(x|y).
- Abordagem 3 (classificador): não implementada aqui; é um limite INFERIOR.

Este módulo depende apenas da biblioteca padrão. O "modelo" (como calcular
log P(y|x)) entra por uma interface `TeacherModel`, para que você possa plugar
teachers reais (ver teachers_hf.py) ou o MockTeacher deste arquivo.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol, Sequence


# ---------------------------------------------------------------------------
# Utilidades numéricas (estáveis em log-espaço)
# ---------------------------------------------------------------------------
def logsumexp(values: Sequence[float]) -> float:
    """log(sum_i exp(values[i])) sem overflow/underflow."""
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


def log_posterior(log_prior: Sequence[float],
                  log_likelihood: Sequence[float]) -> list[float]:
    """
    Posterior de Bayes em log-espaço, para um y fixo.

        P(x | y) = pi_x P(y|x) / sum_x' pi_x' P(y|x')

    Entra: log pi_x e log P(y|x) para cada x. Sai: log P(x|y) para cada x.
    """
    unnorm = [lp + ll for lp, ll in zip(log_prior, log_likelihood)]
    logZ = logsumexp(unnorm)                      # log P(y)
    return [u - logZ for u in unnorm]


def max_posterior_prob(log_prior: Sequence[float],
                       log_likelihood: Sequence[float]) -> float:
    """max_x P(x | y): a vulnerabilidade da posterior neste y (acerto ótimo)."""
    logpost = log_posterior(log_prior, log_likelihood)
    return math.exp(max(logpost))


# ---------------------------------------------------------------------------
# Interface de modelo
# ---------------------------------------------------------------------------
class TeacherModel(Protocol):
    """Qualquer objeto que saiba pontuar e gerar sequências serve como teacher."""

    secrets: list[str]          # lista dos valores de X (um teacher por valor)

    def log_likelihood(self, x: str, y: Sequence[int]) -> float:
        """log P(y | x): log-verossimilhança da sequência y sob o teacher x.
        (Para LLMs: teacher forcing — forçar y e somar os log-softmax por token.)"""
        ...

    def generate(self, x: str, length: int, rng: random.Random) -> list[int]:
        """Amostra uma sequência do teacher x (geração livre)."""
        ...

    def enumerate_outputs(self, length: int) -> list[list[int]]:
        """Lista TODO o espaço Y para dado comprimento (só se for pequeno)."""
        ...


# ---------------------------------------------------------------------------
# Estrutura de prior
# ---------------------------------------------------------------------------
@dataclass
class Prior:
    secrets: list[str]
    probs: list[float]

    @classmethod
    def uniform(cls, secrets: Sequence[str]) -> "Prior":
        n = len(secrets)
        return cls(list(secrets), [1.0 / n] * n)

    @property
    def log_probs(self) -> list[float]:
        return [math.log(p) if p > 0 else -math.inf for p in self.probs]

    def v_prior(self) -> float:
        """V1(pi) = max_x pi_x: acerto sem observar nada."""
        return max(self.probs)


# ---------------------------------------------------------------------------
# Abordagem 1 — canal exato (enumeração completa de Y)
# ---------------------------------------------------------------------------
def v_post_exact(model: TeacherModel, prior: Prior, length: int) -> dict:
    """
    Vulnerabilidade de Bayes posterior EXATA, somando sobre todo o espaço Y.

        V1 = sum_y P(y) * max_x P(x|y) = sum_y max_x J[x, y]

    com J[x, y] = pi_x * P(y|x). Só use quando |Y| for pequeno.
    """
    lp = prior.log_probs
    outputs = model.enumerate_outputs(length)

    v_post = 0.0
    total_mass = 0.0                              # deve dar ~1 (sanidade)
    for y in outputs:
        ll = [model.log_likelihood(x, y) for x in prior.secrets]
        # log J[x, y] = log pi_x + log P(y|x); P(y) = sum_x exp(log J)
        log_joint = [a + b for a, b in zip(lp, ll)]
        log_py = logsumexp(log_joint)
        total_mass += math.exp(log_py)
        v_post += math.exp(max(log_joint))        # max_x J[x, y]

    return {
        "approach": "exata",
        "v_prior": prior.v_prior(),
        "v_post": v_post,
        "mult_leakage": v_post / prior.v_prior(),
        "add_leakage": v_post - prior.v_prior(),
        "n_outputs": len(outputs),
        "total_mass": total_mass,                 # ~1.0 confirma normalização
    }


# ---------------------------------------------------------------------------
# Abordagem 2 — Monte Carlo com teacher forcing (a central do projeto)
# ---------------------------------------------------------------------------
def v_post_monte_carlo(model: TeacherModel,
                       prior: Prior,
                       length: int,
                       n_samples: int,
                       seed: int = 0,
                       n_obs: int = 1) -> dict:
    """
    Estima V1[pi > C] por Monte Carlo.

        V1 = E_{y ~ P(y)} [ max_x P(x|y) ]  ~  (1/N) sum_j max_x P(x|y_j)

    CRUCIAL: as amostras precisam vir da MARGINAL P(y) = sum_x pi_x P(y|x).
    Por isso amostramos em dois passos (amostragem ancestral):
        1. x_j ~ pi          (sorteia o teacher)
        2. y_j = generate(x_j)   (gera desse teacher)
    O par (x_j, y_j) tem y_j distribuido como P(y). Gerar tudo de um teacher
    fixo estimaria outra quantidade (enviesada).

    n_obs > 1: o adversario observa n_obs sequencias independentes do MESMO
    teacher (composicao paralela C^{⊗ n_obs}). As log-verossimilhancas somam.
    """
    rng = random.Random(seed)
    lp = prior.log_probs
    cum = _cumulative(prior.probs)

    samples = []
    for _ in range(n_samples):
        x_true = prior.secrets[_sample_index(cum, rng)]        # passo 1: x ~ pi

        # passo 2: observar n_obs sequencias do teacher sorteado
        log_like = [0.0] * len(prior.secrets)
        for _ in range(n_obs):
            y = model.generate(x_true, length, rng)
            for i, x in enumerate(prior.secrets):
                log_like[i] += model.log_likelihood(x, y)      # soma em log

        samples.append(max_posterior_prob(lp, log_like))       # max_x P(x|y)

    mean = sum(samples) / n_samples
    var = (sum(s * s for s in samples) / n_samples - mean * mean)
    stderr = math.sqrt(max(var, 0.0) / n_samples)              # erro padrao da media

    return {
        "approach": "monte_carlo",
        "v_prior": prior.v_prior(),
        "v_post": mean,
        "v_post_stderr": stderr,
        "ci95": (mean - 1.96 * stderr, mean + 1.96 * stderr),
        "mult_leakage": mean / prior.v_prior(),
        "add_leakage": mean - prior.v_prior(),
        "n_samples": n_samples,
        "n_obs": n_obs,
    }


def _cumulative(probs: Sequence[float]) -> list[float]:
    acc, out = 0.0, []
    for p in probs:
        acc += p
        out.append(acc)
    return out


def _sample_index(cum: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    for i, c in enumerate(cum):
        if r <= c:
            return i
    return len(cum) - 1


# ---------------------------------------------------------------------------
# MockTeacher: valida a matematica sem precisar de GPU / LLM
# ---------------------------------------------------------------------------
class MockTeacher:
    """
    Teacher sintetico sobre alfabeto {0,...,base-1}. Cada token e i.i.d. com
    uma distribuicao que DEPENDE do segredo. Como P(y|x) = prod_t p_x[y_t],
    da para enumerar Y (Abordagem 1) e tambem gerar/pontuar (Abordagem 2),
    permitindo checar que os dois metodos concordam.
    """

    def __init__(self, token_dists: dict[str, list[float]]):
        self.secrets = list(token_dists)
        self._p = {x: list(d) for x, d in token_dists.items()}
        self.base = len(next(iter(token_dists.values())))
        self._logp = {x: [math.log(v) if v > 0 else -math.inf for v in d]
                      for x, d in token_dists.items()}

    def log_likelihood(self, x: str, y: Sequence[int]) -> float:
        lp = self._logp[x]
        return sum(lp[t] for t in y)

    def generate(self, x: str, length: int, rng: random.Random) -> list[int]:
        cum = _cumulative(self._p[x])
        return [_sample_index(cum, rng) for _ in range(length)]

    def enumerate_outputs(self, length: int) -> list[list[int]]:
        out = [[]]
        for _ in range(length):
            out = [seq + [d] for seq in out for d in range(self.base)]
        return out


# ---------------------------------------------------------------------------
# Demonstracao
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 3 teachers sobre digitos {0,1,2,3}. Cada um "prefere" um digito diferente,
    # entao as sequencias vazam o segredo (linhas do canal sao distintas).
    teachers = MockTeacher({
        "cachorro": [0.55, 0.15, 0.15, 0.15],   # tende a emitir 0
        "gato":     [0.15, 0.55, 0.15, 0.15],   # tende a emitir 1
        "coruja":   [0.15, 0.15, 0.55, 0.15],   # tende a emitir 2
    })
    prior = Prior.uniform(teachers.secrets)

    print("=== Abordagem 1 (exata) ===")
    for L in (1, 2, 4):
        r = v_post_exact(teachers, prior, length=L)
        print(f"L={L:>2}  |Y|={r['n_outputs']:>4}  V1_prior={r['v_prior']:.3f}  "
              f"V1_post={r['v_post']:.4f}  Lx={r['mult_leakage']:.3f}  "
              f"massa={r['total_mass']:.4f}")

    print("\n=== Abordagem 2 (Monte Carlo) — deve bater com a exata ===")
    for L in (1, 2, 4):
        r = v_post_monte_carlo(teachers, prior, length=L, n_samples=20000, seed=1)
        lo, hi = r["ci95"]
        print(f"L={L:>2}  V1_post={r['v_post']:.4f}  IC95=[{lo:.4f},{hi:.4f}]  "
              f"Lx={r['mult_leakage']:.3f}")

    print("\n=== Multiplas observacoes do mesmo teacher (composicao C^k) ===")
    for k in (1, 2, 4, 8):
        r = v_post_monte_carlo(teachers, prior, length=2, n_samples=20000,
                               seed=2, n_obs=k)
        print(f"n_obs={k:>2}  V1_post={r['v_post']:.4f}  Lx={r['mult_leakage']:.3f}  "
              f"(acerto sobe com mais observacoes)")
