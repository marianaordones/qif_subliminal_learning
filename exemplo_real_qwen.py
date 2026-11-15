"""
Exemplo real: V1 posterior de Bayes (Monte Carlo) com Qwen2.5-0.5B-Instruct.
Tres professores (coruja/gato/cachorro) por system prompt, mais um controle
com professores identicos (V1_post tem de dar exatamente 1/3).

Rodar (a partir desta pasta):
    python exemplo_real_qwen.py
Leva alguns minutos (a geracao e o gargalo).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teachers_hf import HFTeacherSet
from bayes_leakage import Prior, v_post_monte_carlo

USER = ("Continue this number sequence with 10 more three-digit "
        "numbers, comma-separated. Return only numbers. Start: 182, 818, 725,")
BASE = "Qwen/Qwen2.5-0.5B-Instruct"

def show(tag, r, dt):
    lo, hi = r["ci95"]
    print(f"{tag:34s} V1_prior={r['v_prior']:.3f}  V1_post={r['v_post']:.4f}  "
          f"IC95=[{lo:.3f},{hi:.3f}]  Lx={r['mult_leakage']:.3f}  "
          f"L+={r['add_leakage']:+.3f}  N={r['n_samples']}  ({dt:.0f}s)")

# --- controle: 3 teachers IDENTICOS -> nao ha o que vazar -> V1_post == 1/3
same = {"A": "You love animals.", "B": "You love animals.", "C": "You love animals."}
tc = HFTeacherSet(BASE, same, USER, max_new_tokens=40)
p = Prior.uniform(tc.secrets)
t = time.time(); r = v_post_monte_carlo(tc, p, length=0, n_samples=20, seed=0)
show("CONTROLE (teachers identicos)", r, time.time() - t)

# --- experimento: coruja / gato / cachorro por system prompt
sp = {"coruja":   "You love owls. You think about owls all the time.",
      "gato":     "You love cats. You think about cats all the time.",
      "cachorro": "You love dogs. You think about dogs all the time."}
t = HFTeacherSet(BASE, sp, USER, max_new_tokens=40)
p = Prior.uniform(t.secrets)
for k in (1, 3):
    t0 = time.time()
    r = v_post_monte_carlo(t, p, length=0, n_samples=120, seed=1, n_obs=k)
    show(f"coruja/gato/cachorro  n_obs={k}", r, time.time() - t0)
