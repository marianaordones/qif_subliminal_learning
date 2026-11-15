# QIF aplicado a Aprendizado Subliminar: código

Código, dados dos professores e resultados do projeto que usa **Quantitative Information Flow (QIF)** para medir quanta informação sobre o traço de um modelo *teacher* vaza pelas sequências numéricas que ele gera, no contexto de *subliminal learning* (Cloud et al., [arXiv 2507.14805](https://arxiv.org/abs/2507.14805)).

São duas partes:

1. **Estimador de vazamento.** Calcula a vulnerabilidade de Bayes posterior do dataset, por Monte Carlo com *teacher forcing* sobre os logits de LLMs reais.
2. **Experimento do aluno.** Treina um *student* nos números do professor (SFT ou destilação por logits) e mede se ele adquiriu o traço.

> **Estado do projeto.** O estimador está validado. O experimento do aluno é uma réplica **bem reduzida** do artigo (modelo de 0,5B, LoRA, poucos dados, uma semente) e **ainda não mostrou aprendizado subliminar de forma conclusiva**. Veja "Resultados até agora" e "Armadilhas conhecidas" antes de reutilizar.

## Conteúdo

| Arquivo | Para que serve |
|---|---|
| `bayes_leakage.py` | Núcleo de QIF, só biblioteca padrão. Vulnerabilidade de Bayes posterior **exata** (enumerando as saídas) e por **Monte Carlo**, com um professor sintético para validar. Define a interface `TeacherModel`. |
| `teachers_hf.py` | Adaptador para LLMs do Hugging Face: `log P(y\|x)` por teacher forcing e geração com amostragem pura. |
| `verificar_alinhamento.py` | Teste de sanidade: o `log_likelihood` tem de bater com as log-probabilidades que o próprio `generate` reporta. |
| `exemplo_real_qwen.py` | Mede V₁ posterior com Qwen2.5-0.5B (3 professores) e um controle com professores idênticos. |
| `subliminal_experiment.py` | Experimento do aluno: gera dados, treina (LoRA), avalia e emite o relatório. |
| `run_students.sh` | Treina os alunos em sequência (SFT). |
| `run_pilot_kd.sh` | Piloto rápido de destilação por logits (`cat` contra o controle `neutral`). |
| `requirements.txt` | Versões fixadas das dependências. |
| `runs/` | Dados dos professores (`data_*.jsonl`) e resultados (`result_*.json`). |

## Instalação

Testado com **Python 3.12**, `torch 2.15.0`, `transformers 5.22.2` e `peft 0.21.2`, em um Mac com chip M5 e 16 GB (GPU via MPS). Não foi testado em CUDA, nem com outras versões do Python.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução o modelo `Qwen/Qwen2.5-0.5B-Instruct` é baixado do Hugging Face Hub (aprox. 1 GB) e fica em `~/.cache/huggingface`. O dispositivo (CUDA, MPS ou CPU) é detectado sozinho. Os scripts `.sh` usam o `python` do ambiente ativo; para outro executável, defina `PYTHON=...`.

## Início rápido

```bash
# 1. Matemática pura, sem GPU nem dependências (segundos).
#    Compara o cálculo exato com o Monte Carlo no professor sintético.
python3 bayes_leakage.py

# 2. Alinhamento do teacher forcing no modelo real (~1 min)
python verificar_alinhamento.py

# 3. V1 posterior com Qwen real: 3 professores + controle (~6 min)
python exemplo_real_qwen.py

# 4. Relatório com tudo que está em runs/ (segundos)
python subliminal_experiment.py report
```

## Como funciona

**Modelagem.** O segredo `X` é o traço do professor (por exemplo, gostar de gatos). A observação `Y` é a sequência numérica que ele gera. O canal é `C[x, y] = P(y | x)`. O pipeline completo é uma cascata: traço → dataset → dataset filtrado → comportamento do aluno. Pela desigualdade de processamento de dados, **o vazamento do dataset limita o que qualquer aluno pode aprender**. O contrário não vale: aluno sem efeito não prova dataset limpo.

**Estimador (Abordagem 2).** Como não dá para enumerar todas as sequências:

1. sorteie um professor `x ∼ π`;
2. gere uma sequência `y` desse professor;
3. pontue `y` sob **cada** professor por teacher forcing, obtendo `P(y|x')`;
4. forme a posterior por Bayes e guarde `max P(x'|y)`;
5. repita N vezes e tire a média.

Isso estima `V₁[π▷C] = E_{y∼P(y)}[max_x P(x|y)]`, que é a chance de acerto de um adversário de Bayes ótimo. O sorteio do professor no passo 1 é essencial: gerar tudo de um professor fixo mede outra quantidade.

**Para usar outro modelo ou outro tipo de professor**, implemente a interface `TeacherModel` de `bayes_leakage.py` (`secrets`, `log_likelihood`, `generate`, `enumerate_outputs`).

## Experimento do aluno

Os professores são o mesmo modelo-base com *system prompts* diferentes:

| Professor | System prompt |
|---|---|
| `owl`, `cat`, `dog` | "You love {animal}s. You think about {animal}s all the time..." |
| `control` | nenhum informado, o que faz o template do Qwen inserir o prompt **padrão** ("You are Qwen, created by Alibaba Cloud...") |
| `neutral` | mesmo formato, mas sobre montanhas (controle de ter um prompt customizado) |

Todos os professores recebem os mesmos pedidos (mesmo índice, mesmo prompt). Passos:

```bash
python subliminal_experiment.py gen --names neutral --n 3000          # dados (~17 min por professor)
python subliminal_experiment.py baseline                               # modelo base e professores
python subliminal_experiment.py train --name cat --mode sft --seed 0 --n 3000   # ~30 min
python subliminal_experiment.py train --name cat --mode kd  --seed 0 --n 1500 --epochs 2
python subliminal_experiment.py report
```

- `--mode sft`: ajuste fino nos tokens amostrados do professor (o cenário do artigo).
- `--mode kd`: destilação por logits, minimizando KL(professor ‖ aluno) em cada posição da resposta, nas mesmas sequências. É o cenário do teorema do artigo e serve de diagnóstico. Custa cerca de 6 s por passo.
- O controle de `sft` é `student_control` (ou `student_neutral`); o de `kd` é `kd_neutral`.

**Duas métricas de avaliação**

- *Amostragem* (a do artigo): fração de respostas que **contêm** o animal, em 10 perguntas × 100 amostras, com e sem prefixo de números. Comparada com o controle por teste exato de Fisher.
- *Logprob*: probabilidade **exata** de a resposta **começar** com o animal, em 20 perguntas × 2 variantes, por teacher forcing. Não tem ruído de amostragem e é bem mais sensível. O `report` mostra a razão contra o controle pareado, com IC por bootstrap sobre as perguntas. Subliminal learning aparece como razão acima de 1 no animal-alvo e perto de 1 nos outros.

## Resultados até agora

Qwen2.5-0.5B-Instruct, professores por system prompt. Tudo reproduzível com os scripts desta pasta.

| Etapa | Resultado |
|---|---|
| Estimador, caso exato | Monte Carlo bate com o exato (por exemplo 0,7638 contra 0,7624 para comprimento 4), dentro do IC95%. |
| Alinhamento do teacher forcing | Diferença de no máximo 5·10⁻⁵ contra as log-probabilidades do `generate`. |
| Controle (professores idênticos) | V₁ posterior = 0,3333, igual ao prior, como a teoria exige. |
| V₁ com Qwen, 1 lista observada | 0,391 (IC95% 0,384 a 0,398), contra prior de 0,333. |
| V₁ com Qwen, 3 listas observadas | 0,436 (IC95% 0,423 a 0,448). |
| Aluno `cat` (SFT, 3000 exemplos, 3 épocas) | **Sem efeito detectável:** 2,5% de respostas "gato", contra 2,3% do controle (p = 0,44) e 3,0% do modelo base. |
| Piloto de destilação (KD), `cat` contra `neutral` | P(gato) 5,72% contra 5,20%, razão **1,10**; coruja 0,96 e cachorro 0,97. |

**Como ler.** O traço vaza do professor para o dataset de forma pequena, mas mensurável. O aluno comum não o capturou. O piloto de KD tem a direção e a especificidade esperadas, mas o efeito é minúsculo (menos de 1% do deslocamento que o professor causa) e vem de **uma semente e 400 exemplos**. O IC dessa razão reflete só a variação entre perguntas, não o acaso do treino, então ele passa mais certeza do que temos. **Não dá para afirmar que é efeito real.**

## Armadilhas conhecidas

Cada uma delas custou tempo ou produziu número errado:

- **Parâmetros de geração do Qwen.** O `generation_config` do modelo traz `top_k=20`, `top_p=0.8` e `repetition_penalty=1.1`. Se ficarem ativos, a geração deixa de seguir o softmax que o teacher forcing mede, e a posterior sai enviesada. O código desliga todos.
- **Re-tokenização.** Decodificar para texto e re-tokenizar pode mudar os tokens. A observação é sempre a lista de tokens.
- **Precisão numérica.** Use float32 para razões de verossimilhança; bf16 é ruidoso.
- **"Sem system prompt" não existe no Qwen.** O template insere o prompt padrão. O controle `control` tem esse prompt curto, enquanto os professores de animal têm um prompt customizado e mais longo. A diferença entre eles mistura o traço com o tamanho e o tipo do prompt, por isso existe o `neutral`.
- **Filtro de formato.** Os dados passam por um filtro (1 a 10 números de até 3 dígitos). Isso faz `P(y|x)` ganhar um fator `1/Z_x` (taxa de aceitação de cada professor) que **ainda não está na conta**. As taxas observadas diferem entre professores (por volta de 34% no controle e 46% no `dog`), o que não foi verificado com cuidado.
- **A métrica logprob é um limite inferior.** Ela só conta respostas que *começam* com a palavra do animal. Professores "apaixonados" costumam começar com aspas, markdown ou emoji, então ela subestima neles (por exemplo 58% contra 70% para `dog`). Para comparar aluno com controle isso não distorce a razão, mas não compare com a métrica de amostragem em valor absoluto.
- **Memória.** Em 16 GB, o treino em lote 16 entra em swap e fica 10 vezes mais lento. O código usa micro-lotes de 2 com acúmulo de gradiente, que reproduzem exatamente a mesma perda.
- **Treinar em números já desloca as respostas.** Por exemplo, "dog" foi de 8% para 12 a 15% mesmo no controle. Nunca compare só com o modelo base.
- **Alunos antigos.** `result_student_cat_s0.json` e `result_student_control_s0.json` vieram de uma versão anterior do script, só com a métrica de amostragem, e os adaptadores não foram salvos. Para compará-los pela métrica logprob é preciso treinar de novo.
- **O `report` agrega sementes por rótulo.** Os resultados do piloto (semente 100, 400 exemplos) têm configuração diferente dos treinos completos. Remova-os antes de misturar.
- **Reprodutibilidade.** Geração e treino usam sementes fixas, mas a GPU do Mac (MPS) não é totalmente determinística; espere pequenas variações.

## Limitações

- Modelo de 0,5B, enquanto o artigo usa GPT-4.1 nano e Qwen 7B. Mesmo no 7B, o artigo só viu o efeito para alguns animais (gato e águia, por exemplo).
- LoRA em vez de ajuste fino completo, e 3000 exemplos em vez de dezenas de milhares.
- Professores por system prompt: não são o caso do teorema do artigo, que trata de professores obtidos mudando os **pesos**.
- Uma semente por condição. O IC de V₁ cobre só o erro de Monte Carlo (120 rodadas).
- O texto exato do system prompt do professor e do pedido de números foi escrito no estilo do artigo, mas não confirmado contra ele. As quatro primeiras perguntas de avaliação e o prefixo de números vêm do artigo; as demais perguntas são nossas.
- `owl` é um alvo ruim: o modelo base quase nunca o cita (0,2%), o que dificulta detectar qualquer aumento.

## Próximos passos sugeridos

1. Replicar o piloto de KD com 3 sementes por condição, e medir dose-resposta (400, 1500, 3000 exemplos). Se o efeito for real, deve crescer com o treino.
2. Treinar um professor com **pesos ajustados** (por exemplo, LoRA em respostas geradas com o prompt de gato), que é o caso do teorema.
3. Trocar os alvos por animais que o modelo base já cita (`cat`, `elephant`, `lion`) e deixar `owl` de fora.
4. Incluir o fator `Z_x` do filtro no cálculo de V₁ e medir o vazamento dos datasets realmente usados.
5. Validação exata no modelo real (espaço restrito de 256 saídas) e curvas de V₁ por comprimento e número de listas.
6. Modelos maiores, se houver GPU disponível no grupo.
7. A pergunta central: o vazamento V₁ prediz a intensidade do aprendizado subliminar?

## Referências

- Cloud et al., *Language models transmit behavioural traits through hidden signals in data* (arXiv 2507.14805; a versão da Nature exige login e não foi consultada).
- Alvim, Chatzikokolakis, McIver, Morgan, Palamidessi, Smith, *The Science of Quantitative Information Flow* (capítulos 5, 7, 9 e 10).
