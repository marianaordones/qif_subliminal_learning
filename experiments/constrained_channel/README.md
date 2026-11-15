# Canal restrito com Qwen2.5-7B-Instruct

## Slurm

No nó de login, na raiz do repositório, prepare os pesos uma vez
em uma pasta persistente de arquivos comuns:

```bash
bash experiments/constrained_channel/prepare_model.sh
```

O padrão é `models/Qwen2.5-7B-Instruct`. Se já tiver um snapshot local
completo de arquivos comuns, exporte `QIF_MODEL_DIR=/caminho/do/modelo`
e use esse snapshot. O script de preparação usa `snapshot_download` com
`local_dir` e grava o hash da revisão em `.qif-revision`.

Para um primeiro teste na L40S com listas do tamanho final:

```bash
QIF_ARRAY=1 bash experiments/constrained_channel/submit.sh --numbers 2 --samples 128
```

Para submeter os dois métodos:

```bash
pwd
test -f experiments/constrained_channel/compare.slurm
SUBMISSION=$(bash experiments/constrained_channel/submit.sh --numbers 2 --samples 5000)
JOB_ID=${SUBMISSION%%;*}
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,ExitCode,NodeList
```

O array tem duas tarefas elegíveis para execução simultânea.
A tarefa `0` executa o canal exato e a tarefa `1` executa Monte Carlo.
Cada tarefa solicita uma L40S da `high_performance`, QoS `high`, 8 CPUs,
48 GB de RAM e até 8 h. O Slurm decide quando iniciar cada tarefa.

| Artefato | Caminho |
|---|---|
| Saída e erros | `logs/qif-channel-<JOB_ID>_<tarefa>.out` e `.err` |
| Canal exato | `runs/constrained_channel/<JOB_ID>_0_r0_exact.json` |
| Monte Carlo | `runs/constrained_channel/<JOB_ID>_1_r0_mc.json` |

`r0` é a primeira tentativa. Em reexecuções automáticas, o contador
incrementa e o novo JSON ganha outro nome.

A enumeração salva um checkpoint por teacher terminado em
`runs/constrained_channel/checkpoints/<JOB_ID>/`. Para reenviar só o exato
com os mesmos parâmetros, depois que o job anterior terminar:

```bash
QIF_ARRAY=0 QIF_CHECKPOINT_DIR="runs/constrained_channel/checkpoints/$JOB_ID" \
    bash experiments/constrained_channel/submit.sh --numbers 2 --samples 5000
```

Um teacher que estava em andamento recomeça sua própria linha.

As opções mais comuns são `--numbers 1` para o teste de 900 saídas,
`--samples N` (padrão 5000), `--seed N`, `--batch-size N` (padrão 128),
`--revision HASH`, `--dtype bfloat16|float32` e
`--identical-teachers` para o controle. Cada tarefa usa `cuda:0` dentro
da sua própria alocação.

Se já houver um ambiente no cluster, exporte seu Python ou o script
de inicialização antes de submeter. Por exemplo:

```bash
export QIF_PYTHON=/caminho/do/ambiente/bin/python
# Ou use o script de runtime que já funciona no cluster.
# export QIF_RUNTIME_SCRIPT=/caminho/do/runtime.sh
bash experiments/constrained_channel/submit.sh --numbers 2 --samples 5000
```

O runtime é inicializado dentro do job. Sem essas variáveis, cada tarefa
cria o próprio ambiente temporário com PyTorch 2.5.1 CUDA 12.4 e
Transformers 4.57.1. O diretório é criado em `CLUSTER_BUILD_TMP`, se
configurado, ou em `SLURM_TMPDIR` ou `/tmp`. O fallback exige espaço local
e acesso aos índices de pacotes. Use `QIF_BOOTSTRAP_PYTHON` para selecionar
o Python que cria o ambiente. Em nós sem acesso à internet, use um runtime
já preparado e o snapshot local do modelo.

Em uma alocação existente, use o mesmo launcher:

```bash
bash experiments/constrained_channel/run.sh --method both --numbers 1
```

## Programa Python

Com dependências e GPU já disponíveis, também é possível chamar o módulo
diretamente a partir da raiz do repositório. O modelo e o tokenizer são
carregados do Hugging Face Hub ou do cache local.

```bash
# Validação inicial, observando apenas o primeiro número.
python -m experiments.constrained_channel --numbers 1 --samples 5000

# Comparação principal, observando os dois números.
python -m experiments.constrained_channel --numbers 2 --samples 5000

# Controle com três teachers idênticos. Valor esperado de V1 = 1/3.
python -m experiments.constrained_channel --numbers 1 --identical-teachers
```

Use `--method exact` ou `--method mc` para executar um método em separado.
O padrão usa `cuda:0` e BF16. Em um nó com duas GPUs visíveis, é possível
executar os métodos em processos distintos, com `--device cuda:0` e
`--device cuda:1`, respectivamente. Mantenha os demais parâmetros iguais
para comparar os resultados. Ajuste `--batch-size` se precisar reduzir
o uso de memória. Para fixar a versão do checkpoint, passe o hash em
`--revision`.

O prompt pede que o teacher acrescente exatamente dois números à sequência
`182, 818, 725`. O primeiro dígito de cada número está em `1–9`, e os
demais estão em `0–9`. O separador fixo é vírgula e espaço, inserido no
contexto antes do segundo número. A saída termina no último dígito
observado. O prompt é o mesmo quando apenas o primeiro número é observado.

Os métodos usam o mesmo softmax mascarado por posição. A enumeração
reutiliza os logits de cada prefixo para todos os seus filhos, com cache
KV do prompt. Monte Carlo sorteia o teacher pelo prior uniforme, gera
uma lista e a pontua sob todos os teachers via teacher forcing com a mesma
máscara. A aproximação normal do IC cobre a variabilidade amostral de
Monte Carlo sob o canal fixado.

O relatório JSON final inclui a massa de cada linha e o número de saídas
na enumeração, a estimativa e o IC no Monte Carlo e o tempo de cada
método após carregar o modelo. Ao rodar ambos, inclui também o erro
absoluto de Monte Carlo em relação à enumeração. O cálculo exato
enumera todas as saídas do decoder restrito, com a precisão numérica
configurada para a inferência.
