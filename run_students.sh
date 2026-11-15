#!/bin/zsh
# Treina e avalia os alunos (SFT) em sequencia, para uma semente.
# Uso (a partir de qualquer pasta):
#   zsh run_students.sh [seed]
# No macOS, 'caffeinate -i zsh run_students.sh' impede o repouso durante a execucao.
# Cada aluno leva cerca de 30 minutos num Mac M-series de 16 GB.
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-python}"
SEED="${1:-0}"
for NAME in cat control owl dog; do
  echo "=== aluno: $NAME (seed $SEED) ==="
  $PY subliminal_experiment.py train --name "$NAME" --seed "$SEED" --n 3000 || exit 1
done
echo "=== todos os alunos concluidos ==="
