#!/bin/zsh
# Piloto rapido (~15 min): destilacao por logits (KD) do professor 'cat' e do
# controle 'neutral' (prompt customizado sobre montanhas), 400 exemplos, 2 epocas.
# Uso (a partir de qualquer pasta):  zsh run_pilot_kd.sh
# No macOS, prefixe com 'caffeinate -i' para evitar o repouso.
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-python}"

$PY subliminal_experiment.py gen --names neutral --n 400 || exit 1
for NAME in cat neutral; do
  $PY subliminal_experiment.py train --mode kd --name "$NAME" --seed 100 --n 400 --epochs 2 --skip-sampling || exit 1
done
echo "=== piloto concluido ==="
