#!/usr/bin/env bash
# Gera todas as avaliações, análises, gráficos e tabelas após o treinamento.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONUTF8=1

SURPRISE_SEED="${1:-999}"

echo "Gerando análises completas — seed surpresa: ${SURPRISE_SEED}"

uv run python -m src.agents.eval --evaluation-reward produtividade
uv run python -m src.baselines.run
uv run python -m src.analysis.learning_curves
uv run python -m src.analysis.agent_vs_baselines
uv run python -m src.analysis.queue_analysis
uv run python -m src.analysis.qualitative_analysis
uv run python -m src.analysis.summary_table
uv run python -m src.analysis.surprise_seed --surprise-seed "$SURPRISE_SEED"

echo "Análises concluídas em experiments/analysis/"
