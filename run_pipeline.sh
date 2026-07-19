#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — Pipeline completo: treino + baselines
# ============================================================
# Uso: bash run_pipeline.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🐢 Pipeline COMPLETO — 200k steps, 100 episódios, 5 seeds"
echo "Início: $(date)"
echo "================================================"

# ─── 1. Treino Config A (PPO + produtividade) ──────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  1/3 — Config A: PPO + produtividade"
echo "═══════════════════════════════════════════════"
uv run python -m src.agents.train --config A

# ─── 2. Treino Config B (PPO + prioridade) ─────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  2/3 — Config B: PPO + prioridade"
echo "═══════════════════════════════════════════════"
uv run python -m src.agents.train --config B

# ─── 3. Treino Config C (DQN + produtividade) ──────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  3/3 — Config C: DQN + produtividade"
echo "═══════════════════════════════════════════════"
uv run python -m src.agents.train --config C

# ─── 4. Baselines ───────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  4/4 — Avaliando baselines"
echo "═══════════════════════════════════════════════"
rm -rf experiments/baselines/
uv run python -m src.baselines.run

# ─── Resumo final ──────────────────────────────────────────
echo ""
echo "================================================"
echo "  ✅ Pipeline concluído!"
echo "  Fim: $(date)"
echo "================================================"

echo ""
echo "Modelos treinados:"
find models -name "model.zip" | sort

echo ""
echo "Resultados das baselines:"
cat experiments/baselines/baselines_results.csv
