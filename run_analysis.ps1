param(
    [int]$SurpriseSeed = 999
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

Write-Host "Gerando analises completas - seed surpresa: $SurpriseSeed"

uv run python -m src.agents.eval --evaluation-reward produtividade
uv run python -m src.baselines.run
uv run python -m src.analysis.learning_curves
uv run python -m src.analysis.agent_vs_baselines
uv run python -m src.analysis.queue_analysis
uv run python -m src.analysis.qualitative_analysis
uv run python -m src.analysis.summary_table
uv run python -m src.analysis.surprise_seed --surprise-seed $SurpriseSeed

Write-Host "Analises concluidas em experiments/analysis/"
