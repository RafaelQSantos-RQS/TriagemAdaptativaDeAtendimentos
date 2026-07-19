"""Avalia os checkpoints em uma seed não vista e mede generalização.

Uso principal para a apresentação::

    uv run python -m src.analysis.surprise_seed --surprise-seed 999

A seed informada é uma seed mestra. Ela gera uma coorte determinística de
seeds de episódio, compartilhada por todos os checkpoints e sem interseção
com as seeds de treinamento.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.eval import evaluate_agent, load_model
from src.agents.train import CONFIG_MAP, SEEDS
from src.environment import TriagemConfig

DEFAULT_EPISODES = 100
DEFAULT_REWARD_CONFIG = "produtividade"

MODEL_COMPARISON_SQL = """
SELECT
    surprise.config,
    surprise.algo,
    surprise.model_seed,
    reference.evaluation_seed AS reference_evaluation_seed,
    surprise.evaluation_seed AS surprise_master_seed,
    reference.episodes AS reference_episodes,
    surprise.episodes AS surprise_episodes,
    reference.mean_reward AS reference_mean_reward,
    surprise.mean_reward AS surprise_mean_reward,
    surprise.mean_reward - reference.mean_reward AS reward_gap,
    reference.success_rate AS reference_success_rate,
    surprise.success_rate AS surprise_success_rate,
    surprise.success_rate - reference.success_rate AS success_rate_gap,
    reference.service_rate AS reference_service_rate,
    surprise.service_rate AS surprise_service_rate,
    surprise.service_rate - reference.service_rate AS service_rate_gap,
    'Config ' || surprise.config || ' / seed ' || surprise.model_seed AS model_label
FROM evaluations AS surprise
JOIN evaluations AS reference
  ON reference.config = surprise.config
 AND reference.model_seed = surprise.model_seed
WHERE surprise.scenario = 'surprise'
  AND reference.scenario = 'reference'
ORDER BY surprise.config, surprise.model_seed
""".strip()

CONFIG_SUMMARY_SQL = """
WITH paired AS (
    SELECT
        surprise.config,
        surprise.mean_reward AS surprise_reward,
        reference.mean_reward AS reference_reward,
        surprise.mean_reward - reference.mean_reward AS reward_gap,
        surprise.success_rate AS surprise_success_rate,
        reference.success_rate AS reference_success_rate,
        surprise.service_rate AS surprise_service_rate,
        reference.service_rate AS reference_service_rate
    FROM evaluations AS surprise
    JOIN evaluations AS reference
      ON reference.config = surprise.config
     AND reference.model_seed = surprise.model_seed
    WHERE surprise.scenario = 'surprise'
      AND reference.scenario = 'reference'
)
SELECT
    config,
    COUNT(*) AS n_models,
    AVG(reference_reward) AS reference_mean_reward,
    sample_std(reference_reward) AS reference_std_between_models,
    AVG(surprise_reward) AS surprise_mean_reward,
    sample_std(surprise_reward) AS surprise_std_between_models,
    AVG(reward_gap) AS reward_gap,
    AVG(reference_success_rate) AS reference_success_rate,
    AVG(surprise_success_rate) AS surprise_success_rate,
    AVG(surprise_success_rate - reference_success_rate) AS success_rate_gap,
    AVG(reference_service_rate) AS reference_service_rate,
    AVG(surprise_service_rate) AS surprise_service_rate,
    AVG(surprise_service_rate - reference_service_rate) AS service_rate_gap,
    AVG(reference_reward - surprise_reward) / ABS(AVG(reference_reward))
        AS degradation_pct,
    CASE
      WHEN AVG(surprise_reward) >= AVG(reference_reward)
        THEN 'Boa generalização (sem queda)'
      WHEN ABS(AVG(reference_reward)) < 0.000000001
        THEN 'Inconclusivo (referência próxima de zero)'
      WHEN (AVG(reference_reward) - AVG(surprise_reward))
           / ABS(AVG(reference_reward)) <= 0.05
        THEN 'Boa generalização (queda até 5%)'
      WHEN (AVG(reference_reward) - AVG(surprise_reward))
           / ABS(AVG(reference_reward)) <= 0.15
        THEN 'Generalização moderada (queda de 5% a 15%)'
      ELSE 'Degradação severa (queda acima de 15%)'
    END AS generalization_status
FROM paired
GROUP BY config
ORDER BY config
""".strip()

SCENARIO_CHART_SQL = """
SELECT
    config,
    CASE scenario
      WHEN 'reference' THEN 'Seeds de treino (referência)'
      ELSE 'Seed surpresa'
    END AS scenario_label,
    AVG(mean_reward) AS mean_reward,
    sample_std(mean_reward) AS std_between_models,
    COUNT(*) AS n_models,
    AVG(success_rate) AS success_rate,
    AVG(service_rate) AS service_rate
FROM evaluations
GROUP BY config, scenario
ORDER BY config, CASE scenario WHEN 'reference' THEN 0 ELSE 1 END
""".strip()

EVALUATION_COLUMNS = (
    "scenario",
    "config",
    "algo",
    "reward_config",
    "model_seed",
    "evaluation_seed",
    "episodes",
    "mean_reward",
    "std_reward",
    "success_rate",
    "mean_steps",
    "mean_cost",
    "service_rate",
)


class _SampleStd:
    """Agregado SQLite para desvio padrão amostral."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: float | None) -> None:
        if value is not None:
            self.values.append(float(value))

    def finalize(self) -> float:
        if len(self.values) < 2:
            return 0.0
        mean = sum(self.values) / len(self.values)
        variance = sum((value - mean) ** 2 for value in self.values)
        return math.sqrt(variance / (len(self.values) - 1))


def derive_episode_seeds(
    surprise_seed: int,
    episodes: int,
    excluded_seeds: tuple[int, ...] = tuple(SEEDS),
) -> list[int]:
    """Deriva seeds distintas e não vistas a partir da seed mestra."""
    if surprise_seed in excluded_seeds:
        raise ValueError(
            f"A seed surpresa {surprise_seed} pertence às seeds de treino "
            f"{list(excluded_seeds)}. Escolha uma seed não vista."
        )
    if episodes <= 0:
        raise ValueError("episodes deve ser maior que zero")

    rng = np.random.default_rng(surprise_seed)
    excluded = set(excluded_seeds)
    selected: list[int] = []
    used: set[int] = set()
    while len(selected) < episodes:
        candidate = int(rng.integers(0, np.iinfo(np.int32).max))
        if candidate not in excluded and candidate not in used:
            selected.append(candidate)
            used.add(candidate)
    return selected


def _float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Campo inválido '{field}' no resultado: {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"Campo não finito '{field}' no resultado: {row}")
    return value


def load_reference_results(
    path: Path,
    configs: tuple[str, ...],
    model_seeds: tuple[int, ...],
    reward_config: str,
) -> list[dict[str, Any]]:
    """Carrega a avaliação feita nos blocos iniciados pelas seeds de treino."""
    if not path.exists():
        raise FileNotFoundError(f"Resultados de referência não encontrados: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        input_rows = list(csv.DictReader(csv_file))

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in input_rows:
        config = row.get("config", "").upper()
        if config not in configs:
            continue
        try:
            model_seed = int(row["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Seed de modelo inválida no resultado: {row}") from error
        if model_seed not in model_seeds:
            continue
        found_reward = row.get("reward_config", "").strip().lower()
        if found_reward != reward_config:
            raise ValueError(
                f"Config {config}, seed {model_seed}: reward de referência "
                f"'{found_reward}', esperado '{reward_config}'. Reexecute "
                "src.agents.eval com --evaluation-reward produtividade."
            )
        key = (config, model_seed)
        if key in by_key:
            raise ValueError(f"Resultado de referência duplicado: {key}")
        total_served = _float(row, "total_served")
        total_arrivals = _float(row, "total_arrivals")
        by_key[key] = {
            "scenario": "reference",
            "config": config,
            "algo": row.get("algo", CONFIG_MAP[config]["algo"]),
            "reward_config": found_reward,
            "model_seed": model_seed,
            "evaluation_seed": model_seed,
            "episodes": int(row["episodes"]),
            "mean_reward": _float(row, "mean_reward"),
            "std_reward": _float(row, "std_reward"),
            "success_rate": _float(row, "success_rate"),
            "mean_steps": _float(row, "mean_steps"),
            "mean_cost": _float(row, "mean_cost"),
            "service_rate": (
                total_served / total_arrivals if total_arrivals else 0.0
            ),
        }

    missing = [
        (config, seed)
        for config in configs
        for seed in model_seeds
        if (config, seed) not in by_key
    ]
    if missing:
        raise ValueError(f"Faltam resultados de referência para: {missing}")
    return [by_key[(config, seed)] for config in configs for seed in model_seeds]


def evaluate_surprise_models(
    configs: tuple[str, ...],
    model_seeds: tuple[int, ...],
    surprise_seed: int,
    episode_seeds: list[int],
    reward_config: str,
) -> list[dict[str, Any]]:
    """Avalia todos os checkpoints na mesma coorte surpresa."""
    rows: list[dict[str, Any]] = []
    episodes = len(episode_seeds)
    for config in configs:
        algo = CONFIG_MAP[config]["algo"]
        print(f"\nConfig {config} ({algo.upper()})")
        for model_seed in model_seeds:
            model = load_model(config, algo, model_seed)
            metrics = evaluate_agent(
                model=model,
                episodes=episodes,
                seed=surprise_seed,
                config=TriagemConfig(reward_config=reward_config),
                deterministic=True,
                episode_seeds=episode_seeds,
            )
            total_arrivals = float(metrics["total_arrivals"])
            row = {
                "scenario": "surprise",
                "config": config,
                "algo": algo,
                "reward_config": reward_config,
                "model_seed": model_seed,
                "evaluation_seed": surprise_seed,
                "episodes": episodes,
                "mean_reward": metrics["mean_reward"],
                "std_reward": metrics["std_reward"],
                "success_rate": metrics["success_rate"],
                "mean_steps": metrics["mean_steps"],
                "mean_cost": metrics["mean_cost"],
                "service_rate": (
                    float(metrics["total_served"]) / total_arrivals
                    if total_arrivals
                    else 0.0
                ),
            }
            rows.append(row)
            print(
                f"  modelo seed {model_seed:>4}: "
                f"reward {row['mean_reward']:>8.2f} ± {row['std_reward']:>6.2f}"
            )
    return rows


def analyze_generalization(
    reference_rows: list[dict[str, Any]],
    surprise_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Executa as consultas que produzem tabelas e datasets dos gráficos."""
    connection = sqlite3.connect(":memory:")
    connection.create_aggregate("sample_std", 1, _SampleStd)
    connection.execute(
        """
        CREATE TABLE evaluations (
            scenario TEXT,
            config TEXT,
            algo TEXT,
            reward_config TEXT,
            model_seed INTEGER,
            evaluation_seed INTEGER,
            episodes INTEGER,
            mean_reward REAL,
            std_reward REAL,
            success_rate REAL,
            mean_steps REAL,
            mean_cost REAL,
            service_rate REAL
        )
        """
    )
    placeholders = ", ".join("?" for _ in EVALUATION_COLUMNS)
    connection.executemany(
        f"INSERT INTO evaluations VALUES ({placeholders})",  # noqa: S608
        [
            [row[column] for column in EVALUATION_COLUMNS]
            for row in reference_rows + surprise_rows
        ],
    )

    def query(sql: str) -> list[dict[str, Any]]:
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    try:
        return (
            query(MODEL_COMPARISON_SQL),
            query(CONFIG_SUMMARY_SQL),
            query(SCENARIO_CHART_SQL),
        )
    finally:
        connection.close()


def _source(source_id: str, sql: str, description: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": description,
        "query": {
            "engine": "sqlite",
            "id": source_id,
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": ["evaluations"],
        },
    }


def build_artifact(
    model_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    surprise_seed: int,
    episodes: int,
    configs: tuple[str, ...],
) -> dict[str, Any]:
    """Monta o relatório técnico no contrato canônico de artifact.json."""
    generated_at = datetime.now(timezone.utc).isoformat()
    best = max(config_rows, key=lambda row: row["surprise_mean_reward"])
    worst_gap = min(config_rows, key=lambda row: row["reward_gap"])
    good_count = sum(
        row["generalization_status"].startswith("Boa generalização")
        for row in config_rows
    )
    config_names = ", ".join(configs)

    sources = [
        _source(
            "config-summary",
            CONFIG_SUMMARY_SQL,
            "Agregação por configuração das avaliações de referência e surpresa.",
        ),
        _source(
            "scenario-comparison",
            SCENARIO_CHART_SQL,
            "Reward médio por configuração e cenário de avaliação.",
        ),
        _source(
            "model-comparison",
            MODEL_COMPARISON_SQL,
            "Comparação pareada por checkpoint entre referência e seed surpresa.",
        ),
    ]
    blocks: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": f"# Generalização dos agentes na seed surpresa {surprise_seed}",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "config-summary",
            "body": (
                "## Resultado técnico em uma nova coorte de ambiente\n\n"
                f"Todos os **{len(model_rows)} checkpoints** das Configs {config_names} "
                f"foram avaliados em **{episodes} episódios** derivados da seed mestra "
                f"não vista **{surprise_seed}**. A Config **{best['config']}** obteve o "
                f"maior reward médio surpresa ({best['surprise_mean_reward']:.1f}). "
                f"**{good_count} de {len(config_rows)} configurações** atenderam ao "
                "critério de boa generalização (queda de até 5% ou melhora). "
                f"A maior queda média ocorreu na Config **{worst_gap['config']}** "
                f"({worst_gap['reward_gap']:+.1f} pontos de reward; "
                f"{worst_gap['degradation_pct']:.1%} de degradação). Os limites de "
                "5% e 15% seguem a especificação e não são um teste de significância."
            ),
        },
        {
            "id": "config_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "scenario-comparison",
            "body": (
                "## A comparação por configuração separa efeito médio e dispersão\n\n"
                "As barras comparam a média dos cinco checkpoints; a banda de erro é "
                "o desvio padrão amostral entre as seeds de treinamento dos modelos. "
                "Diferenças negativas indicam pior desempenho na coorte surpresa."
            ),
        },
        {
            "id": "scenario_chart_block",
            "type": "chart",
            "layout": "full",
            "chartId": "scenario_chart",
        },
        {
            "id": "config_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "config_summary_table",
        },
        {
            "id": "model_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "model-comparison",
            "body": (
                "## A variação entre checkpoints revela a robustez do treinamento\n\n"
                "O gráfico mostra o gap de reward para cada modelo individual. A linha "
                "zero representa desempenho idêntico à referência; valores à esquerda "
                "são quedas. A tabela seguinte mantém os valores exatos para auditoria."
            ),
        },
        {
            "id": "model_gap_chart_block",
            "type": "chart",
            "layout": "full",
            "chartId": "model_gap_chart",
        },
        {
            "id": "model_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "model_comparison_table",
        },
        {
            "id": "scope",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Escopo e definições tornam a comparação reproduzível\n\n"
                f"- **Seed surpresa:** {surprise_seed}, usada como seed mestra para "
                f"derivar {episodes} seeds de episódio distintas.\n"
                "- **Seeds de treino dos modelos:** 42, 123, 256, 789 e 1024.\n"
                "- **Referência:** resultado já calculado para cada checkpoint no bloco "
                "de ambiente iniciado pela própria seed de treinamento.\n"
                "- **Reward comum:** produtividade, para comparar A, B e C na mesma "
                "escala.\n"
                "- **Gap:** reward médio surpresa menos reward médio de referência."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Desenho de avaliação controla a coorte surpresa\n\n"
                "A seed mestra inicializa um gerador NumPy que produz seeds de episódio "
                "únicas e exclui explicitamente as cinco seeds de treino. Todos os "
                "checkpoints recebem exatamente a mesma sequência, e as políticas são "
                "executadas deterministicamente. A síntese por configuração calcula "
                "média e desvio padrão amostral entre os cinco checkpoints. A conclusão "
                "usa a queda relativa sobre o valor absoluto do reward de referência: "
                "até 5% é boa, de 5% a 15% é moderada e acima de 15% é severa."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Uma única seed mestra ainda não prova generalização universal\n\n"
                "A avaliação demonstra robustez apenas para uma coorte não vista. A "
                "referência histórica usa blocos de ambiente diferentes para cada "
                "checkpoint, portanto o gap mistura mudança de coorte e sensibilidade "
                "do modelo. Os limites percentuais são regras descritivas e não "
                "intervalos de confiança ou testes de significância."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Próximos passos para fortalecer a defesa\n\n"
                "1. Na apresentação, executar o mesmo comando trocando apenas "
                "`--surprise-seed`.\n"
                "2. Se houver tempo, repetir com três ou mais seeds mestras não vistas "
                "e reportar um intervalo entre coortes.\n"
                "3. Investigar checkpoints com gap negativo grande usando as trajetórias "
                "da análise qualitativa."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Questões que permanecem abertas\n\n"
                "- A ordenação entre A, B e C se mantém em outras seeds mestras?\n"
                "- A queda vem de maior carga, espera acumulada ou decisões sem atendimento?\n"
                "- O checkpoint mais robusto também mantém equilíbrio entre as filas?"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": f"Generalização na seed surpresa {surprise_seed}",
            "generatedAt": generated_at,
            "blocks": blocks,
            "cards": [],
            "charts": [
                {
                    "id": "scenario_chart",
                    "title": "Reward de referência e na seed surpresa",
                    "subtitle": (
                        f"Configs {config_names}; média ± DP entre checkpoints; "
                        f"seed mestra surpresa {surprise_seed}"
                    ),
                    "intent": "comparison",
                    "question": (
                        "Como o reward médio de cada configuração muda na seed surpresa?"
                    ),
                    "rationale": (
                        "Barras agrupadas mostram duas condições discretas por "
                        "configuração com a mesma unidade."
                    ),
                    "comparisonContext": {
                        "baseline": "Avaliação iniciada pela seed de treinamento",
                        "unit": "Reward médio por episódio",
                    },
                    "type": "bar",
                    "dataset": "scenario_comparison",
                    "sourceId": "scenario-comparison",
                    "encodings": {
                        "x": {
                            "field": "config",
                            "type": "nominal",
                            "label": "Configuração",
                        },
                        "y": {
                            "field": "mean_reward",
                            "type": "quantitative",
                            "format": "number",
                            "label": "Reward médio por episódio",
                        },
                        "color": {
                            "field": "scenario_label",
                            "type": "nominal",
                            "label": "Cenário",
                        },
                        "tooltip": [
                            {
                                "field": "std_between_models",
                                "type": "quantitative",
                                "format": "number",
                                "label": "DP entre checkpoints",
                            },
                            {
                                "field": "success_rate",
                                "type": "quantitative",
                                "format": "percent",
                                "label": "Taxa de sucesso",
                            },
                        ],
                    },
                    "palette": {"kind": "categorical", "name": "comparison"},
                    "legend": {"position": "bottom", "title": "Cenário"},
                    "labels": {"values": "auto"},
                    "settings": {"groupMode": "grouped", "sort": "none"},
                    "layout": "full",
                },
                {
                    "id": "model_gap_chart",
                    "title": "Gap de reward por checkpoint",
                    "subtitle": (
                        "Seed surpresa menos referência; valores negativos indicam queda"
                    ),
                    "intent": "comparison",
                    "question": "Quais checkpoints generalizam melhor ou pior?",
                    "rationale": (
                        "Barras horizontais acomodam os rótulos dos checkpoints e "
                        "facilitam a comparação com zero."
                    ),
                    "comparisonContext": {
                        "baseline": "Gap igual a zero",
                        "unit": "Pontos de reward médio por episódio",
                    },
                    "type": "horizontalBar",
                    "dataset": "model_comparison",
                    "sourceId": "model-comparison",
                    "encodings": {
                        "x": {
                            "field": "model_label",
                            "type": "nominal",
                            "label": "Checkpoint",
                        },
                        "y": {
                            "field": "reward_gap",
                            "type": "quantitative",
                            "format": "number",
                            "label": "Gap de reward",
                        },
                        "tooltip": [
                            {
                                "field": "reference_mean_reward",
                                "type": "quantitative",
                                "format": "number",
                                "label": "Reward de referência",
                            },
                            {
                                "field": "surprise_mean_reward",
                                "type": "quantitative",
                                "format": "number",
                                "label": "Reward surpresa",
                            },
                        ],
                    },
                    "palette": {"kind": "diverging", "name": "gap"},
                    "labels": {"values": "auto"},
                    "referenceLines": [
                        {
                            "axis": "x",
                            "value": 0,
                            "label": "Sem mudança",
                            "color": "neutral",
                        }
                    ],
                    "settings": {
                        "orientation": "horizontal",
                        "categoryLabelPolicy": "wrap",
                        "sort": "ascending",
                    },
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "config_summary_table",
                    "title": "Resumo de generalização por configuração",
                    "dataset": "config_summary",
                    "sourceId": "config-summary",
                    "columns": [
                        {"field": "config", "label": "Config"},
                        {"field": "n_models", "label": "Modelos", "format": "number"},
                        {
                            "field": "reference_mean_reward",
                            "label": "Reward referência",
                            "format": "number",
                        },
                        {
                            "field": "surprise_mean_reward",
                            "label": "Reward surpresa",
                            "format": "number",
                        },
                        {
                            "field": "reward_gap",
                            "label": "Gap",
                            "format": "number",
                            "movement": True,
                        },
                        {
                            "field": "surprise_success_rate",
                            "label": "Sucesso surpresa",
                            "format": "percent",
                        },
                        {
                            "field": "degradation_pct",
                            "label": "Degradação",
                            "format": "percent",
                        },
                        {"field": "generalization_status", "label": "Classificação"},
                    ],
                    "defaultSort": {"field": "config", "direction": "asc"},
                },
                {
                    "id": "model_comparison_table",
                    "title": "Comparação por checkpoint",
                    "dataset": "model_comparison",
                    "sourceId": "model-comparison",
                    "columns": [
                        {"field": "config", "label": "Config"},
                        {"field": "model_seed", "label": "Seed modelo", "format": "number"},
                        {
                            "field": "reference_mean_reward",
                            "label": "Referência",
                            "format": "number",
                        },
                        {
                            "field": "surprise_mean_reward",
                            "label": "Surpresa",
                            "format": "number",
                        },
                        {
                            "field": "reward_gap",
                            "label": "Gap",
                            "format": "number",
                            "movement": True,
                        },
                        {
                            "field": "surprise_success_rate",
                            "label": "Sucesso surpresa",
                            "format": "percent",
                        },
                    ],
                    "defaultSort": {"field": "reward_gap", "direction": "asc"},
                },
            ],
            "sources": [
                {"id": source["id"], "label": source["label"]}
                for source in sources
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "model_comparison": model_rows,
                "config_summary": config_rows,
                "scenario_comparison": scenario_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Não há dados para salvar em {path}")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_results(
    output_dir: Path,
    surprise_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    episode_seeds: list[int],
    artifact: dict[str, Any],
) -> None:
    """Exporta resultados, trilha de seeds e relatório canônico."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "surprise_evaluations.csv", surprise_rows)
    _write_csv(output_dir / "model_comparison.csv", model_rows)
    _write_csv(output_dir / "config_summary.csv", config_rows)
    _write_csv(
        output_dir / "episode_seeds.csv",
        [
            {"episode_index": index, "environment_seed": seed}
            for index, seed in enumerate(episode_seeds)
        ],
    )
    with (output_dir / "artifact.json").open("w", encoding="utf-8") as json_file:
        json.dump(artifact, json_file, indent=2, ensure_ascii=False)


def package_html_report(output_dir: Path) -> Path | None:
    """Empacota o artifact como HTML quando o builder local está disponível."""
    plugin_cache = (
        Path.home()
        / ".codex"
        / "plugins"
        / "cache"
        / "openai-curated-remote"
        / "data-analytics"
    )
    candidates = sorted(
        plugin_cache.glob(
            "*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        ),
        reverse=True,
    )
    if not candidates:
        print(
            "Aviso: builder HTML não encontrado; artifact.json e CSVs foram "
            "gerados normalmente."
        )
        return None

    artifact_path = output_dir / "artifact.json"
    html_path = output_dir / "generalization_report.html"
    try:
        result = subprocess.run(
            [
                "node",
                str(candidates[0]),
                "--input",
                str(artifact_path),
                "--output",
                str(html_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        print(
            "Aviso: não foi possível empacotar o HTML; artifact.json e CSVs "
            f"foram preservados. Detalhe: {detail.strip()}"
        )
        return None
    if result.stdout.strip():
        print(f"Validação do relatório: {result.stdout.strip()}")
    return html_path


def print_summary(config_rows: list[dict[str, Any]], surprise_seed: int) -> None:
    """Mostra um relatório curto no terminal para uso na apresentação."""
    print(f"\nGENERALIZAÇÃO — SEED SURPRESA {surprise_seed}")
    print("Config | Referência | Surpresa | Gap | Classificação")
    print("-" * 78)
    for row in config_rows:
        print(
            f"  {row['config']}    | {row['reference_mean_reward']:>10.2f} | "
            f"{row['surprise_mean_reward']:>8.2f} | {row['reward_gap']:>+7.2f} | "
            f"{row['generalization_status']}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliar generalização em uma seed de ambiente não vista",
    )
    parser.add_argument("--surprise-seed", type=int, default=999)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--config", choices=["A", "B", "C"], default=None)
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="Seeds dos checkpoints treinados",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("experiments/results/eval_results.csv"),
    )
    parser.add_argument(
        "--evaluation-reward",
        choices=["produtividade", "prioridade"],
        default=DEFAULT_REWARD_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: experiments/analysis/surprise_seed/seed_<seed>",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Não tentar empacotar o relatório HTML portátil",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = (args.config,) if args.config else ("A", "B", "C")
    model_seeds = tuple(args.model_seeds)
    episode_seeds = derive_episode_seeds(args.surprise_seed, args.episodes)
    output_dir = args.output_dir or (
        Path("experiments/analysis/surprise_seed") / f"seed_{args.surprise_seed}"
    )

    reference_rows = load_reference_results(
        args.reference_results,
        configs,
        model_seeds,
        args.evaluation_reward,
    )
    surprise_rows = evaluate_surprise_models(
        configs,
        model_seeds,
        args.surprise_seed,
        episode_seeds,
        args.evaluation_reward,
    )
    model_rows, config_rows, scenario_rows = analyze_generalization(
        reference_rows, surprise_rows
    )
    artifact = build_artifact(
        model_rows,
        config_rows,
        scenario_rows,
        args.surprise_seed,
        args.episodes,
        configs,
    )
    export_results(
        output_dir,
        surprise_rows,
        model_rows,
        config_rows,
        episode_seeds,
        artifact,
    )
    html_path = None if args.skip_html else package_html_report(output_dir)
    print_summary(config_rows, args.surprise_seed)
    print(f"\nResultados salvos em: {output_dir}")
    if html_path is not None:
        print(f"Relatório HTML: {html_path}")


if __name__ == "__main__":
    main()
