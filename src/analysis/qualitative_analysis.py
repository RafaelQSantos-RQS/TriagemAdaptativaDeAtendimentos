"""Seleciona e disseca episódios bem-sucedidos e com falha.

O módulo avalia um único agente de forma determinística, preserva as
trajetórias passo a passo e gera datasets auditáveis mais um ``artifact.json``
para empacotamento como relatório HTML portátil.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import src.environment  # noqa: F401
from src.agents.eval import load_model
from src.agents.train import CONFIG_MAP
from src.environment import TriagemConfig

ACTION_LABELS = {
    0: "Atender maior prioridade",
    1: "Atender fila mais longa",
    2: "Encaminhar fila 0",
    3: "Encaminhar fila 1",
}
PHASES = ("Inicial", "Intermediária", "Final")
SELECTED_CASES_SQL = """
SELECT *
FROM selected_cases
ORDER BY CASE selection WHEN 'Top sucesso' THEN 0 ELSE 1 END, reward DESC
""".strip()
ACTION_DISTRIBUTION_TABLE_SQL = """
SELECT action_id, action, count_all, pct_all, count_success, pct_success,
       count_failure, pct_failure
FROM action_distribution
ORDER BY action_id
""".strip()
ACTION_DISTRIBUTION_CHART_SQL = """
WITH chart_rows AS (
    SELECT action_id, action, 'Sucesso' AS outcome, pct_success AS percentage
    FROM action_distribution
    UNION ALL
    SELECT action_id, action, 'Falha' AS outcome, pct_failure AS percentage
    FROM action_distribution
)
SELECT action, outcome, percentage
FROM chart_rows
ORDER BY action_id, CASE outcome WHEN 'Sucesso' THEN 0 ELSE 1 END
""".strip()


@dataclass
class EpisodeTrace:
    """Resumo e trajetória de um episódio de avaliação."""

    episode_index: int
    env_seed: int
    total_reward: float
    steps: int
    outcome: str
    termination_reason: str
    total_served: int
    total_arrivals: int
    served_by_queue: tuple[int, int, int]
    final_queue_sizes: tuple[int, int, int]
    max_queue_sizes: tuple[int, int, int]
    max_wait_times: tuple[float, float, float]
    action_counts: dict[int, int]
    no_service_actions: int
    step_records: list[dict[str, Any]] = field(repr=False)

    @property
    def service_rate(self) -> float:
        """Proporção dos chamados que foram resolvidos."""
        return self.total_served / self.total_arrivals if self.total_arrivals else 0.0

    @property
    def no_service_ratio(self) -> float:
        """Fração de ações que não resolveram chamado naquele passo."""
        return self.no_service_actions / self.steps if self.steps else 0.0


def _phase_for_step(step: int, max_steps: int) -> str:
    fraction = step / max_steps
    if fraction <= 1 / 3:
        return "Inicial"
    if fraction <= 2 / 3:
        return "Intermediária"
    return "Final"


def run_episode(
    model: Any,
    config: TriagemConfig,
    episode_index: int,
    env_seed: int,
) -> EpisodeTrace:
    """Executa um episódio e registra decisões, estado e resultados."""
    env = gym.make("TriagemAdaptativa-v0", config=config)
    observation, _ = env.reset(seed=env_seed)
    terminated = False
    truncated = False
    cumulative_reward = 0.0
    action_counts: Counter[int] = Counter()
    no_service_actions = 0
    step_records: list[dict[str, Any]] = []
    previous_served = 0
    previous_arrivals = 0
    previous_served_by_queue = np.zeros(config.num_queues, dtype=int)
    max_queue_sizes = np.zeros(config.num_queues, dtype=int)
    max_wait_times = np.zeros(config.num_queues, dtype=float)
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        step = len(step_records) + 1
        pre_queues = np.asarray(observation[: config.num_queues], dtype=int)
        action, _states = model.predict(observation, deterministic=True)
        action_id = int(action)
        next_observation, reward, terminated, truncated, info = env.step(action_id)
        post_queues = np.asarray(info["queue_sizes"], dtype=int)
        waits = np.asarray(info["avg_wait_times"], dtype=float)
        served_by_queue = np.asarray(info["served_by_queue"], dtype=int)
        served_delta_by_queue = served_by_queue - previous_served_by_queue
        served_delta = int(info["total_served"]) - previous_served
        arrivals_delta = int(info["total_arrivals"]) - previous_arrivals

        cumulative_reward += float(reward)
        action_counts[action_id] += 1
        if served_delta == 0:
            no_service_actions += 1
        max_queue_sizes = np.maximum(max_queue_sizes, post_queues)
        max_wait_times = np.maximum(max_wait_times, waits)

        record: dict[str, Any] = {
            "episode_index": episode_index,
            "env_seed": env_seed,
            "step": step,
            "phase": _phase_for_step(step, config.max_steps),
            "action_id": action_id,
            "action": ACTION_LABELS.get(action_id, f"Ação {action_id}"),
            "step_reward": float(reward),
            "cumulative_reward": cumulative_reward,
            "served_delta": served_delta,
            "arrivals_delta": arrivals_delta,
        }
        for queue in range(config.num_queues):
            record[f"pre_queue_{queue}"] = int(pre_queues[queue])
            record[f"post_queue_{queue}"] = int(post_queues[queue])
            record[f"wait_queue_{queue}"] = float(waits[queue])
            record[f"served_queue_{queue}"] = int(served_delta_by_queue[queue])
        step_records.append(record)

        previous_served = int(info["total_served"])
        previous_arrivals = int(info["total_arrivals"])
        previous_served_by_queue = served_by_queue.copy()
        observation = next_observation

    env.close()
    steps = len(step_records)
    total_reward = float(cumulative_reward)
    return EpisodeTrace(
        episode_index=episode_index,
        env_seed=env_seed,
        total_reward=total_reward,
        steps=steps,
        outcome="Sucesso" if total_reward >= 0 else "Falha",
        termination_reason=(
            "Sobrecarga" if steps < config.max_steps else "Horizonte máximo"
        ),
        total_served=int(info["total_served"]),
        total_arrivals=int(info["total_arrivals"]),
        served_by_queue=tuple(int(value) for value in info["served_by_queue"]),
        final_queue_sizes=tuple(int(value) for value in info["queue_sizes"]),
        max_queue_sizes=tuple(int(value) for value in max_queue_sizes),
        max_wait_times=tuple(float(value) for value in max_wait_times),
        action_counts=dict(action_counts),
        no_service_actions=no_service_actions,
        step_records=step_records,
    )


def evaluate_episodes(
    model: Any,
    config: TriagemConfig,
    episodes: int,
    evaluation_seed: int,
) -> list[EpisodeTrace]:
    """Executa uma coorte determinística de episódios."""
    return [
        run_episode(model, config, index, evaluation_seed + index)
        for index in range(episodes)
    ]


def select_cases(
    episodes: list[EpisodeTrace], n_cases: int = 3
) -> tuple[list[EpisodeTrace], list[EpisodeTrace]]:
    """Seleciona os maiores sucessos e as falhas de menor reward."""
    successful = sorted(
        (episode for episode in episodes if episode.total_reward >= 0),
        key=lambda episode: episode.total_reward,
        reverse=True,
    )
    failed = sorted(
        (episode for episode in episodes if episode.total_reward < 0),
        key=lambda episode: episode.total_reward,
    )
    if len(successful) < n_cases or len(failed) < n_cases:
        raise ValueError(
            f"A amostra contém {len(successful)} sucessos e {len(failed)} falhas; "
            f"são necessários pelo menos {n_cases} de cada. Aumente --episodes."
        )
    return successful[:n_cases], failed[:n_cases]


def action_distribution(episodes: list[EpisodeTrace]) -> list[dict[str, Any]]:
    """Calcula a distribuição de ações geral e por resultado."""
    groups = {
        "all": episodes,
        "success": [episode for episode in episodes if episode.outcome == "Sucesso"],
        "failure": [episode for episode in episodes if episode.outcome == "Falha"],
    }
    totals = {
        group: sum(sum(episode.action_counts.values()) for episode in members)
        for group, members in groups.items()
    }
    rows = []
    for action_id, label in ACTION_LABELS.items():
        row: dict[str, Any] = {"action_id": action_id, "action": label}
        for group, members in groups.items():
            count = sum(episode.action_counts.get(action_id, 0) for episode in members)
            row[f"count_{group}"] = count
            row[f"pct_{group}"] = count / totals[group] if totals[group] else 0.0
        rows.append(row)
    return rows


def diagnose_failure(episode: EpisodeTrace, config: TriagemConfig) -> str:
    """Produz um diagnóstico descritivo da principal causa observável."""
    if episode.termination_reason == "Sobrecarga":
        return "Sobrecarga simultânea das filas encerrou o episódio antecipadamente."
    if episode.no_service_ratio >= 0.15:
        return (
            f"{episode.no_service_ratio:.0%} das decisões não resolveram chamados, "
            "reduzindo produtividade e acumulando espera."
        )
    if episode.service_rate < 0.9:
        return (
            f"A taxa de resolução foi {episode.service_rate:.1%}; a demanda não "
            "atendida permaneceu tempo suficiente para gerar penalidades."
        )
    if max(episode.max_wait_times) > config.wait_threshold:
        queue = int(np.argmax(episode.max_wait_times))
        return (
            f"A fila {queue} atingiu espera máxima de "
            f"{episode.max_wait_times[queue]:.0f} passos, acima do limiar "
            f"{config.wait_threshold:.0f}, acumulando penalidades."
        )
    return "As penalidades acumuladas superaram os ganhos de atendimento."


def _dominant_action(counts: dict[int, int]) -> str:
    if not counts:
        return "Nenhuma"
    action_id = max(counts, key=counts.get)
    total = sum(counts.values())
    return f"{ACTION_LABELS[action_id]} ({counts[action_id] / total:.0%})"


def _phase_summary(episode: EpisodeTrace) -> str:
    lines = ["| Fase | Ação dominante | Reward da fase | Resolvidos |", "|---|---|---:|---:|"]
    for phase in PHASES:
        records = [record for record in episode.step_records if record["phase"] == phase]
        counts = Counter(int(record["action_id"]) for record in records)
        reward = sum(float(record["step_reward"]) for record in records)
        served = sum(int(record["served_delta"]) for record in records)
        lines.append(
            f"| {phase} | {_dominant_action(dict(counts))} | {reward:.1f} | {served} |"
        )
    return "\n".join(lines)


def describe_episode(episode: EpisodeTrace, config: TriagemConfig) -> str:
    """Gera a seção Markdown de um episódio selecionado."""
    served = "/".join(str(value) for value in episode.served_by_queue)
    final_queues = "/".join(str(value) for value in episode.final_queue_sizes)
    max_wait = "/".join(f"{value:.0f}" for value in episode.max_wait_times)
    interpretation = (
        "O episódio combinou alta cobertura da demanda com poucas decisões sem "
        "atendimento; o reward não foi consumido pelas penalidades de espera."
        if episode.outcome == "Sucesso"
        else diagnose_failure(episode, config)
    )
    return (
        f"## {episode.outcome}: episódio {episode.episode_index} "
        f"(seed {episode.env_seed})\n\n"
        f"**Reward:** {episode.total_reward:.1f} · **Resolvidos/chegadas:** "
        f"{episode.total_served}/{episode.total_arrivals} "
        f"({episode.service_rate:.1%}) · **Término:** {episode.termination_reason}\n\n"
        f"- Ação dominante: {_dominant_action(episode.action_counts)}.\n"
        f"- Resolvidos nas filas 0/1/2: {served}.\n"
        f"- Filas finais 0/1/2: {final_queues}.\n"
        f"- Espera máxima nas filas 0/1/2: {max_wait} passos.\n"
        f"- Decisões sem atendimento: {episode.no_service_ratio:.1%}.\n\n"
        f"{_phase_summary(episode)}\n\n"
        f"**Leitura do comportamento.** {interpretation}"
    )


def _episode_row(episode: EpisodeTrace, selection: str = "Não selecionado") -> dict[str, Any]:
    row = {
        "episode_index": episode.episode_index,
        "env_seed": episode.env_seed,
        "selection": selection,
        "outcome": episode.outcome,
        "reward": episode.total_reward,
        "steps": episode.steps,
        "termination_reason": episode.termination_reason,
        "total_served": episode.total_served,
        "total_arrivals": episode.total_arrivals,
        "service_rate": episode.service_rate,
        "no_service_ratio": episode.no_service_ratio,
        "dominant_action": _dominant_action(episode.action_counts),
    }
    for queue in range(3):
        row[f"served_queue_{queue}"] = episode.served_by_queue[queue]
        row[f"final_queue_{queue}"] = episode.final_queue_sizes[queue]
        row[f"max_wait_queue_{queue}"] = episode.max_wait_times[queue]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Não há linhas para exportar em {path}")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _query_rows(
    table_name: str,
    source_rows: list[dict[str, Any]],
    sql: str,
) -> list[dict[str, Any]]:
    """Materializa linhas em SQLite e executa a consulta declarada no relatório."""
    if not source_rows:
        return []
    columns = list(source_rows[0])
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            f'CREATE TABLE "{table_name}" ({quoted_columns})'  # noqa: S608
        )
        connection.executemany(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',  # noqa: S608
            [[row[column] for column in columns] for row in source_rows],
        )
        cursor = connection.execute(sql)
        result_columns = [description[0] for description in cursor.description]
        return [dict(zip(result_columns, row, strict=True)) for row in cursor.fetchall()]


def _sqlite_source(query_id: str, sql: str, description: str) -> dict[str, Any]:
    """Descreve a consulta SQLite realmente usada para montar um widget."""
    return {
        "query": {
            "engine": "sqlite",
            "id": query_id,
            "sql": sql,
            "description": description,
        }
    }


def _distribution_finding(rows: list[dict[str, Any]]) -> str:
    success_row = max(rows, key=lambda row: row["pct_success"])
    failure_row = max(rows, key=lambda row: row["pct_failure"])
    return (
        f"Nos episódios bem-sucedidos, a ação mais frequente foi "
        f"**{success_row['action']}** ({success_row['pct_success']:.1%}); nas falhas, "
        f"foi **{failure_row['action']}** ({failure_row['pct_failure']:.1%}). "
        "A diferença é descritiva e não demonstra causalidade."
    )


def build_artifact(
    episodes: list[EpisodeTrace],
    successful: list[EpisodeTrace],
    failed: list[EpisodeTrace],
    distribution: list[dict[str, Any]],
    config_name: str,
    model_seed: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    """Monta o relatório no contrato canônico de artifact.json."""
    generated_at = datetime.now(timezone.utc).isoformat()
    selected = successful + failed
    selected_source_rows = [
        _episode_row(
            episode,
            "Top sucesso" if episode.outcome == "Sucesso" else "Pior falha",
        )
        for episode in selected
    ]
    selected_rows = _query_rows(
        "selected_cases", selected_source_rows, SELECTED_CASES_SQL
    )
    distribution_table_rows = _query_rows(
        "action_distribution", distribution, ACTION_DISTRIBUTION_TABLE_SQL
    )
    distribution_chart_rows = _query_rows(
        "action_distribution", distribution, ACTION_DISTRIBUTION_CHART_SQL
    )
    success_count = sum(episode.outcome == "Sucesso" for episode in episodes)
    summary = (
        "## Resumo técnico\n\n"
        f"Foram avaliados **{len(episodes)} episódios determinísticos** da Config "
        f"{config_name}, modelo seed {model_seed}. O critério reward ≥ 0 classificou "
        f"**{success_count} sucessos** e **{len(episodes) - success_count} falhas**. "
        "Os três maiores sucessos e as três menores falhas foram dissecados; por "
        "serem extremos, eles esclarecem mecanismos de comportamento, mas não "
        "representam a frequência de um episódio típico.\n\n"
        f"{_distribution_finding(distribution)}"
    )
    definitions = (
        "## Escopo, dados e definições\n\n"
        f"- **Agente:** Config {config_name}, modelo treinado com seed {model_seed}.\n"
        f"- **Coorte:** seeds de ambiente {evaluation_seed} a "
        f"{evaluation_seed + len(episodes) - 1}.\n"
        "- **Sucesso:** reward total do episódio maior ou igual a zero.\n"
        "- **Falha:** reward total negativo.\n"
        "- **Distribuição de ações:** percentual calculado sobre todos os passos "
        "dos episódios do respectivo grupo, não sobre episódios.\n"
        "- **Ação sem atendimento:** passo em que `total_served` não aumentou."
    )
    methodology = (
        "## Método de seleção e diagnóstico\n\n"
        "A política foi executada deterministicamente. Os sucessos foram ordenados "
        "por reward decrescente e as falhas por reward crescente. Cada trajetória "
        "foi segmentada em três terços do horizonte de 100 passos. O diagnóstico "
        "de falha usa somente sinais observáveis: sobrecarga, taxa de resolução, "
        "decisões sem atendimento e espera máxima. Assim, os diagnósticos são "
        "descritivos e não causais."
    )
    limitations = (
        "## Limitações e robustez\n\n"
        "A análise usa um único checkpoint (Config A, uma seed de treinamento) e "
        "uma sequência fixa de seeds de ambiente. Os seis casos são extremos e "
        "podem exagerar diferenças. A comparação das distribuições usa todos os "
        "episódios para reduzir esse viés, mas não controla o estado enfrentado "
        "quando cada ação foi escolhida. Uma análise causal exigiria comparar "
        "ações alternativas nos mesmos estados."
    )
    next_steps = (
        "## Próximos passos recomendados\n\n"
        "1. Repetir a análise para as Configs B e C somente quando a comparação "
        "qualitativa entre objetivos for necessária.\n"
        "2. Estratificar a distribuição de ações por carga e espera das filas.\n"
        "3. Revisar estados com ação sem atendimento para separar fila vazia de "
        "limite de capacidade."
    )
    questions = (
        "## Questões em aberto\n\n"
        "- A política mudaria de ação nos mesmos estados sob outra seed de treino?\n"
        "- Quais ações reduzem espera sem sacrificar a taxa total de resolução?\n"
        "- Os padrões observados persistem sob taxas de chegada diferentes?"
    )

    blocks: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": "# Análise qualitativa do agente de triagem",
        },
        {"id": "summary", "type": "markdown", "layout": "full", "body": summary},
        {
            "id": "selected_cases_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Seis episódios revelam os mecanismos dos extremos\n\n"
                "A tabela resume os casos selecionados. As seções seguintes mostram "
                "como a política mudou entre as fases do episódio."
            ),
        },
        {
            "id": "selected_cases_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "selected_cases_table",
        },
    ]
    for index, episode in enumerate(successful, start=1):
        blocks.append(
            {
                "id": f"success_{index}",
                "type": "markdown",
                "layout": "full",
                "body": describe_episode(episode, TriagemConfig()),
            }
        )
    for index, episode in enumerate(failed, start=1):
        blocks.append(
            {
                "id": f"failure_{index}",
                "type": "markdown",
                "layout": "full",
                "body": describe_episode(episode, TriagemConfig()),
            }
        )
    blocks.extend(
        [
            {
                "id": "action_distribution",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Distribuição de ações em sucessos e falhas\n\n"
                    f"{_distribution_finding(distribution)} O gráfico facilita a "
                    "comparação visual; a tabela abaixo preserva contagens e "
                    "percentuais exatos."
                ),
            },
            {
                "id": "action_distribution_chart_block",
                "type": "chart",
                "layout": "full",
                "chartId": "action_distribution_chart",
            },
            {
                "id": "action_distribution_table_block",
                "type": "table",
                "layout": "full",
                "tableId": "action_distribution_table",
            },
            {"id": "definitions", "type": "markdown", "layout": "full", "body": definitions},
            {"id": "methodology", "type": "markdown", "layout": "full", "body": methodology},
            {"id": "limitations", "type": "markdown", "layout": "full", "body": limitations},
            {"id": "next_steps", "type": "markdown", "layout": "full", "body": next_steps},
            {"id": "questions", "type": "markdown", "layout": "full", "body": questions},
        ]
    )
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Análise qualitativa do agente de triagem",
            "generatedAt": generated_at,
            "blocks": blocks,
            "tables": [
                {
                    "id": "selected_cases_table",
                    "dataset": "selected_cases",
                    "title": "Episódios selecionados",
                    "columns": [
                        {"field": "selection", "label": "Seleção"},
                        {"field": "episode_index", "label": "Episódio"},
                        {"field": "env_seed", "label": "Seed"},
                        {"field": "reward", "label": "Reward", "format": "number"},
                        {"field": "service_rate", "label": "Taxa resolvida", "format": "percent"},
                        {"field": "dominant_action", "label": "Ação dominante"},
                    ],
                    "defaultSort": {"field": "reward", "direction": "desc"},
                    "source": _sqlite_source(
                        "selected-cases",
                        SELECTED_CASES_SQL,
                        "Ordena os seis episódios extremos selecionados para a tabela.",
                    ),
                },
                {
                    "id": "action_distribution_table",
                    "dataset": "action_distribution",
                    "title": "Tabela de distribuição de ações",
                    "columns": [
                        {"field": "action", "label": "Ação"},
                        {"field": "count_all", "label": "Total", "format": "number"},
                        {"field": "pct_all", "label": "% total", "format": "percent"},
                        {
                            "field": "count_success",
                            "label": "Sucessos",
                            "format": "number",
                        },
                        {
                            "field": "pct_success",
                            "label": "% sucessos",
                            "format": "percent",
                        },
                        {
                            "field": "count_failure",
                            "label": "Falhas",
                            "format": "number",
                        },
                        {
                            "field": "pct_failure",
                            "label": "% falhas",
                            "format": "percent",
                        },
                    ],
                    "source": _sqlite_source(
                        "action-distribution-table",
                        ACTION_DISTRIBUTION_TABLE_SQL,
                        "Recupera contagens e percentuais por ação para a tabela.",
                    ),
                },
            ],
            "charts": [
                {
                    "id": "action_distribution_chart",
                    "title": "Distribuição de ações por resultado",
                    "subtitle": "Percentual de todos os passos do respectivo grupo",
                    "intent": "comparison",
                    "question": (
                        "Como a frequência das ações difere entre episódios "
                        "bem-sucedidos e episódios com falha?"
                    ),
                    "rationale": (
                        "Barras horizontais agrupadas permitem comparar os dois "
                        "resultados e acomodam rótulos longos das ações."
                    ),
                    "comparisonContext": {
                        "denominator": "Todos os passos dos episódios do grupo",
                        "unit": "Percentual dos passos",
                    },
                    "type": "horizontalBar",
                    "dataset": "action_distribution_chart",
                    "encodings": {
                        "x": {
                            "field": "action",
                            "type": "nominal",
                            "label": "Ação",
                        },
                        "y": {
                            "field": "percentage",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Percentual dos passos",
                        },
                        "color": {
                            "field": "outcome",
                            "type": "nominal",
                            "label": "Resultado",
                        },
                    },
                    "palette": {"kind": "categorical", "name": "analysis"},
                    "legend": {"position": "bottom", "title": "Resultado"},
                    "labels": {"values": "auto"},
                    "settings": {
                        "groupMode": "grouped",
                        "orientation": "horizontal",
                        "categoryLabelPolicy": "wrap",
                        "sort": "none",
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                    "source": _sqlite_source(
                        "action-distribution-chart",
                        ACTION_DISTRIBUTION_CHART_SQL,
                        "Converte os percentuais de sucesso e falha para formato longo.",
                    ),
                }
            ],
            "cards": [],
            "sources": [],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "selected_cases": selected_rows,
                "action_distribution": distribution_table_rows,
                "action_distribution_chart": distribution_chart_rows,
            },
            "accessIssues": [],
        },
        "sources": [],
    }


def export_analysis(
    output_dir: Path,
    episodes: list[EpisodeTrace],
    successful: list[EpisodeTrace],
    failed: list[EpisodeTrace],
    distribution: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> None:
    """Exporta dados detalhados e o artifact.json do relatório."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_lookup = {
        episode.episode_index: (
            "Top sucesso" if episode.outcome == "Sucesso" else "Pior falha"
        )
        for episode in successful + failed
    }
    _write_csv(
        output_dir / "episode_summary.csv",
        [_episode_row(episode) for episode in episodes],
    )
    _write_csv(
        output_dir / "selected_episodes.csv",
        [
            _episode_row(episode, selected_lookup[episode.episode_index])
            for episode in successful + failed
        ],
    )
    _write_csv(
        output_dir / "selected_episode_steps.csv",
        [record for episode in successful + failed for record in episode.step_records],
    )
    _write_csv(output_dir / "action_distribution.csv", distribution)
    with (output_dir / "artifact.json").open("w", encoding="utf-8") as json_file:
        json.dump(artifact, json_file, indent=2, ensure_ascii=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Gerar análise qualitativa do agente")
    parser.add_argument("--config", choices=["A", "B", "C"], default="A")
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis/qualitative"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Executa a análise qualitativa e exporta o relatório canônico."""
    args = parse_args(argv)
    algo = CONFIG_MAP[args.config]["algo"]
    reward_config = CONFIG_MAP[args.config]["reward_config"]
    config = TriagemConfig(reward_config=reward_config)
    model = load_model(args.config, algo, args.model_seed)
    episodes = evaluate_episodes(model, config, args.episodes, args.evaluation_seed)
    successful, failed = select_cases(episodes, args.n_cases)
    distribution = action_distribution(episodes)
    artifact = build_artifact(
        episodes,
        successful,
        failed,
        distribution,
        args.config,
        args.model_seed,
        args.evaluation_seed,
    )
    export_analysis(
        args.output_dir,
        episodes,
        successful,
        failed,
        distribution,
        artifact,
    )
    print(f"Análise exportada: {args.output_dir}")
    print(
        f"Selecionados {len(successful)} sucessos e {len(failed)} falhas "
        f"entre {len(episodes)} episódios."
    )


if __name__ == "__main__":
    main()
