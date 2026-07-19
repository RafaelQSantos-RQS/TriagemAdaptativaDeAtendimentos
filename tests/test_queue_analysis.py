"""Testes da análise de chamados resolvidos por fila."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.analysis.queue_analysis import (
    export_summary,
    load_queue_data,
    plot_queue_comparison,
)
from src.agents.eval import evaluate_agent
from src.baselines.run import evaluate_baseline

SEEDS = (42, 123, 256, 789, 1024)


class _PriorityAgent:
    """Agente mínimo que sempre seleciona a ação de maior prioridade."""

    def predict(self, _observation, deterministic=True):
        return 0, None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_results(tmp_path: Path) -> tuple[Path, Path]:
    metric_fields = [f"mean_served_queue_{queue}" for queue in range(3)]
    agents_path = tmp_path / "agents.csv"
    baselines_path = tmp_path / "baselines.csv"
    agent_rows = [
        {
            "config": config,
            "seed": seed,
            **{
                field: index * 10 + queue * 3 + seed / 1000
                for queue, field in enumerate(metric_fields)
            },
        }
        for index, config in enumerate(("A", "B", "C"), start=1)
        for seed in SEEDS
    ]
    baseline_rows = [
        {
            "baseline": baseline,
            "seed": seed,
            **{
                field: index * 5 + queue * 2 + seed / 1000
                for queue, field in enumerate(metric_fields)
            },
        }
        for index, baseline in enumerate(
            ("aleatorio", "prioridade_fixa", "fila_mais_longa"), start=1
        )
        for seed in SEEDS
    ]
    _write_csv(agents_path, ["config", "seed", *metric_fields], agent_rows)
    _write_csv(baselines_path, ["baseline", "seed", *metric_fields], baseline_rows)
    return agents_path, baselines_path


def test_load_queue_data_aggregates_six_methods_and_three_queues(
    tmp_path: Path,
) -> None:
    agents_path, baselines_path = _write_results(tmp_path)

    entries = load_queue_data(agents_path, baselines_path, SEEDS)

    assert len(entries) == 6
    assert entries[0].label == "Config A"
    assert entries[0].mean_served.shape == (3,)
    expected = np.asarray([10 + seed / 1000 for seed in SEEDS])
    assert entries[0].mean_served[0] == pytest.approx(expected.mean())
    assert entries[0].std_served[0] == pytest.approx(expected.std(ddof=1))


def test_load_queue_data_reports_missing_metrics(tmp_path: Path) -> None:
    agents_path, baselines_path = _write_results(tmp_path)
    rows = list(csv.DictReader(agents_path.open(encoding="utf-8")))
    reduced_rows = [{"config": row["config"], "seed": row["seed"]} for row in rows]
    _write_csv(agents_path, ["config", "seed"], reduced_rows)

    with pytest.raises(ValueError, match="métricas por fila"):
        load_queue_data(agents_path, baselines_path, SEEDS)


@pytest.mark.parametrize(
    "result",
    [
        evaluate_agent(_PriorityAgent(), episodes=2, seed=42),
        evaluate_baseline("prioridade_fixa", episodes=2, seed=42),
    ],
)
def test_evaluations_collect_mean_served_by_queue(result: dict[str, float]) -> None:
    per_queue = [result[f"mean_served_queue_{queue}"] for queue in range(3)]

    assert sum(per_queue) * 2 == pytest.approx(result["total_served"])


def test_plot_and_summary_are_exported(tmp_path: Path) -> None:
    agents_path, baselines_path = _write_results(tmp_path)
    entries = load_queue_data(agents_path, baselines_path, SEEDS)
    chart_path = tmp_path / "output" / "queues.png"
    summary_path = tmp_path / "output" / "queues.csv"

    plot_queue_comparison(entries, chart_path, dpi=72)
    export_summary(entries, summary_path)

    assert chart_path.stat().st_size > 0
    with summary_path.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 18
    assert rows[0]["method"] == "Config A"
    assert rows[0]["queue"] == "0"
