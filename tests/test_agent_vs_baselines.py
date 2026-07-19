"""Testes da comparação entre agentes e baselines."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.analysis.agent_vs_baselines import (
    export_summary,
    load_comparison_data,
    plot_comparison,
)


SEEDS = (42, 123, 256, 789, 1024)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_results(
    tmp_path: Path, reward_config: str = "produtividade"
) -> tuple[Path, Path]:
    agents_path = tmp_path / "agents.csv"
    baselines_path = tmp_path / "baselines.csv"
    agent_rows = [
        {
            "config": config,
            "seed": seed,
            "mean_reward": index * 10 + seed / 100,
            "reward_config": reward_config,
        }
        for index, config in enumerate(("A", "B", "C"), start=1)
        for seed in SEEDS
    ]
    baseline_rows = [
        {
            "baseline": baseline,
            "seed": seed,
            "mean_reward": -(index * 10 + seed / 100),
        }
        for index, baseline in enumerate(
            ("aleatorio", "prioridade_fixa", "fila_mais_longa"), start=1
        )
        for seed in SEEDS
    ]
    _write_csv(
        agents_path,
        ["config", "seed", "mean_reward", "reward_config"],
        agent_rows,
    )
    _write_csv(
        baselines_path,
        ["baseline", "seed", "mean_reward"],
        baseline_rows,
    )
    return agents_path, baselines_path


def test_load_comparison_aggregates_six_methods_between_seeds(tmp_path: Path) -> None:
    agents_path, baselines_path = _write_results(tmp_path)

    entries = load_comparison_data(agents_path, baselines_path, SEEDS)

    assert [entry.label for entry in entries] == [
        "Config A",
        "Config B",
        "Config C",
        "Aleatório",
        "Prioridade Fixa",
        "Fila Longa",
    ]
    expected = np.asarray([10 + seed / 100 for seed in SEEDS])
    assert entries[0].mean_reward == pytest.approx(expected.mean())
    assert entries[0].std_reward == pytest.approx(expected.std(ddof=1))


def test_load_comparison_rejects_mixed_reward_functions(tmp_path: Path) -> None:
    agents_path, baselines_path = _write_results(tmp_path, reward_config="prioridade")

    with pytest.raises(ValueError, match="--evaluation-reward produtividade"):
        load_comparison_data(agents_path, baselines_path, SEEDS)


def test_load_comparison_reports_missing_seed(tmp_path: Path) -> None:
    agents_path, baselines_path = _write_results(tmp_path)

    with pytest.raises(ValueError, match="faltam resultados"):
        load_comparison_data(agents_path, baselines_path, SEEDS + (2048,))


def test_plot_and_summary_are_exported(tmp_path: Path) -> None:
    agents_path, baselines_path = _write_results(tmp_path)
    entries = load_comparison_data(agents_path, baselines_path, SEEDS)
    chart_path = tmp_path / "output" / "comparison.png"
    summary_path = tmp_path / "output" / "summary.csv"

    plot_comparison(entries, chart_path, dpi=72)
    export_summary(entries, summary_path)

    assert chart_path.stat().st_size > 0
    with summary_path.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 6
    assert rows[0]["method"] == "Config A"
    assert rows[0]["n_seeds"] == "5"
