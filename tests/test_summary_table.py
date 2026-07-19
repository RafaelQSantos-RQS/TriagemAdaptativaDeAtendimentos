"""Testes rápidos da tabela-resumo final."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.analysis.summary_table import build_summary_table, export_summary

SEEDS = (42, 123, 256, 789, 1024)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summary_has_all_six_methods_and_weighted_success_rate(tmp_path: Path) -> None:
    agents_path = tmp_path / "agents.csv"
    baselines_path = tmp_path / "baselines.csv"
    common = {
        "episodes": 100,
        "mean_steps": 90.0,
        "mean_cost": 12.0,
        "total_served": 80,
        "total_arrivals": 100,
    }
    agent_rows = [
        {
            "config": config,
            "seed": seed,
            "reward_config": "produtividade",
            "mean_reward": 10 + index,
            **common,
        }
        for config in ("A", "B", "C")
        for index, seed in enumerate(SEEDS)
    ]
    baseline_rows = [
        {
            "baseline": baseline,
            "seed": seed,
            "mean_reward": -10 - index,
            **common,
        }
        for baseline in ("aleatorio", "prioridade_fixa", "fila_mais_longa")
        for index, seed in enumerate(SEEDS)
    ]
    _write(agents_path, agent_rows)
    _write(baselines_path, baseline_rows)

    rows = build_summary_table(agents_path, baselines_path, SEEDS)
    csv_path, markdown_path = export_summary(rows, tmp_path / "output")

    assert len(rows) == 6
    assert rows[0].success_rate == pytest.approx(0.8)
    assert rows[0].std_reward_between_seeds == pytest.approx(1.58113883)
    assert csv_path.stat().st_size > 0
    assert "Fila Mais Longa" in markdown_path.read_text(encoding="utf-8")
