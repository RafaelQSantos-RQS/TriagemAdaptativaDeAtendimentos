"""Testes da agregação e visualização das curvas de aprendizado."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.analysis.learning_curves import (
    aggregate_config,
    export_summary,
    load_seed_curve,
    plot_comparison,
    plot_config_curve,
)


def _write_evaluations(
    root: Path,
    config: str,
    seed: int,
    timesteps: list[int],
    results: list[list[float]],
) -> Path:
    path = root / f"config_{config.lower()}" / f"seed_{seed:03d}" / "evaluations.npz"
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        timesteps=np.asarray(timesteps),
        results=np.asarray(results, dtype=float),
        ep_lengths=np.ones_like(results, dtype=int),
    )
    return path


def test_load_seed_curve_averages_evaluation_episodes(tmp_path: Path) -> None:
    path = _write_evaluations(
        tmp_path,
        "A",
        42,
        [10_000, 20_000],
        [[1.0, 3.0], [4.0, 8.0]],
    )

    timesteps, rewards = load_seed_curve(path)

    np.testing.assert_array_equal(timesteps, [10_000, 20_000])
    np.testing.assert_allclose(rewards, [2.0, 6.0])


def test_aggregate_config_uses_mean_and_sample_std_between_seeds(
    tmp_path: Path,
) -> None:
    _write_evaluations(tmp_path, "A", 42, [10_000, 20_000], [[1, 3], [5, 7]])
    _write_evaluations(tmp_path, "A", 123, [10_000, 20_000], [[3, 5], [9, 11]])

    curve = aggregate_config(tmp_path, "A", seeds=(42, 123))

    np.testing.assert_allclose(curve.mean_reward, [3.0, 8.0])
    np.testing.assert_allclose(curve.std_reward, [np.sqrt(2), np.sqrt(8)])
    assert curve.seeds == (42, 123)


def test_aggregate_config_reports_missing_seeds(tmp_path: Path) -> None:
    _write_evaluations(tmp_path, "A", 42, [10_000], [[1, 2]])

    with pytest.raises(FileNotFoundError, match="--allow-partial"):
        aggregate_config(tmp_path, "A", seeds=(42, 123))


def test_plots_and_summary_are_exported(tmp_path: Path) -> None:
    _write_evaluations(tmp_path, "A", 42, [10_000, 20_000], [[1, 3], [5, 7]])
    _write_evaluations(tmp_path, "A", 123, [10_000, 20_000], [[3, 5], [9, 11]])
    curve = aggregate_config(tmp_path, "A", seeds=(42, 123))

    individual = tmp_path / "output" / "config_a.png"
    comparison = tmp_path / "output" / "comparison.png"
    summary = tmp_path / "output" / "summary.csv"
    plot_config_curve(curve, individual, dpi=72)
    plot_comparison([curve], comparison, dpi=72)
    export_summary([curve], summary)

    assert individual.stat().st_size > 0
    assert comparison.stat().st_size > 0
    with summary.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0] == {
        "config": "A",
        "timestep": "10000",
        "mean_reward": "3.000000",
        "std_reward": f"{np.sqrt(2):.6f}",
        "n_seeds": "2",
    }
