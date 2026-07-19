"""Testes rápidos da análise de seed surpresa usando somente a Config A."""

from __future__ import annotations

import pytest

from src.agents.train import SEEDS
from src.analysis.surprise_seed import (
    analyze_generalization,
    build_artifact,
    derive_episode_seeds,
)


def _evaluation(
    scenario: str,
    model_seed: int,
    mean_reward: float,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "config": "A",
        "algo": "ppo",
        "reward_config": "produtividade",
        "model_seed": model_seed,
        "evaluation_seed": model_seed if scenario == "reference" else 999,
        "episodes": 10,
        "mean_reward": mean_reward,
        "std_reward": 2.0,
        "success_rate": 0.5 if scenario == "reference" else 0.6,
        "mean_steps": 100.0,
        "mean_cost": 20.0,
        "service_rate": 0.8 if scenario == "reference" else 0.85,
    }


def test_episode_seeds_are_deterministic_unique_and_unseen() -> None:
    first = derive_episode_seeds(999, 100)
    second = derive_episode_seeds(999, 100)

    assert first == second
    assert len(set(first)) == 100
    assert set(first).isdisjoint(SEEDS)


def test_rejects_a_training_seed_as_surprise() -> None:
    with pytest.raises(ValueError, match="pertence às seeds de treino"):
        derive_episode_seeds(42, 10)


def test_config_a_generalization_and_report_contract() -> None:
    reference = [
        _evaluation("reference", seed, 10.0 + index)
        for index, seed in enumerate(SEEDS)
    ]
    surprise = [
        _evaluation("surprise", seed, 8.0 + index)
        for index, seed in enumerate(SEEDS)
    ]

    model_rows, config_rows, scenario_rows = analyze_generalization(
        reference, surprise
    )
    artifact = build_artifact(
        model_rows,
        config_rows,
        scenario_rows,
        surprise_seed=999,
        episodes=10,
        configs=("A",),
    )

    assert len(model_rows) == 5
    assert config_rows[0]["reward_gap"] == pytest.approx(-2.0)
    assert config_rows[0]["success_rate_gap"] == pytest.approx(0.1)
    assert len(artifact["manifest"]["charts"]) == 2
    assert len(artifact["manifest"]["tables"]) == 2
    assert artifact["snapshot"]["datasets"]["config_summary"] == config_rows
