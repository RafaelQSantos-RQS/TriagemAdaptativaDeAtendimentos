"""Testes rápidos da seleção e distribuição da análise qualitativa."""

from src.analysis.qualitative_analysis import (
    EpisodeTrace,
    action_distribution,
    select_cases,
)


def _episode(index: int, reward: float, action: int) -> EpisodeTrace:
    return EpisodeTrace(
        episode_index=index,
        env_seed=10_000 + index,
        total_reward=reward,
        steps=10,
        outcome="Sucesso" if reward >= 0 else "Falha",
        termination_reason="Horizonte máximo",
        total_served=8,
        total_arrivals=10,
        served_by_queue=(2, 3, 3),
        final_queue_sizes=(1, 1, 0),
        max_queue_sizes=(4, 5, 3),
        max_wait_times=(3.0, 4.0, 2.0),
        action_counts={action: 10},
        no_service_actions=2,
        step_records=[],
    )


def test_selects_three_best_successes_and_three_worst_failures() -> None:
    episodes = [
        _episode(0, 5.0, 0),
        _episode(1, 20.0, 0),
        _episode(2, 10.0, 1),
        _episode(3, 15.0, 1),
        _episode(4, -5.0, 2),
        _episode(5, -20.0, 2),
        _episode(6, -10.0, 3),
        _episode(7, -15.0, 3),
    ]

    successful, failed = select_cases(episodes)

    assert [episode.total_reward for episode in successful] == [20.0, 15.0, 10.0]
    assert [episode.total_reward for episode in failed] == [-20.0, -15.0, -10.0]


def test_action_distribution_separates_success_and_failure() -> None:
    episodes = [_episode(0, 5.0, 0), _episode(1, -5.0, 2)]

    rows = action_distribution(episodes)

    assert rows[0]["pct_success"] == 1.0
    assert rows[2]["pct_failure"] == 1.0
    assert sum(row["pct_all"] for row in rows) == 1.0
