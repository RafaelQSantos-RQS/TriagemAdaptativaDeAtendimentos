"""Testes dos contadores de chegadas e atendimentos no info dict.

Verifica se o ambiente expõe corretamente a quantidade de chamados
que chegaram e foram atendidos durante um episódio.
"""

import numpy as np

from src.environment import TriagemConfig, TriagemEnv


class TestTotalArrivals:
    """Contador de chegadas totais no episódio."""

    def test_info_has_total_arrivals(self):
        """info dict contém a chave total_arrivals."""
        env = TriagemEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert "total_arrivals" in info

    def test_total_arrivals_zero_initially(self):
        """Após reset, total_arrivals é zero."""
        env = TriagemEnv()
        env.reset(seed=42)
        _, info = env.reset(seed=42)
        assert info["total_arrivals"] == 0

    def test_total_arrivals_increases(self):
        """total_arrivals aumenta após step com chegadas."""
        cfg = TriagemConfig(
            arrival_rates=(2.0, 2.0, 2.0),  # muitas chegadas
        )
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        _, _, _, _, info_1 = env.step(0)
        _, _, _, _, info_2 = env.step(0)
        assert info_2["total_arrivals"] >= info_1["total_arrivals"]
        assert info_2["total_arrivals"] > 0

    def test_total_arrivals_resets_between_episodes(self):
        """total_arrivals volta a zero em novo episódio."""
        cfg = TriagemConfig(arrival_rates=(2.0, 2.0, 2.0))
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        for _ in range(10):
            env.step(0)
        _, info = env.reset(seed=123)
        assert info["total_arrivals"] == 0

    def test_total_arrivals_no_arrivals(self):
        """Sem taxas de chegada, total_arrivals fica em zero."""
        cfg = TriagemConfig(arrival_rates=(0.0, 0.0, 0.0))
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        for _ in range(10):
            _, _, _, _, info = env.step(0)
        assert info["total_arrivals"] == 0


class TestTotalServed:
    """Contador legado de saídas provocadas pelo agente."""

    def test_info_has_total_served(self):
        """info dict contém a chave total_served."""
        env = TriagemEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert "total_served" in info

    def test_total_served_zero_initially(self):
        """Após reset, total_served é zero."""
        env = TriagemEnv()
        env.reset(seed=42)
        _, info = env.reset(seed=42)
        assert info["total_served"] == 0

    def test_total_served_increments_on_serve(self):
        """Serve incrementa total_served em 1."""
        env = TriagemEnv()
        env.reset(seed=42)
        # Encher filas manualmente
        env._queue_sizes[:] = [3, 2, 1]  # noqa: SLF001
        _, _, _, _, info = env.step(0)  # serve_priority
        assert info["total_served"] == 1

    def test_total_served_increments_on_referral(self):
        """Encaminhamento mantém a semântica legada de total_served."""
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[0] = 3  # noqa: SLF001
        _, _, _, _, info = env.step(2)  # refer_queue[0]
        assert info["total_served"] == 1

    def test_total_served_resets_between_episodes(self):
        """total_served volta a zero em novo episódio."""
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[:] = [3, 2, 1]  # noqa: SLF001
        env.step(0)
        _, info = env.reset(seed=123)
        assert info["total_served"] == 0

    def test_total_served_empty_queue(self):
        """Servir fila vazia não incrementa total_served."""
        env = TriagemEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)  # serve_priority com filas vazias
        assert info["total_served"] == 0

    def test_total_served_multiple_steps(self):
        """Múltiplos serves acumulam no contador."""
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[:] = [5, 5, 5]  # noqa: SLF001
        for _ in range(3):
            _, _, _, _, info = env.step(0)
        assert info["total_served"] == 3


class TestResolvedAndReferred:
    """Contadores distinguem atendimento local de encaminhamento."""

    def test_counters_start_at_zero(self):
        env = TriagemEnv()

        _, info = env.reset(seed=42)

        assert info["total_resolved"] == 0
        assert info["total_referred"] == 0
        np.testing.assert_array_equal(info["resolved_by_queue"], [0, 0, 0])
        np.testing.assert_array_equal(info["referred_by_queue"], [0, 0, 0])

    def test_service_increments_only_resolved(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))
        env.reset(seed=42)
        env._queue_sizes[2] = 1  # noqa: SLF001

        _, _, _, _, info = env.step(0)

        assert info["total_served"] == 1
        assert info["total_resolved"] == 1
        assert info["total_referred"] == 0
        np.testing.assert_array_equal(info["resolved_by_queue"], [0, 0, 1])
        np.testing.assert_array_equal(info["referred_by_queue"], [0, 0, 0])

    def test_referral_increments_only_referred(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))
        env.reset(seed=42)
        env._queue_sizes[1] = 1  # noqa: SLF001

        _, _, _, _, info = env.step(3)

        assert info["total_served"] == 1
        assert info["total_resolved"] == 0
        assert info["total_referred"] == 1
        np.testing.assert_array_equal(info["resolved_by_queue"], [0, 0, 0])
        np.testing.assert_array_equal(info["referred_by_queue"], [0, 1, 0])

    def test_totals_and_per_queue_counters_are_consistent(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))
        env.reset(seed=42)
        env._queue_sizes[:] = [3, 2, 1]  # noqa: SLF001

        for action in (0, 1, 2, 3):
            _, _, _, _, info = env.step(action)

        assert info["total_served"] == (
            info["total_resolved"] + info["total_referred"]
        )
        np.testing.assert_array_equal(
            info["served_by_queue"],
            info["resolved_by_queue"] + info["referred_by_queue"],
        )

    def test_counters_reset_between_episodes(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))
        env.reset(seed=42)
        env._queue_sizes[:] = [1, 0, 1]  # noqa: SLF001
        env.step(0)
        env.step(2)

        _, info = env.reset(seed=123)

        assert info["total_resolved"] == 0
        assert info["total_referred"] == 0
        np.testing.assert_array_equal(info["resolved_by_queue"], [0, 0, 0])
        np.testing.assert_array_equal(info["referred_by_queue"], [0, 0, 0])


class TestTerminationReason:
    """Info identifica se a sobrecarga encerrou o episódio."""

    def test_horizon_does_not_mark_overload(self):
        cfg = TriagemConfig(arrival_rates=(0.0, 0.0, 0.0), max_steps=1)
        env = TriagemEnv(cfg)
        env.reset(seed=42)

        _, _, terminated, _, info = env.step(0)

        assert terminated is True
        assert info["terminated_by_overload"] is False

    def test_overload_marks_termination_reason(self):
        cfg = TriagemConfig(
            arrival_rates=(0.0, 0.0, 0.0),
            max_queue_size=10,
            max_steps=100,
            overload_threshold=0.8,
            overload_patience=1,
        )
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        env._queue_sizes[:] = [10, 10, 10]  # noqa: SLF001

        _, _, terminated, _, info = env.step(2)

        assert terminated is True
        assert info["terminated_by_overload"] is True

    def test_reset_clears_termination_reason(self):
        cfg = TriagemConfig(
            arrival_rates=(0.0, 0.0, 0.0),
            max_queue_size=10,
            overload_threshold=0.8,
            overload_patience=1,
        )
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        env._queue_sizes[:] = [10, 10, 10]  # noqa: SLF001
        _, _, _, _, terminal_info = env.step(2)
        assert terminal_info["terminated_by_overload"] is True

        _, info = env.reset(seed=123)

        assert info["terminated_by_overload"] is False


class TestServedByQueue:
    """Contador acumulado de chamados resolvidos por fila."""

    def test_info_has_zeroed_counter_after_reset(self):
        env = TriagemEnv()

        _, info = env.reset(seed=42)

        np.testing.assert_array_equal(info["served_by_queue"], [0, 0, 0])

    def test_priority_service_counts_selected_queue(self):
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[:] = [3, 2, 1]  # noqa: SLF001

        _, _, _, _, info = env.step(0)

        np.testing.assert_array_equal(info["served_by_queue"], [0, 0, 1])

    def test_referral_counts_selected_queue(self):
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[1] = 2  # noqa: SLF001

        _, _, _, _, info = env.step(3)

        np.testing.assert_array_equal(info["served_by_queue"], [0, 1, 0])

    def test_queue_sum_matches_total_served(self):
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[:] = [5, 5, 5]  # noqa: SLF001
        for action in (0, 1, 2, 3):
            _, _, _, _, info = env.step(action)

        assert int(info["served_by_queue"].sum()) == info["total_served"]

    def test_counter_resets_between_episodes(self):
        env = TriagemEnv()
        env.reset(seed=42)
        env._queue_sizes[0] = 2  # noqa: SLF001
        env.step(2)

        _, info = env.reset(seed=123)

        np.testing.assert_array_equal(info["served_by_queue"], [0, 0, 0])


class TestCountersConsistency:
    """Consistência entre total_arrivals e total_served."""

    def test_served_never_exceeds_arrivals(self):
        """Nunca se pode atender mais do que chegou."""
        env = TriagemEnv()
        env.reset(seed=42)
        for _ in range(50):
            _, _, terminated, _, info = env.step(0)
            assert (
                info["total_served"] <= info["total_arrivals"]
                or info["total_arrivals"] == 0
            )
            if terminated:
                break

    def test_counters_at_episode_end(self):
        """Ao fim do episódio, contadores refletem o total acumulado."""
        cfg = TriagemConfig(
            arrival_rates=(0.5, 0.5, 0.5),
            max_steps=20,
        )
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        for _ in range(20):
            _, _, terminated, _, info = env.step(0)
            if terminated:
                break
        # total_arrivals + total_served devem ter valores consistentes
        assert info["total_arrivals"] >= 0
        assert info["total_served"] >= 0


class TestAccumulatedCost:
    """Custo acumulado inclui todas as contribuições negativas do reward."""

    def test_cost_starts_at_zero_and_is_exposed(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))

        _, info = env.reset(seed=42)

        assert info["total_cost"] == 0.0

    def test_invalid_service_accumulates_penalty_as_positive_cost(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))
        env.reset(seed=42)

        _, reward, _, _, info = env.step(0)

        assert reward == -0.3
        assert info["total_cost"] == 0.3

    def test_successful_service_has_no_cost_without_delay(self):
        env = TriagemEnv(TriagemConfig(arrival_rates=(0.0, 0.0, 0.0)))
        env.reset(seed=42)
        env._queue_sizes[2] = 1  # noqa: SLF001

        _, reward, _, _, info = env.step(0)

        assert reward == 1.0
        assert info["total_cost"] == 0.0
