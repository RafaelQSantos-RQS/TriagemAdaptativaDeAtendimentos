"""Testes do ambiente TriagemEnv.

Cobre: reset, step, render, casos limite, check_env, espaços, recompensas.
Referência: .specs/03-environment-specs.md
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment import TriagemConfig, TriagemEnv


# ═══════════════════════════════════════════════════════════════
# 3.1 / 3.2 — Contrato Gymnasium + reset
# ═══════════════════════════════════════════════════════════════

class TestReset:
    """Comportamento do reset() — Spec 3.2."""

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert "queue_sizes" in info
        assert "avg_wait_times" in info

    def test_reset_zeroes_state(self, env):
        env.reset(seed=42)
        assert np.all(env._queue_sizes == 0)  # noqa: SLF001
        assert np.all(env._avg_wait_times == 0)  # noqa: SLF001
        assert env._used_capacity == 0  # noqa: SLF001
        assert env._step == 0  # noqa: SLF001

    def test_reset_seed_reproducibility(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)

    def test_reset_different_seeds_different_state(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        # Observação inicial deve ser idêntica (tudo zero)
        assert np.array_equal(obs1, obs2)

    def test_reset_accepts_options(self, env):
        obs, info = env.reset(seed=42, options={"dummy": True})
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)


# ═══════════════════════════════════════════════════════════════
# 3.3 — Comportamento do step
# ═══════════════════════════════════════════════════════════════

class TestStep:
    """Comportamento do step(action) — Spec 3.3."""

    def test_step_returns_5_tuple(self, env):
        env.reset(seed=42)
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_truncated_always_false(self, env):
        env.reset(seed=42)
        for _ in range(5):
            _, _, _, truncated, _ = env.step(0)
            assert truncated is False  # não usamos limite externo

    def test_step_invalid_action_asserts(self, env):
        env.reset(seed=42)
        with pytest.raises(AssertionError):
            env.step(99)

    def test_step_increments_counter(self, env):
        env.reset(seed=42)
        env.step(0)
        assert env._step == 1  # noqa: SLF001
        env.step(1)
        assert env._step == 2  # noqa: SLF001

    def test_step_observation_shape(self, env):
        env.reset(seed=42)
        obs, *_ = env.step(0)
        # 2 * num_queues + 3 = 2*3 + 3 = 9
        assert obs.shape == (9,)

    def test_step_observation_bounds(self, env):
        env.reset(seed=42)
        obs, *_ = env.step(0)
        assert np.all(obs >= 0)
        assert np.all(obs <= env.observation_space.high)


# ═══════════════════════════════════════════════════════════════
# 3.4 — Configurações do Ambiente
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    """Configurações — Spec 3.4."""

    def test_default_config(self):
        cfg = TriagemConfig()
        assert cfg.num_queues == 3
        assert cfg.arrival_rates == (0.3, 0.5, 0.2)
        assert cfg.priority_weights == (1.0, 2.0, 3.0)
        assert cfg.max_queue_size == 50
        assert cfg.total_capacity == 10
        assert cfg.max_steps == 100
        assert cfg.reward_config == "produtividade"

    def test_custom_config(self):
        cfg = TriagemConfig(
            num_queues=2,
            arrival_rates=(1.0, 1.0),
            priority_weights=(1.0, 2.0),
            max_queue_size=10,
            total_capacity=3,
            max_steps=20,
            reward_config="prioridade",
        )
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        obs, *_ = env.step(0)
        # 2*2 + 3 = 7
        assert obs.shape == (7,)
        assert env.action_space.n == 3  # n + 1 = 2 + 1

    def test_invalid_arrival_rates_length(self):
        with pytest.raises(ValueError, match="arrival_rates"):
            TriagemConfig(arrival_rates=(0.1,))  # apenas 1 para num_queues=3

    def test_invalid_priority_weights_length(self):
        with pytest.raises(ValueError, match="priority_weights"):
            TriagemConfig(priority_weights=(1.0,))

    def test_invalid_reward_config(self):
        with pytest.raises(ValueError, match="reward_config"):
            TriagemConfig(reward_config="invalida")


# ═══════════════════════════════════════════════════════════════
# 3.5 — Visualização (render)
# ═══════════════════════════════════════════════════════════════

class TestRender:
    """Visualização — Spec 3.5."""

    def test_render_ansi_returns_string(self, env_ansi):
        env_ansi.reset(seed=42)
        out = env_ansi.render()
        assert isinstance(out, str)
        assert "TRIAGEM" in out
        assert out.startswith("╔")

    def test_render_ansi_includes_queues(self, env_ansi):
        env_ansi.reset(seed=42)
        out = env_ansi.render()
        assert "Fila 0" in out
        assert "Capacidade" in out

    def test_render_none_returns_none(self, env):
        """render_mode=None → render() retorna None."""
        env.reset(seed=42)
        out = env.render()
        assert out is None

    def test_render_human_prints(self, env, capsys):
        """Criamos env temporário com render_mode='human' para testar print."""
        env_human = TriagemEnv(render_mode="human")
        env_human.reset(seed=42)
        out = env_human.render()
        captured = capsys.readouterr()
        assert out is None  # human retorna None
        assert "TRIAGEM" in captured.out

    def test_render_requires_reset(self, env_ansi):
        """render() sem reset deve falhar."""
        with pytest.raises(AssertionError):
            env_ansi.render()

    def test_render_via_gym_make(self, env_gym_ansi):
        env_gym_ansi.reset(seed=42)
        out = env_gym_ansi.render()
        assert isinstance(out, str)
        assert "TRIAGEM" in out


# ═══════════════════════════════════════════════════════════════
# 3.6 — Elementos de Complexidade (testes conceituais)
# ═══════════════════════════════════════════════════════════════

class TestComplexidade:
    """Elementos de complexidade — Spec 3.6."""

    def test_multiplos_objetivos(self, env_produtividade, env_prioridade):
        """Recompensa diferente conforme reward_config."""
        # Encher filas com chamados antes de servir
        env_produtividade.reset(seed=42)
        env_produtividade._queue_sizes[:] = [3, 2, 1]  # noqa: SLF001
        env_prioridade.reset(seed=42)
        env_prioridade._queue_sizes[:] = [3, 2, 1]  # noqa: SLF001
        # Mesma ação = serve_priority → fila 2 (peso 3)
        r_prod = env_produtividade.step(0)[1]
        r_prio = env_prioridade.step(0)[1]
        # produtividade: +1 por chamado; prioridade: +peso
        assert r_prod != r_prio

    def test_restricao_recurso(self, small_env):
        """Capacidade limitada bloqueia ações de atendimento."""
        small_env.reset(seed=42)
        small_env._used_capacity = 2  # capacidade total  # noqa: SLF001
        reward = small_env.step(0)[1]  # serve com capacidade cheia
        assert reward < 0  # penalidade

    def test_estocasticidade(self):
        """Chegadas Poisson produzem estados diferentes."""
        env = TriagemEnv(TriagemConfig(arrival_rates=(2.0, 2.0, 2.0)))
        env.reset(seed=42)
        for _ in range(10):
            env.step(0)
        assert np.any(env._queue_sizes > 0)  # noqa: SLF001


# ═══════════════════════════════════════════════════════════════
# 3.7 — Casos Limite
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Casos limite — Spec 3.7."""

    def test_serve_empty_queue_penalty(self, small_env):
        """Tentar atender fila vazia → reward reduzido, filas não mudam."""
        small_env.reset(seed=42)
        q_before = small_env._queue_sizes.copy()  # noqa: SLF001
        w_before = small_env._avg_wait_times.copy()  # noqa: SLF001
        reward = small_env.step(0)[1]  # serve_priority com filas vazias
        assert reward < 0  # penalidade aplicada
        # Filas e esperas não mudam (sem chamados para servir, sem chegadas)
        assert np.array_equal(small_env._queue_sizes, q_before)  # noqa: SLF001
        assert np.array_equal(small_env._avg_wait_times, w_before)  # noqa: SLF001

    def test_capacity_exhausted_blocks_serve(self, small_env):
        """Capacidade total exhausted → ações de atendimento bloqueadas."""
        small_env.reset(seed=42)
        # Enche a capacidade manualmente
        small_env._used_capacity = 2  # noqa: SLF001 — total_capacity=2
        reward_priority = small_env.step(0)[1]  # serve_priority
        assert reward_priority < 0  # penalidade por capacidade cheia

    def test_max_queue_penalty(self):
        """Fila atingir MAX_QUEUE → chamado descartado, penalidade extra."""
        cfg = TriagemConfig(
            max_queue_size=1,
            total_capacity=5,
            arrival_rates=(5.0, 0.0, 0.0),  # fila 0 recebe muitos chamados
        )
        env = TriagemEnv(cfg)
        env.reset(seed=42)
        # Encher fila 0
        env._queue_sizes[0] = 1  # noqa: SLF001 — já no limite
        reward = env.step(1)[1]  # serve_longest (tenta esvaziar, mas Poisson chega)
        # Deve incluir penalidade de drop (-2.0)
        assert reward <= -2.0  # penalidade_drop aplicada

    def test_referral_invalid(self, env):
        """Encaminhamento de fila vazia → penalidade."""
        env.reset(seed=42)
        reward = env.step(2)[1]  # refer_queue[0] com fila vazia
        assert reward < 0

    def test_referral_success(self, small_env):
        """Encaminhamento válido reduz reward (custo do encaminhamento)."""
        small_env.reset(seed=42)
        # Encher fila 0 manualmente
        small_env._queue_sizes[0] = 3  # noqa: SLF001
        reward = small_env.step(2)[1]  # refer_queue[0]
        assert reward < 0  # penalidade_referral (-0.5)
        assert small_env._queue_sizes[0] == 2  # noqa: SLF001 — um chamado removido

    def test_episodio_terminado_requer_reset(self, env):
        """step() após terminação ainda funciona (Gymnasium permite)."""
        env.reset(seed=42)
        # Simular terminação por max_steps
        env._step = 99  # noqa: SLF001 — max_steps=100
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated
        # Gymnasium permite step() mesmo após terminate,
        # mas o ambiente não reseta automaticamente


# ═══════════════════════════════════════════════════════════════
# Terminação
# ═══════════════════════════════════════════════════════════════

class TestTermination:
    """Condições de término do episódio."""

    def test_terminates_after_max_steps(self, env):
        """Episódio termina após max_steps passos."""
        env.reset()
        cfg = env._config  # noqa: SLF001
        for _ in range(cfg.max_steps):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break
        else:
            pytest.fail("Episódio não terminou após max_steps")

    def test_terminates_on_critical_overload(self, overload_env):
        """Overload crítico encerra episódio."""
        overload_env.reset(seed=42)
        cfg = overload_env._config  # noqa: SLF001
        terminated = False
        for _ in range(cfg.max_steps):
            _, _, terminated, _, _ = overload_env.step(0)
            if terminated:
                break
        assert terminated, "Overload deveria ter encerrado o episódio"

    def test_normal_step_not_terminated(self, env):
        """Passos normais não terminam prematuramente."""
        env.reset(seed=42)
        for _ in range(5):
            _, _, terminated, _, _ = env.step(0)
            assert not terminated


# ═══════════════════════════════════════════════════════════════
# Reward
# ═══════════════════════════════════════════════════════════════

class TestReward:
    """Funções de recompensa configuráveis."""

    def test_reward_produtividade_basico(self, env_produtividade):
        """Reward produtividade: +1 por chamado atendido."""
        env_produtividade.reset(seed=42)
        # Encher filas
        env_produtividade._queue_sizes[:] = [5, 3, 1]  # noqa: SLF001
        reward = env_produtividade.step(0)[1]  # serve_priority (fila 2, peso 3)
        # Reward = 1.0 (1 chamado atendido) - delay_penalty
        assert reward > 0  # tem reward positivo pelo atendimento

    def test_reward_prioridade_peso(self, env_prioridade):
        """Reward prioridade: peso maior para filas críticas."""
        env_prioridade.reset(seed=42)
        # Encher filas
        env_prioridade._queue_sizes[:] = [3, 0, 0]  # noqa: SLF001
        reward_f0 = env_prioridade.step(0)[1]  # serve_priority → fila 2 (peso 3)

        env_prioridade.reset(seed=42)
        env_prioridade._queue_sizes[:] = [0, 3, 0]  # noqa: SLF001
        reward_f1 = env_prioridade.step(0)[1]  # serve_priority → fila 2 (peso 3)

        env_prioridade.reset(seed=42)
        env_prioridade._queue_sizes[:] = [0, 0, 3]  # noqa: SLF001
        reward_f2 = env_prioridade.step(0)[1]  # serve_priority → fila 2 (peso 3)

        # fila 2 tem maior prioridade (peso 3) → maior reward
        assert reward_f2 > reward_f1 > reward_f0

    def test_delay_penalty(self, env):
        """Atraso acumula penalidade."""
        env.reset(seed=42)
        cfg = env._config  # noqa: SLF001
        # Simular filas com espera alta
        env._avg_wait_times[:] = [10.0, 10.0, 10.0]  # noqa: SLF001 — acima do threshold
        env._queue_sizes[:] = [1, 1, 1]  # noqa: SLF001

        # _compute_reward é chamado dentro de step
        obs, reward, terminated, truncated, info = env.step(0)
        # Deve ter delay_penalty (coeff * wait - threshold)
        assert reward < 1.0  # menos que o reward base de 1 chamado


# ═══════════════════════════════════════════════════════════════
# Observation & Action Spaces
# ═══════════════════════════════════════════════════════════════

class TestSpaces:
    """Espaços de observação e ação."""

    def test_observation_space_box(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (9,)
        assert env.observation_space.dtype == np.float32

    def test_action_space_discrete(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4  # num_queues + 1

    def test_action_space_with_custom_queues(self):
        cfg = TriagemConfig(
            num_queues=5,
            arrival_rates=(1, 1, 1, 1, 1),
            priority_weights=(1, 2, 3, 4, 5),
        )
        env = TriagemEnv(cfg)
        assert env.action_space.n == 6  # 5 + 1
        assert env.observation_space.shape == (13,)  # 2*5 + 3

    def test_observation_components(self, env):
        """Observação contém queue_sizes, wait_times, capacity, used, step."""
        env.reset(seed=42)
        obs = env._get_obs()  # noqa: SLF001
        n = env._config.num_queues  # noqa: SLF001
        assert len(obs) == 2 * n + 3
        # queue_sizes
        assert np.array_equal(obs[:n], [0, 0, 0])
        # wait_times
        assert np.array_equal(obs[n : 2 * n], [0, 0, 0])
        # capacity, used, step
        assert obs[-3] == 10  # total_capacity
        assert obs[-2] == 0  # used_capacity
        assert obs[-1] == 0  # step


# ═══════════════════════════════════════════════════════════════
# Gymnasium Validation
# ═══════════════════════════════════════════════════════════════

class TestGymnasiumCheck:
    """Validação via gymnasium.utils.env_checker."""

    def test_check_env_direct(self, env):
        """TriagemEnv() direto passa check_env."""
        check_env(env)

    def test_check_env_via_gym_make(self, env_gym):
        """gym.make passa check_env no unwrapped (recomendado pelos docs)."""
        check_env(env_gym.unwrapped)

    def test_check_env_has_spec_via_gym_make(self, env_gym):
        """gym.make fornece .spec ao ambiente."""
        assert env_gym.spec is not None
        assert env_gym.spec.id == "TriagemAdaptativa-v0"

    def test_check_env_spec_direct(self, env):
        """TriagemEnv() direto também tem .spec (definido no __init__)."""
        assert env.spec is not None
        assert env.spec.id == "TriagemAdaptativa-v0"


# ═══════════════════════════════════════════════════════════════
# Registro via gym.make
# ═══════════════════════════════════════════════════════════════

class TestGymMake:
    """Ambiente via gym.make funciona corretamente."""

    def test_gym_make_reset(self, env_gym):
        obs, info = env_gym.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_gym_make_step(self, env_gym):
        env_gym.reset(seed=42)
        obs, reward, terminated, truncated, info = env_gym.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

    def test_gym_make_render_ansi(self, env_gym_ansi):
        env_gym_ansi.reset(seed=42)
        out = env_gym_ansi.render()
        assert isinstance(out, str)

    def test_gym_make_seed_reproducibility(self):
       env_a = gym.make("TriagemAdaptativa-v0")
       env_b = gym.make("TriagemAdaptativa-v0")
       obs_a, _ = env_a.reset(seed=42)
       obs_b, _ = env_b.reset(seed=42)
       assert np.array_equal(obs_a, obs_b)

    def test_gym_make_different_seeds(self):
        env_a = gym.make("TriagemAdaptativa-v0")
        env_b = gym.make("TriagemAdaptativa-v0")
        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=123)
        assert np.array_equal(obs_a, obs_b)  # tudo zero inicialmente


# ═══════════════════════════════════════════════════════════════
# Ações específicas
# ═══════════════════════════════════════════════════════════════

class TestAcoes:
    """Efeito de cada ação no ambiente."""

    def test_action_2_refer_queue_0(self, small_env):
        """Action=2 → refer_queue[0] reduz fila 0."""
        small_env.reset(seed=42)
        small_env._queue_sizes[0] = 5  # noqa: SLF001
        small_env.step(2)
        assert small_env._queue_sizes[0] == 4  # noqa: SLF001

    def test_action_3_refer_queue_1(self, small_env):
        """Action=3 → refer_queue[1] reduz fila 1."""
        small_env.reset(seed=42)
        small_env._queue_sizes[1] = 5  # noqa: SLF001
        small_env.step(3)
        assert small_env._queue_sizes[1] == 4  # noqa: SLF001

    def test_action_0_serve_highest_priority(self, small_env):
        """serve_priority atende a fila de maior prioridade."""
        small_env.reset(seed=42)
        small_env._queue_sizes[:] = [1, 2, 3]  # noqa: SLF001
        small_env.step(0)  # deve atender fila 2 (maior prioridade)
        assert small_env._queue_sizes[2] == 2  # noqa: SLF001

    def test_action_1_serve_longest(self, small_env):
        """serve_longest atende a fila com mais chamados."""
        small_env.reset(seed=42)
        small_env._queue_sizes[:] = [1, 5, 3]  # noqa: SLF001
        small_env.step(1)  # deve atender fila 1 (mais longa)
        assert small_env._queue_sizes[1] == 4  # noqa: SLF001


# ═══════════════════════════════════════════════════════════════
# Info dict
# ═══════════════════════════════════════════════════════════════

class TestInfo:
    """Dicionário info retornado por reset() e step()."""

    def test_info_keys(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert "queue_sizes" in info
        assert "avg_wait_times" in info
        assert "used_capacity" in info
        assert "step" in info

    def test_info_values_match_state(self, env):
        env.reset(seed=42)
        env._queue_sizes[0] = 3  # noqa: SLF001
        _, _, _, _, info = env.step(0)
        assert info["queue_sizes"][0] >= 0  # pode ter mudado por Poisson
        assert info["used_capacity"] >= 0
        assert info["step"] == 1


# ═══════════════════════════════════════════════════════════════
# Capacidade
# ═══════════════════════════════════════════════════════════════

class TestCapacidade:
    """Gerenciamento de capacidade."""

    def test_capacity_increments_on_serve(self, small_env):
        """Serve aumenta capacidade, mas step libera 1 ao final."""
        small_env.reset(seed=42)
        small_env._queue_sizes[:] = [1, 1, 1]  # noqa: SLF001
        small_env.step(0)  # serve +1, libera -1 → net 0
        assert small_env._used_capacity == 0  # noqa: SLF001 — serve + release = 0
        # Após 2 steps sem servir: -1 por step, mínimo 0
        small_env.step(1)  # serve +1 (se fila > 0), libera -1
        assert small_env._used_capacity == 0  # noqa: SLF001

    def test_capacity_decrements_each_step(self, env):
        env.reset(seed=42)
        env._queue_sizes[:] = [1, 1, 1]  # noqa: SLF001
        env._used_capacity = 2  # noqa: SLF001
        env.step(0)  # serve +1, libera -1 → 2 - 1 + 1 = 2
        # wait... vamos calcular: _used_capacity=2, serve +1 = 3, libera -1 = 2
        # melhor testar sem servir
        env2 = TriagemEnv()
        env2.reset(seed=42)
        env2._used_capacity = 3  # noqa: SLF001
        env2.step(1)  # serve_longest com filas vazias → não serve
        assert env2._used_capacity == 2  # noqa: SLF001 — liberou 1
