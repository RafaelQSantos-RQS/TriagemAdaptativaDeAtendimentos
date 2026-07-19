"""Testes dos módulos de baseline.

Verifica se cada baseline executa sem erro e retorna ações válidas.
Testes de integração com o ambiente.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np

import src.environment  # noqa: F401
from src.baselines import fixed_priority, longest_queue, random as random_baseline


class TestRandomBaseline:
    """Baseline aleatória — ações uniformemente distribuídas."""

    def test_select_action_returns_int(self):
        """select_action retorna um int."""
        obs = np.zeros(9, dtype=np.float32)
        action = random_baseline.select_action(obs, 4, random.Random(42))
        assert isinstance(action, int)

    def test_select_action_in_range(self):
        """Ação está dentro do espaço de ações."""
        obs = np.zeros(9, dtype=np.float32)
        rng = random.Random(42)
        n_actions = 4
        for _ in range(100):
            action = random_baseline.select_action(obs, n_actions, rng)
            assert 0 <= action < n_actions

    def test_seed_reproducibility(self):
        """Mesma semente produz mesma sequência de ações."""
        obs = np.zeros(9, dtype=np.float32)
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        actions_a = [random_baseline.select_action(obs, 4, rng_a) for _ in range(20)]
        actions_b = [random_baseline.select_action(obs, 4, rng_b) for _ in range(20)]
        assert actions_a == actions_b

    def test_distribution_uniform(self):
        """1000 ações devem distribuir aproximadamente uniforme entre 4 ações."""
        obs = np.zeros(9, dtype=np.float32)
        rng = random.Random(42)
        counts = [0, 0, 0, 0]
        for _ in range(1000):
            action = random_baseline.select_action(obs, 4, rng)
            counts[action] += 1
        # Cada ação deve ter ~250 (± tolerância)
        for c in counts:
            assert 200 <= c <= 300, f"Contagem {c} fora do esperado (200-300)"

    def test_integration_with_env(self):
        """Baseline aleatória executa um episódio completo sem erro."""
        env = gym.make("TriagemAdaptativa-v0")
        obs, _ = env.reset(seed=42)
        rng = random.Random(42)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action = random_baseline.select_action(obs, int(env.action_space.n), rng)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
        assert steps > 0


class TestFixedPriorityBaseline:
    """Baseline de prioridade fixa — sempre action 0."""

    def test_select_action_returns_zero(self):
        """select_action sempre retorna 0."""
        obs = np.zeros(9, dtype=np.float32)
        for _ in range(10):
            assert fixed_priority.select_action(obs, 4) == 0

    def test_select_action_ignores_obs(self):
        """Retorna 0 independente do estado."""
        for _ in range(10):
            obs = np.random.randn(9).astype(np.float32)
            assert fixed_priority.select_action(obs, 4) == 0

    def test_integration_with_env(self):
        """Baseline prioridade fixa executa um episódio completo sem erro."""
        env = gym.make("TriagemAdaptativa-v0")
        obs, _ = env.reset(seed=42)
        n_actions = int(env.action_space.n)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action = fixed_priority.select_action(obs, n_actions)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
        assert steps > 0


class TestLongestQueueBaseline:
    """Baseline de fila mais longa — sempre action 1."""

    def test_select_action_returns_one(self):
        """select_action sempre retorna 1."""
        obs = np.zeros(9, dtype=np.float32)
        for _ in range(10):
            assert longest_queue.select_action(obs, 4) == 1

    def test_select_action_ignores_obs(self):
        """Retorna 1 independente do estado."""
        for _ in range(10):
            obs = np.random.randn(9).astype(np.float32)
            assert longest_queue.select_action(obs, 4) == 1

    def test_integration_with_env(self):
        """Baseline fila mais longa executa um episódio completo sem erro."""
        env = gym.make("TriagemAdaptativa-v0")
        obs, _ = env.reset(seed=42)
        n_actions = int(env.action_space.n)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action = longest_queue.select_action(obs, n_actions)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
        assert steps > 0
