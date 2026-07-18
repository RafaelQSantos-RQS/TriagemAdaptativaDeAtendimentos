"""Fixtures compartilhadas para testes do ambiente."""

import gymnasium as gym
import pytest

import src.environment  # noqa: F401 — registra o ambiente
from src.environment import TriagemConfig, TriagemEnv


@pytest.fixture
def env():
    """Ambiente padrão via construtor direto (render_mode=None)."""
    return TriagemEnv()


@pytest.fixture
def env_ansi():
    """Ambiente com render_mode='ansi' via construtor direto."""
    return TriagemEnv(render_mode="ansi")


@pytest.fixture
def env_gym():
    """Ambiente registrado via gym.make."""
    return gym.make("TriagemAdaptativa-v0")


@pytest.fixture
def env_gym_ansi():
    """Ambiente registrado via gym.make com render_mode='ansi'."""
    return gym.make("TriagemAdaptativa-v0", render_mode="ansi")


@pytest.fixture
def small_env():
    """Ambiente com capacidade reduzida para testar casos limite."""
    cfg = TriagemConfig(
        max_queue_size=3,
        total_capacity=2,
        max_steps=10,
        arrival_rates=(0.0, 0.0, 0.0),  # sem chegadas para controle
    )
    return TriagemEnv(cfg)


@pytest.fixture
def overload_env():
    """Ambiente que encerra rápido por overload."""
    cfg = TriagemConfig(
        max_queue_size=5,
        total_capacity=10,
        max_steps=100,
        overload_threshold=0.5,
        overload_patience=3,
        arrival_rates=(5.0, 5.0, 5.0),  # muitas chegadas
    )
    return TriagemEnv(cfg)


@pytest.fixture
def env_produtividade():
    """Ambiente com reward_config='produtividade'."""
    return TriagemEnv(TriagemConfig(reward_config="produtividade"))


@pytest.fixture
def env_prioridade():
    """Ambiente com reward_config='prioridade'."""
    return TriagemEnv(TriagemConfig(reward_config="prioridade"))


@pytest.fixture
def seeded_env():
    """Ambiente com seed fixa para testes determinísticos."""
    env = TriagemEnv()
    env.reset(seed=42)
    return env
