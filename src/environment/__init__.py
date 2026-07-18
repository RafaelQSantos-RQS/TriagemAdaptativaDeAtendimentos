"""Ambiente de Triagem Adaptativa de Atendimentos.

Uso direto:
    from src.environment import TriagemEnv
    env = TriagemEnv()

Uso via Gymnasium registry:
    import gymnasium as gym
    gym.make("TriagemAdaptativa-v0")
"""

from gymnasium.envs.registration import register

from src.environment.triagem_env import TriagemConfig, TriagemEnv

register(
    id="TriagemAdaptativa-v0",
    entry_point="src.environment.triagem_env:TriagemEnv",
)

__all__ = ["TriagemConfig", "TriagemEnv"]
