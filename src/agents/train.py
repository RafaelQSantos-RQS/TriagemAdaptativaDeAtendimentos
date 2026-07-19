#!/usr/bin/env python3
"""Treinamento de agente RL para Triagem Adaptativa de Atendimentos.

Uso:
    python -m src.agents.train --seed 42
    python -m src.agents.train --algo ppo --config A --seed 42
    python -m src.agents.train --total-timesteps 50000
    python -m src.agents.train  # todas as 5 sementes
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

if __name__ == "__main__" and "src" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.environment  # noqa: F401
from src.environment import TriagemConfig

_HAS_TENSORBOARD: bool = False
try:
    import torch.utils.tensorboard  # noqa: F401

    _HAS_TENSORBOARD = True
except ImportError:
    pass

SEEDS: list[int] = [42, 123, 256, 789, 1024]

CONFIG_MAP: dict[str, dict[str, str]] = {
    "A": {"algo": "ppo", "reward_config": "produtividade"},
    "B": {"algo": "ppo", "reward_config": "prioridade"},
    "C": {"algo": "dqn", "reward_config": "produtividade"},
}


def set_seed(seed: int) -> None:
    """Propaga a semente para todas as fontes de aleatoriedade.

    Args:
        seed: Semente mestre para reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(config_name: str, seed: int, rank: int = 0):
    """Factory para criar ambientes monitorados.

    Args:
        config_name: Identificador da configuração (A, B, C).
        seed: Semente base para o ambiente.
        rank: Índice para diferenciar múltiplas cópias.

    Returns:
        Função que instancia um ambiente TriagemAdaptativa com Monitor.
    """

    def _init() -> gym.Env:
        cfg = TriagemConfig(
            reward_config=CONFIG_MAP[config_name]["reward_config"],
        )
        env = gym.make("TriagemAdaptativa-v0", config=cfg)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def create_vec_env(config_name: str, seed: int, n_envs: int = 1) -> VecEnv:
    """Cria ambiente vetorizado para treino com SB3.

    Args:
        config_name: Identificador da configuração (A, B, C).
        seed: Semente base.
        n_envs: Número de ambientes paralelos.

    Returns:
        Ambiente vetorizado DummyVecEnv.
    """
    return DummyVecEnv([_make_env(config_name, seed, i) for i in range(n_envs)])


def create_model(algo_name: str, env: VecEnv, seed: int, tb_log_dir: Optional[str]):
    """Instancia modelo SB3 com hiperparâmetros conforme AGENTS.md.

    Args:
        algo_name: "ppo" ou "dqn".
        env: Ambiente vetorizado de treino.
        seed: Semente para o modelo.
        tb_log_dir: Diretório para logs TensorBoard (None se não instalado).

    Returns:
        Modelo PPO ou DQN configurado.
    """
    if tb_log_dir and not _HAS_TENSORBOARD:
        print("⚠️  tensorboard ausente. Instale com: uv pip install tensorboard")
        tb_log_dir = None

    common = {"policy": "MlpPolicy", "env": env, "verbose": 1, "seed": seed}
    if tb_log_dir:
        common["tensorboard_log"] = tb_log_dir

    if algo_name == "ppo":
        return PPO(
            **common,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    if algo_name == "dqn":
        return DQN(
            **common,
            learning_rate=1e-3,
            buffer_size=50_000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            train_freq=4,
        )

    raise ValueError(f"Algoritmo desconhecido: {algo_name}")


def setup_logger(log_dir: str) -> logging.Logger:
    """Configura logger com saída para arquivo e console.

    Args:
        log_dir: Diretório onde o arquivo train.log será criado.

    Returns:
        Logger configurado.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger(f"triagem_{os.path.basename(log_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt_file = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fmt_console = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt_file)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train_seed(
    seed: int,
    algo_name: str,
    config_name: str,
    total_timesteps: int,
    render: bool = False,
    tb_dir: Optional[str] = None,
) -> str:
    """Treina um agente para uma semente e retorna o caminho do modelo.

    Args:
        seed: Semente para reprodutibilidade.
        algo_name: Algoritmo ("ppo" | "dqn").
        config_name: Configuração experimental ("A" | "B" | "C").
        total_timesteps: Número total de passos de treino.
        render: Se True, renderiza o ambiente.
        tb_dir: Diretório para logs TensorBoard.

    Returns:
        Caminho absoluto para o modelo salvo (.zip).
    """
    config_label = f"config_{config_name.lower()}"
    model_dir = os.path.join("models", config_label, f"seed_{seed:03d}")
    exp_dir = os.path.join("experiments", config_label, f"seed_{seed:03d}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)
    set_seed(seed)
    logger.info(f"Seed {seed} — numpy, random, torch, gymnasium, SB3")

    train_env = create_vec_env(config_name, seed)
    eval_env = create_vec_env(config_name, seed + 1000)
    reward_config = CONFIG_MAP[config_name]["reward_config"]
    logger.info(
        f"Config {config_name} | {algo_name.upper()} | Reward: {reward_config} | Steps: {total_timesteps}"
    )

    tb_log_dir = tb_dir or os.path.join(exp_dir, "tensorboard")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=exp_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = create_model(algo_name, train_env, seed, tb_log_dir)
    logger.info(f"Modelo {algo_name.upper()} inicializado")

    logger.info("Treino iniciado...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=f"{algo_name}_config{config_name}_seed{seed}",
        )
        logger.info("Treino concluído!")
    except KeyboardInterrupt:
        logger.warning("Treino interrompido pelo usuário")
    except Exception:
        logger.exception("Erro durante treino")
        raise
    finally:
        train_env.close()
        eval_env.close()

    model_path = os.path.join(model_dir, "model.zip")
    model.save(model_path)
    logger.info(f"Modelo salvo: {model_path}")
    return model_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos de linha de comando.

    Args:
        argv: Lista de argumentos (usa sys.argv se None).

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Treinar agente RL para Triagem Adaptativa de Atendimentos",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente específica (default: todas: 42, 123, 256, 789, 1024)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["ppo", "dqn"],
        help="Algoritmo RL (default: segue o CONFIG_MAP)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="Configuração experimental (default: A)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Passos de treino (default: 200000)",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="Diretório para logs TensorBoard",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada: treina agente para uma ou mais sementes.

    Args:
        argv: Argumentos da linha de comando.
    """
    args = parse_args(argv)
    seeds: list[int] = [args.seed] if args.seed is not None else SEEDS
    algo_name = args.algo or CONFIG_MAP[args.config]["algo"]

    for seed in seeds:
        header = f"  Config {args.config} | {algo_name.upper()} | Seed {seed}  "
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}\n")
        train_seed(
            seed=seed,
            algo_name=algo_name,
            config_name=args.config,
            total_timesteps=args.total_timesteps,
            render=False,
            tb_dir=args.tensorboard,
        )


if __name__ == "__main__":
    main()
