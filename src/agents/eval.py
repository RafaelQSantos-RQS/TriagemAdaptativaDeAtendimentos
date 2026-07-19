#!/usr/bin/env python3
"""Avalia agentes RL treinados no ambiente de triagem.

Uso:
    python -m src.agents.eval
    python -m src.agents.eval --episodes 10    # quick test
    python -m src.agents.eval --config A        # só Config A
    python -m src.agents.eval --seed 42         # só seed 42

Cada modelo executa N episódios por semente e coleta:
    - Recompensa média e desvio padrão
    - Taxa de sucesso (chamados resolvidos / total)
    - Passos por episódio
    - Penalidades acumuladas
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO

if __name__ == "__main__" and "src" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.environment  # noqa: F401
from src.agents.train import CONFIG_MAP, SEEDS
from src.environment import TriagemConfig

EVAL_EPISODES: int = 100
NUM_QUEUES: int = 3
ALGO_MAP: dict[str, type] = {"ppo": PPO, "dqn": DQN}


def _model_dir(config_name: str, seed: int) -> str:
    """Retorna o caminho do diretório do modelo."""
    label = f"config_{config_name.lower()}"
    return os.path.join("models", label, f"seed_{seed:03d}")


def _result_dir(config_name: str, seed: int) -> str:
    """Retorna o caminho do diretório de resultados."""
    label = f"config_{config_name.lower()}"
    return os.path.join("experiments", "results", label, f"seed_{seed:03d}")


def load_model(
    config_name: str,
    algo_name: str,
    seed: int,
) -> PPO | DQN:
    """Carrega modelo SB3 treinado do disco.

    Args:
        config_name: Configuração experimental (A, B, C).
        algo_name: Algoritmo ("ppo" | "dqn").
        seed: Semente do modelo.

    Returns:
        Modelo carregado.

    Raises:
        FileNotFoundError: Se o model.zip não existir.
    """
    model_path = os.path.join(_model_dir(config_name, seed), "model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            f"Treine primeiro com: python -m src.agents.train --config {config_name} --seed {seed}"
        )

    algo_class = ALGO_MAP[algo_name]
    return algo_class.load(model_path)


def evaluate_agent(
    model: PPO | DQN,
    episodes: int,
    seed: int,
    config: TriagemConfig | None = None,
    deterministic: bool = True,
) -> dict[str, float]:
    """Avalia um agente por N episódios e retorna métricas agregadas.

    Args:
        model: Modelo SB3 carregado.
        episodes: Número de episódios de avaliação.
        seed: Semente base para o ambiente.
        config: Configuração do ambiente (usa default se None).
        deterministic: Se True, usa ação determinística (avaliação).
                      Se False, amostra da distribuição (exploração).

    Returns:
        Dicionário com métricas: mean_reward, std_reward, success_rate,
        mean_steps, min_reward, max_reward, total_served, total_arrivals.
    """
    cfg = config or TriagemConfig()

    rewards: list[float] = []
    steps_list: list[int] = []
    total_served = 0
    total_arrivals = 0

    env = gym.make("TriagemAdaptativa-v0", config=cfg)
    for ep in range(episodes):
        obs, _info = env.reset(seed=seed + ep)

        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            ep_steps += 1

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        total_served += int(info.get("total_served", 0))
        total_arrivals += int(info.get("total_arrivals", 0))

    env.close()

    rewards_arr = np.array(rewards)
    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "success_rate": float(np.mean(rewards_arr >= 0)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "total_served": int(total_served),
        "total_arrivals": int(total_arrivals),
    }


def save_results(
    results: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Salva resultados em CSV e JSON.

    Args:
        results: Lista de dicionários com métricas por config/seed.
        output_dir: Diretório para salvar os arquivos.

    Returns:
        Caminho do arquivo CSV gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "eval_results.csv")
    fieldnames = [
        "config",
        "algo",
        "reward_config",
        "seed",
        "episodes",
        "mean_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "success_rate",
        "mean_steps",
        "std_steps",
        "total_served",
        "total_arrivals",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return csv_path


def print_summary(results: list[dict[str, Any]]) -> None:
    """Exibe tabela resumo da avaliação no console.

    Args:
        results: Lista de resultados por config/seed.
    """
    print(f"\n{'=' * 78}")
    print("  RESUMO DA AVALIAÇÃO DOS AGENTES")
    print(f"{'=' * 78}")

    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        key = f"Config {r['config']} ({r['algo'].upper()}, {r['reward_config']})"
        grouped[key].append(r)

    for label, seeds in grouped.items():
        rewards = [s["mean_reward"] for s in seeds]
        header = f"\n  ▶ {label}"
        print(f"\n{header}")
        print(f"  {'-' * len(header)}")
        print(f"    Seeds:        {[s['seed'] for s in seeds]}")
        print(f"    Reward médio: {np.mean(rewards):>8.2f} ± {np.std(rewards):>6.2f}")
        print(f"    Sucesso:      {np.mean([s['success_rate'] for s in seeds]):>7.1%}")
        print(f"    Passos:       {np.mean([s['mean_steps'] for s in seeds]):>7.1f}")
        for s in seeds:
            print(
                f"      seed {s['seed']:>4d}: reward {s['mean_reward']:>8.2f}  "
                f"sucesso {s['success_rate']:.0%}"
            )

    print(f"\n{'=' * 78}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos de linha de comando.

    Args:
        argv: Lista de argumentos (usa sys.argv se None).

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Avaliar agentes RL treinados para Triagem Adaptativa",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"Número de episódios por semente (default: {EVAL_EPISODES})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Sementes específicas (default: todas: 42, 123, 256, 789, 1024)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=["A", "B", "C"],
        help="Configuração específica (default: todas)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Usar ação determinística na avaliação (default: True)",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="Amostrar da distribuição de ação (exploração)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results",
        help="Diretório de saída (default: experiments/results)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada: avalia agentes treinados.

    Args:
        argv: Argumentos da linha de comando.
    """
    args = parse_args(argv)
    seeds: list[int] = args.seeds or SEEDS
    configs: list[str] = [args.config] if args.config else ["A", "B", "C"]

    all_results: list[dict[str, Any]] = []

    for config_name in configs:
        algo_name = CONFIG_MAP[config_name]["algo"]
        reward_config = CONFIG_MAP[config_name]["reward_config"]
        print(
            f"\n  → Avaliando Config {config_name} ({algo_name.upper()}, {reward_config})"
        )

        for seed in seeds:
            try:
                model = load_model(config_name, algo_name, seed)
            except FileNotFoundError as e:
                print(f"    ⚠️  {e}")
                continue

            cfg = TriagemConfig(reward_config=reward_config)
            result = evaluate_agent(
                model=model,
                episodes=args.episodes,
                seed=seed,
                config=cfg,
                deterministic=args.deterministic,
            )

            row: dict[str, Any] = {
                "config": config_name,
                "algo": algo_name,
                "reward_config": reward_config,
                "seed": seed,
                "episodes": args.episodes,
                **result,
            }
            all_results.append(row)

            print(
                f"    seed {seed:>4d}: reward {result['mean_reward']:>8.2f} "
                f"± {result['std_reward']:>6.2f}"
            )

    if not all_results:
        print("\n  ⚠️  Nenhum modelo foi avaliado.")
        print("  Treine os modelos primeiro com:")
        print("    python -m src.agents.train --config A")
        print("    python -m src.agents.train --config B")
        print("    python -m src.agents.train --config C")
        return

    csv_path = save_results(all_results, args.output)
    print_summary(all_results)
    print(f"Resultados salvos: {csv_path}")


if __name__ == "__main__":
    main()
