#!/usr/bin/env python3
"""Avalia todas as baselines heurísticas no ambiente de triagem.

Uso:
    python -m src.baselines.run
    python -m src.baselines.run --episodes 50
    python -m src.baselines.run --seeds 42 123

Cada baseline executa N episódios por semente e coleta:
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
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

if __name__ == "__main__" and "src" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.environment  # noqa: F401
from src.baselines import fixed_priority, longest_queue, random as random_baseline
from src.environment import TriagemConfig

SEEDS: list[int] = [42, 123, 256, 789, 1024]
BASELINE_MODULES: dict[str, Any] = {
    "aleatorio": random_baseline,
    "prioridade_fixa": fixed_priority,
    "fila_mais_longa": longest_queue,
}
NUM_QUEUES: int = 3
EVAL_EPISODES: int = 100


def evaluate_baseline(
    name: str,
    episodes: int,
    seed: int,
    config: TriagemConfig | None = None,
) -> dict[str, float]:
    """Avalia uma baseline por N episódios e retorna métricas agregadas.

    Args:
        name: Nome da baseline ("aleatorio", "prioridade_fixa",
            "fila_mais_longa").
        episodes: Número de episódios de avaliação.
        seed: Semente para o ambiente.
        config: Configuração do ambiente (usa default se None).

    Returns:
        Dicionário com métricas: mean_reward, std_reward, success_rate,
        mean_steps, mean_cost, total_served, total_arrivals.
    """
    cfg = config or TriagemConfig()
    module = BASELINE_MODULES[name]

    rewards: list[float] = []
    steps_list: list[int] = []
    total_served = 0
    total_arrivals = 0

    for ep in range(episodes):
        env = gym.make("TriagemAdaptativa-v0", config=cfg)
        obs, _info = env.reset(seed=seed + ep)

        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0

        while not (terminated or truncated):
            action = _select_action(module, name, obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        total_served += int(sum(info.get("queue_sizes", np.zeros(NUM_QUEUES))))
        total_arrivals += ep_steps * sum(cfg.arrival_rates)

        env.close()

    rewards_arr = np.array(rewards)
    return {
        "baseline": name,
        "seed": seed,
        "episodes": episodes,
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "success_rate": float(np.mean(np.array(rewards) >= 0)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "total_served": int(total_served),
        "total_arrivals": int(total_arrivals),
    }


def _select_action(
    module: Any,
    name: str,
    obs: np.ndarray,
    env: gym.Env,
) -> int:
    """Chama o seletor de ação da baseline com a assinatura correta.

    Baselines determinísticas (prioridade_fixa, fila_mais_longa) recebem
    apenas (obs, n_actions); a aleatória recebe também um RNG.
    """
    n = int(env.action_space.n) if hasattr(env.action_space, "n") else NUM_QUEUES + 1
    if name == "aleatorio":
        return module.select_action(obs, n, random.Random())
    return module.select_action(obs, n)


def save_results(results: list[dict[str, float]], output_dir: str) -> str:
    """Salva resultados em CSV e JSON.

    Args:
        results: Lista de dicionários com métricas por baseline/seed.
        output_dir: Diretório para salvar os arquivos.

    Returns:
        Caminho do arquivo CSV gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "baselines_results.csv")
    fieldnames = [
        "baseline",
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

    json_path = os.path.join(output_dir, "baselines_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return csv_path


def print_summary(results: list[dict[str, float]]) -> None:
    """Exibe tabela resumo das baselines no console.

    Args:
        results: Lista de resultados por baseline/seed.
    """
    print(f"\n{'=' * 78}")
    print("  RESUMO DAS BASELINES")
    print(f"{'=' * 78}")

    from collections import defaultdict

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in results:
        grouped[r["baseline"]].append(r)

    for name, seeds in grouped.items():
        rewards = [s["mean_reward"] for s in seeds]
        header = f"\n  ▶ {name.upper()}"
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
        description="Avaliar baselines heurísticas para Triagem Adaptativa",
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
        "--output",
        type=str,
        default="experiments/baselines",
        help="Diretório de saída (default: experiments/baselines)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada: avalia todas as baselines.

    Args:
        argv: Argumentos da linha de comando.
    """
    args = parse_args(argv)
    seeds: list[int] = args.seeds or SEEDS

    all_results: list[dict[str, float]] = []
    baseline_names = list(BASELINE_MODULES.keys())

    for name in baseline_names:
        print(f"\n  → Avaliando baseline: {name}")
        for seed in seeds:
            result = evaluate_baseline(
                name=name,
                episodes=args.episodes,
                seed=seed,
            )
            all_results.append(result)
            print(
                f"    seed {seed:>4d}: reward {result['mean_reward']:>8.2f} "
                f"± {result['std_reward']:>6.2f}"
            )

    csv_path = save_results(all_results, args.output)
    print_summary(all_results)
    print(f"Resultados salvos: {csv_path}")


if __name__ == "__main__":
    main()
