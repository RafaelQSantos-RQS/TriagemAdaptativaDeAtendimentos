"""Compara o reward médio dos agentes treinados com as baselines.

Cada barra representa a média dos valores ``mean_reward`` obtidos nas cinco
seeds. As barras de erro representam o desvio padrão amostral entre seeds.
Todos os resultados devem ter sido avaliados com a mesma função de recompensa.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DEFAULT_SEEDS = (42, 123, 256, 789, 1024)
EXPECTED_REWARD_CONFIG = "produtividade"

SERIES = (
    ("A", "Config A", "agent"),
    ("B", "Config B", "agent"),
    ("C", "Config C", "agent"),
    ("aleatorio", "Aleatório", "baseline"),
    ("prioridade_fixa", "Prioridade Fixa", "baseline"),
    ("fila_mais_longa", "Fila Longa", "baseline"),
)

GROUP_STYLES = {
    "agent": {"color": "#2563A6", "hatch": "", "label": "Agentes RL"},
    "baseline": {"color": "#D69E2E", "hatch": "//", "label": "Baselines"},
}


@dataclass(frozen=True)
class ComparisonEntry:
    """Resultado agregado de um agente ou baseline."""

    key: str
    label: str
    group: str
    mean_reward: float
    std_reward: float
    seeds: tuple[int, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Arquivo de resultados vazio: {path}")
    return rows


def _aggregate_rows(
    rows: list[dict[str, str]],
    *,
    key_field: str,
    key: str,
    label: str,
    group: str,
    expected_seeds: tuple[int, ...],
) -> ComparisonEntry:
    selected = [row for row in rows if row.get(key_field, "").upper() == key.upper()]
    by_seed: dict[int, float] = {}
    for row in selected:
        try:
            seed = int(row["seed"])
            reward = float(row["mean_reward"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Registro inválido para {label}: {row}") from error
        if seed in by_seed:
            raise ValueError(f"Seed {seed} duplicada para {label}")
        if not np.isfinite(reward):
            raise ValueError(f"Reward não finito para {label}, seed {seed}")
        by_seed[seed] = reward

    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"{label}: faltam resultados para as seeds {missing}")

    rewards = np.asarray([by_seed[seed] for seed in expected_seeds], dtype=float)
    ddof = 1 if rewards.size > 1 else 0
    return ComparisonEntry(
        key=key,
        label=label,
        group=group,
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std(ddof=ddof)),
        seeds=expected_seeds,
    )


def load_comparison_data(
    agents_path: Path,
    baselines_path: Path,
    expected_seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[ComparisonEntry]:
    """Carrega e agrega os seis métodos na ordem definida para o gráfico."""
    agent_rows = _read_csv(agents_path)
    baseline_rows = _read_csv(baselines_path)

    reward_configs = {
        row.get("reward_config", "").strip().lower() for row in agent_rows
    }
    if reward_configs != {EXPECTED_REWARD_CONFIG}:
        found = ", ".join(sorted(value or "não informado" for value in reward_configs))
        raise ValueError(
            "Os agentes precisam ser avaliados com uma função de recompensa comum. "
            f"Esperado: '{EXPECTED_REWARD_CONFIG}'; encontrado: {found}. "
            "Execute: uv run python -m src.agents.eval "
            "--evaluation-reward produtividade"
        )

    entries = []
    for key, label, group in SERIES:
        rows = agent_rows if group == "agent" else baseline_rows
        key_field = "config" if group == "agent" else "baseline"
        entries.append(
            _aggregate_rows(
                rows,
                key_field=key_field,
                key=key,
                label=label,
                group=group,
                expected_seeds=expected_seeds,
            )
        )
    return entries


def plot_comparison(
    entries: list[ComparisonEntry], output_path: Path, dpi: int = 180
) -> None:
    """Gera o gráfico de barras horizontais com erro entre seeds."""
    labels = [entry.label for entry in entries]
    means = np.asarray([entry.mean_reward for entry in entries])
    stds = np.asarray([entry.std_reward for entry in entries])
    positions = np.arange(len(entries))
    colors = [GROUP_STYLES[entry.group]["color"] for entry in entries]
    hatches = [GROUP_STYLES[entry.group]["hatch"] for entry in entries]

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    bars = ax.barh(
        positions,
        means,
        xerr=stds,
        color=colors,
        edgecolor="#263238",
        linewidth=0.7,
        error_kw={"ecolor": "#263238", "elinewidth": 1.2, "capsize": 4},
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)

    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reward médio por episódio")
    ax.axvline(0, color="#4B5563", linewidth=1.0)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    lower = means - stds
    upper = means + stds
    span = max(float(upper.max() - lower.min()), 1.0)
    padding = span * 0.025
    ax.set_xlim(
        min(float(lower.min()) - 8 * padding, 0),
        max(float(upper.max()) + 8 * padding, 0),
    )
    for position, mean, std in zip(positions, means, stds, strict=True):
        if mean <= 0:
            x = mean - std - padding
            horizontal_alignment = "right"
        else:
            x = mean + std + padding
            horizontal_alignment = "left"
        ax.text(
            x,
            position,
            f"{mean:.1f} ± {std:.1f}",
            ha=horizontal_alignment,
            va="center",
            fontsize=9,
            color="#263238",
        )

    ax.set_title(
        "Comparação de reward — agentes vs baselines",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )
    ax.text(
        0,
        1.015,
        "Média por episódio entre 5 seeds; barras de erro = ± 1 desvio padrão entre seeds",
        transform=ax.transAxes,
        color="#4B5563",
        fontsize=10,
    )
    legend_handles = [
        Patch(
            facecolor=style["color"],
            edgecolor="#263238",
            hatch=style["hatch"],
            label=style["label"],
        )
        for style in GROUP_STYLES.values()
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def export_summary(entries: list[ComparisonEntry], output_path: Path) -> None:
    """Exporta os valores agregados exibidos no gráfico."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["method", "group", "mean_reward", "std_reward", "n_seeds"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "method": entry.label,
                    "group": entry.group,
                    "mean_reward": f"{entry.mean_reward:.6f}",
                    "std_reward": f"{entry.std_reward:.6f}",
                    "n_seeds": len(entry.seeds),
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Comparar reward médio dos agentes e das baselines",
    )
    parser.add_argument(
        "--agents-results",
        type=Path,
        default=Path("experiments/results/eval_results.csv"),
        help="CSV produzido por src.agents.eval",
    )
    parser.add_argument(
        "--baselines-results",
        type=Path,
        default=Path("experiments/baselines/baselines_results.csv"),
        help="CSV produzido por src.baselines.run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis/agent_vs_baselines"),
        help="Diretório do gráfico e resumo agregado",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds esperadas (default: 42 123 256 789 1024)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Resolução do PNG")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Gera a comparação e o CSV agregado."""
    args = parse_args(argv)
    entries = load_comparison_data(
        args.agents_results,
        args.baselines_results,
        tuple(args.seeds),
    )
    chart_path = args.output_dir / "agent_vs_baselines_reward.png"
    summary_path = args.output_dir / "agent_vs_baselines_summary.csv"
    plot_comparison(entries, chart_path, dpi=args.dpi)
    export_summary(entries, summary_path)
    print(f"Gráfico salvo: {chart_path}")
    print(f"Resumo agregado salvo: {summary_path}")


if __name__ == "__main__":
    main()
