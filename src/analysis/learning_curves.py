"""Gera curvas de aprendizado agregadas entre sementes.

Os dados são lidos dos arquivos ``evaluations.npz`` produzidos pelo
``EvalCallback`` do Stable-Baselines3. Para cada avaliação, primeiro é
calculada a recompensa média entre episódios. Em seguida, as curvas são
agregadas entre sementes usando média e desvio padrão amostral.
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

DEFAULT_CONFIGS = ("A", "B", "C")
DEFAULT_SEEDS = (42, 123, 256, 789, 1024)

CONFIG_STYLES = {
    "A": {
        "label": "Config A — PPO + produtividade",
        "color": "#2563A6",
        "linestyle": "-",
        "marker": "o",
    },
    "B": {
        "label": "Config B — PPO + prioridade",
        "color": "#D97706",
        "linestyle": "--",
        "marker": "s",
    },
    "C": {
        "label": "Config C — DQN + produtividade",
        "color": "#9C4F76",
        "linestyle": "-.",
        "marker": "D",
    },
}


@dataclass(frozen=True)
class LearningCurve:
    """Curva agregada de uma configuração experimental."""

    config: str
    timesteps: np.ndarray
    mean_reward: np.ndarray
    std_reward: np.ndarray
    seed_rewards: np.ndarray
    seeds: tuple[int, ...]


def load_seed_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega timesteps e reward médio por avaliação para uma semente."""
    with np.load(path, allow_pickle=False) as data:
        required = {"timesteps", "results"}
        missing = required.difference(data.files)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{path} não contém os campos obrigatórios: {fields}")

        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        results = np.asarray(data["results"], dtype=np.float64)

    if timesteps.ndim != 1:
        raise ValueError(f"{path}: 'timesteps' deve ser um vetor unidimensional")
    if results.ndim != 2 or results.shape[0] != timesteps.size:
        raise ValueError(
            f"{path}: 'results' deve ter formato (avaliações, episódios) "
            "e corresponder a 'timesteps'"
        )
    if timesteps.size == 0 or results.shape[1] == 0:
        raise ValueError(f"{path}: arquivo de avaliação vazio")
    if not np.all(np.isfinite(results)):
        raise ValueError(f"{path}: recompensas contêm valores não finitos")
    if np.any(np.diff(timesteps) <= 0):
        raise ValueError(f"{path}: timesteps devem ser estritamente crescentes")

    return timesteps, results.mean(axis=1)


def aggregate_config(
    experiments_dir: Path,
    config: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    allow_partial: bool = False,
) -> LearningCurve:
    """Agrega as curvas de uma configuração em timesteps comuns."""
    config = config.upper()
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    missing_paths: list[Path] = []

    for seed in seeds:
        path = (
            experiments_dir
            / f"config_{config.lower()}"
            / f"seed_{seed:03d}"
            / "evaluations.npz"
        )
        if not path.exists():
            missing_paths.append(path)
            continue
        timesteps, rewards = load_seed_curve(path)
        curves.append((seed, timesteps, rewards))

    if missing_paths and not allow_partial:
        formatted = "\n  - ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Faltam avaliações para gerar a curva com todas as seeds:\n"
            f"  - {formatted}\n"
            "Aguarde o pipeline terminar ou use --allow-partial para uma prévia."
        )
    if not curves:
        raise FileNotFoundError(f"Nenhuma avaliação encontrada para a Config {config}")

    common_timesteps = curves[0][1]
    for _, timesteps, _ in curves[1:]:
        common_timesteps = np.intersect1d(common_timesteps, timesteps, assume_unique=True)
    if common_timesteps.size == 0:
        raise ValueError(f"As seeds da Config {config} não têm timesteps em comum")

    aligned_rewards = []
    used_seeds = []
    for seed, timesteps, rewards in curves:
        index_by_timestep = dict(zip(timesteps.tolist(), rewards.tolist(), strict=True))
        aligned_rewards.append([index_by_timestep[int(step)] for step in common_timesteps])
        used_seeds.append(seed)

    seed_rewards = np.asarray(aligned_rewards, dtype=np.float64)
    ddof = 1 if seed_rewards.shape[0] > 1 else 0
    return LearningCurve(
        config=config,
        timesteps=common_timesteps,
        mean_reward=seed_rewards.mean(axis=0),
        std_reward=seed_rewards.std(axis=0, ddof=ddof),
        seed_rewards=seed_rewards,
        seeds=tuple(used_seeds),
    )


def _format_axes(ax: plt.Axes) -> None:
    """Aplica formatação visual comum às curvas."""
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward médio por episódio")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    ax.axhline(0, color="#4B5563", linewidth=0.9, linestyle=":", alpha=0.75)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_curve(ax: plt.Axes, curve: LearningCurve, *, show_seed_lines: bool) -> None:
    """Plota média, banda de desvio padrão e, opcionalmente, cada seed."""
    style = CONFIG_STYLES[curve.config]
    if show_seed_lines:
        for rewards in curve.seed_rewards:
            ax.plot(
                curve.timesteps,
                rewards,
                color=style["color"],
                linewidth=0.8,
                alpha=0.18,
            )

    marker_every = max(1, curve.timesteps.size // 10)
    ax.fill_between(
        curve.timesteps,
        curve.mean_reward - curve.std_reward,
        curve.mean_reward + curve.std_reward,
        color=style["color"],
        alpha=0.16,
        linewidth=0,
        label="± 1 desvio padrão",
    )
    ax.plot(
        curve.timesteps,
        curve.mean_reward,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=marker_every,
        markersize=4.5,
        linewidth=2.2,
        label=style["label"],
    )


def plot_config_curve(curve: LearningCurve, output_path: Path, dpi: int = 180) -> None:
    """Salva o gráfico individual de uma configuração."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    _plot_curve(ax, curve, show_seed_lines=True)
    _format_axes(ax)
    ax.set_title(
        f"Curva de aprendizado — Configuração {curve.config}",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )
    ax.text(
        0,
        1.015,
        f"Média entre {len(curve.seeds)} seeds; banda = ± 1 desvio padrão entre seeds",
        transform=ax.transAxes,
        color="#4B5563",
        fontsize=10,
    )
    ax.legend(frameon=False, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_comparison(
    curves: list[LearningCurve], output_path: Path, dpi: int = 180
) -> None:
    """Salva o comparativo entre configurações."""
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for curve in curves:
        _plot_curve(ax, curve, show_seed_lines=False)

    _format_axes(ax)
    ax.set_title(
        "Curvas de aprendizado — comparação das configurações",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )
    seed_counts = ", ".join(
        f"{curve.config}: {len(curve.seeds)} seeds" for curve in curves
    )
    ax.text(
        0,
        1.015,
        f"Reward médio de avaliação; bandas = ± 1 desvio padrão ({seed_counts})",
        transform=ax.transAxes,
        color="#4B5563",
        fontsize=10,
    )

    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if label != "± 1 desvio padrão"
    ]
    ax.legend(
        [item[0] for item in filtered],
        [item[1] for item in filtered],
        frameon=False,
        loc="best",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def export_summary(curves: list[LearningCurve], output_path: Path) -> None:
    """Exporta os pontos agregados utilizados nos gráficos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["config", "timestep", "mean_reward", "std_reward", "n_seeds"],
        )
        writer.writeheader()
        for curve in curves:
            for timestep, mean, std in zip(
                curve.timesteps,
                curve.mean_reward,
                curve.std_reward,
                strict=True,
            ):
                writer.writerow(
                    {
                        "config": curve.config,
                        "timestep": int(timestep),
                        "mean_reward": f"{mean:.6f}",
                        "std_reward": f"{std:.6f}",
                        "n_seeds": len(curve.seeds),
                    }
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gerar curvas de aprendizado agregadas entre seeds",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Diretório dos experimentos (default: experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis/learning_curves"),
        help="Diretório de saída das figuras e do CSV",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str.upper,
        choices=DEFAULT_CONFIGS,
        default=list(DEFAULT_CONFIGS),
        help="Configurações incluídas (default: A B C)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds esperadas (default: 42 123 256 789 1024)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Gera uma prévia mesmo se algumas seeds ainda não estiverem prontas",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Resolução dos PNGs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Gera gráficos individuais, comparativo e tabela agregada."""
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    curves = [
        aggregate_config(
            args.experiments_dir,
            config,
            seeds,
            allow_partial=args.allow_partial,
        )
        for config in args.configs
    ]

    for curve in curves:
        output_path = args.output_dir / f"learning_curve_config_{curve.config.lower()}.png"
        plot_config_curve(curve, output_path, dpi=args.dpi)
        print(f"Gráfico salvo: {output_path}")

    comparison_path = args.output_dir / "learning_curves_comparison.png"
    plot_comparison(curves, comparison_path, dpi=args.dpi)
    print(f"Comparativo salvo: {comparison_path}")

    summary_path = args.output_dir / "learning_curves_summary.csv"
    export_summary(curves, summary_path)
    print(f"Resumo agregado salvo: {summary_path}")


if __name__ == "__main__":
    main()
