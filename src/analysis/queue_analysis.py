"""Gera a análise de chamados resolvidos por fila e por método."""

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

from src.analysis.agent_vs_baselines import (  # noqa: E402
    DEFAULT_SEEDS,
    GROUP_STYLES,
    SERIES,
)

QUEUE_LABELS = (
    "Fila 0 — prioridade 1",
    "Fila 1 — prioridade 2",
    "Fila 2 — prioridade 3",
)


@dataclass(frozen=True)
class QueueEntry:
    """Médias por fila de um agente ou baseline."""

    key: str
    label: str
    group: str
    mean_served: np.ndarray
    std_served: np.ndarray
    seeds: tuple[int, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Arquivo de resultados vazio: {path}")
    return rows


def _aggregate_method(
    rows: list[dict[str, str]],
    *,
    key_field: str,
    key: str,
    label: str,
    group: str,
    expected_seeds: tuple[int, ...],
) -> QueueEntry:
    selected = [row for row in rows if row.get(key_field, "").upper() == key.upper()]
    by_seed: dict[int, np.ndarray] = {}
    for row in selected:
        try:
            seed = int(row["seed"])
            values = np.asarray(
                [float(row[f"mean_served_queue_{queue}"]) for queue in range(3)],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{label}: métricas por fila ausentes ou inválidas. "
                "Execute novamente as avaliações de agentes e baselines."
            ) from error
        if seed in by_seed:
            raise ValueError(f"Seed {seed} duplicada para {label}")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Valores por fila inválidos para {label}, seed {seed}")
        by_seed[seed] = values

    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"{label}: faltam resultados para as seeds {missing}")

    seed_values = np.vstack([by_seed[seed] for seed in expected_seeds])
    ddof = 1 if seed_values.shape[0] > 1 else 0
    return QueueEntry(
        key=key,
        label=label,
        group=group,
        mean_served=seed_values.mean(axis=0),
        std_served=seed_values.std(axis=0, ddof=ddof),
        seeds=expected_seeds,
    )


def load_queue_data(
    agents_path: Path,
    baselines_path: Path,
    expected_seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[QueueEntry]:
    """Carrega os resultados e agrega os seis métodos entre seeds."""
    agent_rows = _read_csv(agents_path)
    baseline_rows = _read_csv(baselines_path)
    entries = []
    for key, label, group in SERIES:
        rows = agent_rows if group == "agent" else baseline_rows
        key_field = "config" if group == "agent" else "baseline"
        entries.append(
            _aggregate_method(
                rows,
                key_field=key_field,
                key=key,
                label=label,
                group=group,
                expected_seeds=expected_seeds,
            )
        )
    return entries


def plot_queue_comparison(
    entries: list[QueueEntry], output_path: Path, dpi: int = 180
) -> None:
    """Plota um painel por fila com média e desvio padrão entre seeds."""
    positions = np.arange(len(entries))
    method_labels = [entry.label for entry in entries]
    colors = [GROUP_STYLES[entry.group]["color"] for entry in entries]
    hatches = [GROUP_STYLES[entry.group]["hatch"] for entry in entries]
    max_with_error = max(
        float((entry.mean_served + entry.std_served).max()) for entry in entries
    )
    x_max = max(max_with_error * 1.28, 1.0)
    label_padding = x_max * 0.012

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.16, wspace=0.03)
    for queue, ax in enumerate(axes):
        means = np.asarray([entry.mean_served[queue] for entry in entries])
        stds = np.asarray([entry.std_served[queue] for entry in entries])
        bars = ax.barh(
            positions,
            means,
            xerr=stds,
            color=colors,
            edgecolor="#263238",
            linewidth=0.7,
            error_kw={"ecolor": "#263238", "elinewidth": 1.1, "capsize": 3},
        )
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)

        ax.set_title(QUEUE_LABELS[queue], fontsize=12, fontweight="bold")
        ax.set_yticks(positions, method_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Resolvidos por episódio")
        ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for position, mean, std in zip(positions, means, stds, strict=True):
            ax.text(
                mean + std + label_padding,
                position,
                f"{mean:.1f}",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#263238",
            )

    fig.suptitle(
        "Chamados resolvidos por fila — agentes vs baselines",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "Média por episódio entre 5 seeds; barras de erro = ± 1 desvio padrão entre seeds",
        ha="center",
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
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def export_summary(entries: list[QueueEntry], output_path: Path) -> None:
    """Exporta as médias por fila utilizadas no gráfico."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "group",
                "queue",
                "priority",
                "mean_resolved_per_episode",
                "std_between_seeds",
                "n_seeds",
            ],
        )
        writer.writeheader()
        for entry in entries:
            for queue in range(3):
                writer.writerow(
                    {
                        "method": entry.label,
                        "group": entry.group,
                        "queue": queue,
                        "priority": queue + 1,
                        "mean_resolved_per_episode": f"{entry.mean_served[queue]:.6f}",
                        "std_between_seeds": f"{entry.std_served[queue]:.6f}",
                        "n_seeds": len(entry.seeds),
                    }
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Comparar chamados resolvidos por fila",
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
        default=Path("experiments/analysis/by_queue"),
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
    """Gera o gráfico e o CSV da análise por fila."""
    args = parse_args(argv)
    entries = load_queue_data(
        args.agents_results,
        args.baselines_results,
        tuple(args.seeds),
    )
    chart_path = args.output_dir / "resolved_by_queue.png"
    summary_path = args.output_dir / "resolved_by_queue_summary.csv"
    plot_queue_comparison(entries, chart_path, dpi=args.dpi)
    export_summary(entries, summary_path)
    print(f"Gráfico salvo: {chart_path}")
    print(f"Resumo agregado salvo: {summary_path}")


if __name__ == "__main__":
    main()
