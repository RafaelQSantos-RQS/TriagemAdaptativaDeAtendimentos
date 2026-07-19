"""Gera a tabela-resumo final de agentes e baselines.

As métricas seguem ``.specs/06-evaluation-metrics.md``:

- reward médio por episódio;
- taxa de sucesso = total resolvido / total de chegadas;
- passos médios por episódio;
- custo acumulado médio por episódio;
- desvio padrão amostral do reward médio entre seeds.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEEDS = (42, 123, 256, 789, 1024)
EXPECTED_REWARD_CONFIG = "produtividade"

METHODS = (
    ("config", "A", "A — PPO Produtividade", "agent"),
    ("config", "B", "B — PPO Prioridade", "agent"),
    ("config", "C", "C — DQN Produtividade", "agent"),
    ("baseline", "aleatorio", "Aleatório", "baseline"),
    ("baseline", "prioridade_fixa", "Prioridade Fixa", "baseline"),
    ("baseline", "fila_mais_longa", "Fila Mais Longa", "baseline"),
)


@dataclass(frozen=True)
class SummaryRow:
    """Métricas finais agregadas entre seeds para um método."""

    method: str
    group: str
    mean_reward: float
    success_rate: float
    mean_steps: float
    mean_cost: float
    std_reward_between_seeds: float
    n_seeds: int
    episodes_per_seed: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de avaliação não encontrado: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Arquivo de avaliação vazio: {path}")
    return rows


def _number(row: dict[str, str], field: str, label: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: campo '{field}' ausente ou inválido. "
            "Reexecute as avaliações para atualizar as métricas."
        ) from error


def _aggregate_method(
    rows: list[dict[str, str]],
    key_field: str,
    key: str,
    label: str,
    group: str,
    expected_seeds: tuple[int, ...],
) -> SummaryRow:
    selected = [
        row for row in rows if row.get(key_field, "").strip().lower() == key.lower()
    ]
    by_seed: dict[int, dict[str, str]] = {}
    for row in selected:
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label}: seed inválida em {row}") from error
        if seed in by_seed:
            raise ValueError(f"{label}: seed {seed} duplicada")
        by_seed[seed] = row

    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"{label}: faltam resultados para as seeds {missing}")

    ordered = [by_seed[seed] for seed in expected_seeds]
    rewards = [_number(row, "mean_reward", label) for row in ordered]
    total_served = sum(_number(row, "total_served", label) for row in ordered)
    total_arrivals = sum(_number(row, "total_arrivals", label) for row in ordered)
    total_episodes = sum(int(_number(row, "episodes", label)) for row in ordered)
    weighted_steps = sum(
        _number(row, "mean_steps", label) * _number(row, "episodes", label)
        for row in ordered
    )
    weighted_cost = sum(
        _number(row, "mean_cost", label) * _number(row, "episodes", label)
        for row in ordered
    )
    episode_counts = {int(_number(row, "episodes", label)) for row in ordered}
    if len(episode_counts) != 1:
        raise ValueError(f"{label}: número de episódios difere entre seeds")

    return SummaryRow(
        method=label,
        group=group,
        mean_reward=statistics.fmean(rewards),
        success_rate=total_served / total_arrivals if total_arrivals else 0.0,
        mean_steps=weighted_steps / total_episodes,
        mean_cost=weighted_cost / total_episodes,
        std_reward_between_seeds=statistics.stdev(rewards),
        n_seeds=len(expected_seeds),
        episodes_per_seed=episode_counts.pop(),
    )


def build_summary_table(
    agents_path: Path,
    baselines_path: Path,
    expected_seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[SummaryRow]:
    """Carrega e agrega as seis linhas obrigatórias da tabela final."""
    agent_rows = _read_csv(agents_path)
    baseline_rows = _read_csv(baselines_path)
    reward_configs = {
        row.get("reward_config", "").strip().lower() for row in agent_rows
    }
    if reward_configs != {EXPECTED_REWARD_CONFIG}:
        raise ValueError(
            "A tabela exige reward comum 'produtividade' para A, B e C. "
            "Reexecute src.agents.eval com --evaluation-reward produtividade."
        )

    summary = []
    for key_field, key, label, group in METHODS:
        rows = agent_rows if group == "agent" else baseline_rows
        summary.append(
            _aggregate_method(
                rows,
                key_field,
                key,
                label,
                group,
                expected_seeds,
            )
        )
    return summary


def export_summary(rows: list[SummaryRow], output_dir: Path) -> tuple[Path, Path]:
    """Salva versões CSV e Markdown da tabela-resumo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary_table.csv"
    markdown_path = output_dir / "summary_table.md"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "group",
                "mean_reward",
                "success_rate",
                "mean_steps",
                "mean_cost",
                "std_reward_between_seeds",
                "n_seeds",
                "episodes_per_seed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "group": row.group,
                    "mean_reward": f"{row.mean_reward:.6f}",
                    "success_rate": f"{row.success_rate:.6f}",
                    "mean_steps": f"{row.mean_steps:.6f}",
                    "mean_cost": f"{row.mean_cost:.6f}",
                    "std_reward_between_seeds": (
                        f"{row.std_reward_between_seeds:.6f}"
                    ),
                    "n_seeds": row.n_seeds,
                    "episodes_per_seed": row.episodes_per_seed,
                }
            )

    lines = [
        "| Configuração | Reward médio | Taxa de sucesso | Passos/ep | "
        "Custo acum. | Std dev entre seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.method} | {row.mean_reward:.2f} | {row.success_rate:.2%} | "
        f"{row.mean_steps:.2f} | {row.mean_cost:.2f} | "
        f"{row.std_reward_between_seeds:.2f} |"
        for row in rows
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerar tabela-resumo final")
    parser.add_argument(
        "--agents-results",
        type=Path,
        default=Path("experiments/results/eval_results.csv"),
    )
    parser.add_argument(
        "--baselines-results",
        type=Path,
        default=Path("experiments/baselines/baselines_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis/summary"),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = build_summary_table(
        args.agents_results,
        args.baselines_results,
        tuple(args.seeds),
    )
    csv_path, markdown_path = export_summary(rows, args.output_dir)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"CSV salvo: {csv_path}")
    print(f"Markdown salvo: {markdown_path}")


if __name__ == "__main__":
    main()
