"""Ambiente de Triagem Adaptativa de Atendimentos — Gymnasium.

Simula um sistema de atendimento com múltiplas filas de chamados de
diferentes prioridades. O agente decide como alocar recursos limitados
para atender, encaminhar ou priorizar filas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec


@dataclass
class TriagemConfig:
    """Configuração do ambiente de triagem."""

    num_queues: int = 3
    arrival_rates: tuple[float, ...] = (0.3, 0.5, 0.2)
    priority_weights: tuple[float, ...] = (1.0, 2.0, 3.0)
    max_queue_size: int = 50
    total_capacity: int = 10
    max_steps: int = 100
    reward_config: str = "produtividade"  # "produtividade" | "prioridade"

    # Limiares para penalidades
    wait_threshold: float = 3.0
    overload_threshold: float = 0.8  # % da capacidade
    overload_patience: int = 10  # passos antes de encerrar por overload

    # Penalidades
    penalty_referral: float = 0.5
    penalty_drop: float = 2.0
    delay_penalty_coeff: float = 0.1

    def __post_init__(self):
        n = self.num_queues
        if len(self.arrival_rates) != n:
            raise ValueError(f"arrival_rates must have length {n}")
        if len(self.priority_weights) != n:
            raise ValueError(f"priority_weights must have length {n}")
        if self.reward_config not in ("produtividade", "prioridade"):
            raise ValueError(f"unknown reward_config: {self.reward_config}")


class TriagemEnv(gym.Env):
    """Ambiente de triagem adaptativa de atendimentos.

    O agente gerencia K filas de chamados com prioridades distintas,
    capacidade limitada de atendimento e chegada estocástica.

    Parâmetros
    ----------
    config : TriagemConfig
        Configuração do ambiente.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[TriagemConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.spec = EnvSpec(
            id="TriagemAdaptativa-v0",
            entry_point="src.environment.triagem_env:TriagemEnv",
        )
        self._config = config or TriagemConfig()
        cfg = self._config
        n = cfg.num_queues

        # --- Espaço de observação ---
        # queue_sizes[n] + avg_wait_times[n] + capacity + used + step
        obs_dim = 2 * n + 3
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.array(
            [cfg.max_queue_size] * n
            + [cfg.max_steps * cfg.wait_threshold] * n  # wait times
            + [cfg.total_capacity, cfg.total_capacity, cfg.max_steps],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Espaço de ações ---
        # 0: serve_priority, 1: serve_longest, 2..K: refer_queue[i-2]
        self.action_space = spaces.Discrete(n + 1)

        # --- Estado interno ---
        self._queue_sizes: np.ndarray | None = None
        self._avg_wait_times: np.ndarray | None = None
        self._total_capacity: int = cfg.total_capacity
        self._used_capacity: int = 0
        self._step: int = 0
        self._overload_counter: int = 0
        self._rng: np.random.Generator | None = None

    # ──────────────────────────────── reset ────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._np_random is None:
            import random as _random

            self._np_random = np.random.default_rng(
                _random.SystemRandom().randint(0, 2**31 - 1)
            )
        self._rng = np.random.default_rng(self._np_random.bit_generator)

        n = self._config.num_queues
        self._queue_sizes = np.zeros(n, dtype=np.int32)
        self._avg_wait_times = np.zeros(n, dtype=np.float32)
        self._used_capacity = 0
        self._step = 0
        self._overload_counter = 0

        return self._get_obs(), self._get_info()

    # ──────────────────────────────── step ─────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._queue_sizes is not None, "call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action {action}"

        cfg = self._config
        n = cfg.num_queues
        reward = 0.0
        terminated = False

        # 1. Processar ação
        queues_served = np.zeros(n, dtype=np.int32)

        can_serve = self._used_capacity < cfg.total_capacity

        if action in (0, 1) and not can_serve:
            reward -= 0.5  # penalidade por tentar servir sem capacidade
        elif action == 0:  # serve_priority: atender fila de maior prioridade
            served = self._serve_highest_priority()
            if served is not None:
                queues_served[served] = 1
                self._used_capacity += 1
            else:
                reward -= 0.3  # penalidade por tentar atender fila vazia
        elif action == 1:  # serve_longest: atender fila mais longa
            served = self._serve_longest_queue()
            if served is not None:
                queues_served[served] = 1
                self._used_capacity += 1
            else:
                reward -= 0.3  # penalidade por tentar atender fila vazia
        else:  # 2..K: refer_queue, encaminhar chamado
            queue_idx = action - 2
            if queue_idx < n and self._queue_sizes[queue_idx] > 0:
                self._queue_sizes[queue_idx] -= 1
                reward -= cfg.penalty_referral
            else:
                reward -= 0.3  # penalidade por encaminhamento inválido

        # 2. Chegada de novos chamados (Poisson)
        for i in range(n):
            if self._rng is None:
                continue  # safety check, should not happen
            arrival = self._rng.poisson(cfg.arrival_rates[i])
            for _ in range(arrival):
                if self._queue_sizes[i] < cfg.max_queue_size:
                    self._queue_sizes[i] += 1
                else:
                    reward -= cfg.penalty_drop  # chamado descartado

        # 3. Atualizar tempos de espera
        self._avg_wait_times = np.where(
            self._queue_sizes > 0, self._avg_wait_times + 1.0, 0.0
        )

        # 4. Liberar capacidade (simula fim de atendimentos)
        self._used_capacity = max(0, self._used_capacity - 1)

        # 5. Calcular recompensa
        reward += self._compute_reward(queues_served)

        # 6. Verificar término
        self._step += 1
        if self._step >= cfg.max_steps:
            terminated = True

        # Overload crítico: todas as filas acima do threshold por N passos
        queue_load = self._queue_sizes / cfg.max_queue_size
        if np.all(queue_load > cfg.overload_threshold):
            self._overload_counter += 1
            if self._overload_counter >= cfg.overload_patience:
                terminated = True
        else:
            self._overload_counter = 0

        return self._get_obs(), reward, terminated, False, self._get_info()

    # ──────────────────────────────── render ──────────────────────────────

    def render(self) -> Optional[str]:
        assert self._queue_sizes is not None, "call reset() before render()"
        cfg = self._config
        n = cfg.num_queues

        bar_len = 20
        pad = " " * 4
        sep = "║"
        top = "╔" + "═" * 58 + "╗"
        mid = "╠" + "═" * 58 + "╣"
        bot = "╚" + "═" * 58 + "╝"

        lines: list[str] = [top]
        title = f"{pad}TRIAGEM ADAPTATIVA — PASSO {self._step:03d}"
        lines.append(f"{sep}{title}{pad:>27}{sep}")
        lines.append(mid)

        for i in range(n):
            size = int(self._queue_sizes[i])
            wait = float(self._avg_wait_times[i])
            filled = min(bar_len, int(bar_len * size / cfg.max_queue_size))
            bar = "█" * filled + "░" * (bar_len - filled)
            queue_line = (
                f"{sep}  Fila {i} (prioridade {cfg.priority_weights[i]:.0f}):  "
                f"{bar} {size:02d} chamados  {sep}"
            )
            wait_line = f"{sep}    ⏱ Espera média: {wait:5.1f} min{pad:>27}{sep}"
            lines.append(queue_line)
            lines.append(wait_line)

        lines.append(mid)
        cap_filled = min(
            bar_len,
            int(bar_len * self._used_capacity / cfg.total_capacity),
        )
        cap_bar = "■" * cap_filled + "░" * (bar_len - cap_filled)
        cap_line = (
            f"{sep}  Capacidade: {cap_bar} "
            f"{self._used_capacity:02d}/{cfg.total_capacity:02d}{pad:>18}{sep}"
        )
        lines.append(cap_line)
        lines.append(bot)

        output = "\n".join(lines)
        if self.render_mode == "ansi":
            return output
        if self.render_mode == "human":
            print(output)
        return None

    # ──────────── Métodos internos ────────────

    def _get_obs(self) -> np.ndarray:
        assert self._queue_sizes is not None
        assert self._avg_wait_times is not None
        return np.concatenate(
            [
                self._queue_sizes.astype(np.float32),
                self._avg_wait_times.astype(np.float32),
                np.array(
                    [
                        float(self._total_capacity),
                        float(self._used_capacity),
                        float(self._step),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _get_info(self) -> dict[str, Any]:
        assert self._queue_sizes is not None
        return {
            "queue_sizes": self._queue_sizes.copy(),
            "avg_wait_times": self._avg_wait_times.copy(),
            "used_capacity": self._used_capacity,
            "step": self._step,
        }

    def _serve_highest_priority(self) -> int | None:
        """Atende 1 chamado da fila de maior prioridade com chamados."""
        assert self._queue_sizes is not None
        cfg = self._config
        # Filtrar filas com chamados, ordenar por prioridade (decrescente)
        candidates = [i for i in range(cfg.num_queues) if self._queue_sizes[i] > 0]
        if not candidates:
            return None
        # Desempate: maior prioridade; se empatar, maior fila
        best = max(
            candidates,
            key=lambda i: (cfg.priority_weights[i], self._queue_sizes[i]),
        )
        self._queue_sizes[best] -= 1
        return best

    def _serve_longest_queue(self) -> int | None:
        """Atende 1 chamado da fila com mais chamados."""
        assert self._queue_sizes is not None
        candidates = [
            i for i in range(self._config.num_queues) if self._queue_sizes[i] > 0
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda i: self._queue_sizes[i])
        self._queue_sizes[best] -= 1
        return best

    def _compute_reward(self, queues_served: np.ndarray) -> float:
        """Calcula a recompensa com base na config selecionada."""
        cfg = self._config
        reward = 0.0

        if cfg.reward_config == "produtividade":
            reward += float(queues_served.sum()) * 1.0
        elif cfg.reward_config == "prioridade":
            reward += float(
                np.dot(queues_served, cfg.priority_weights)  # type: ignore[arg-type]
            )

        # Penalidade por atraso
        for i in range(cfg.num_queues):
            wait = float(self._avg_wait_times[i])
            if wait > cfg.wait_threshold:
                if cfg.reward_config == "prioridade":
                    reward -= (
                        cfg.priority_weights[i]
                        * cfg.delay_penalty_coeff
                        * (wait - cfg.wait_threshold)
                    )
                else:
                    reward -= cfg.delay_penalty_coeff * (wait - cfg.wait_threshold)

        return reward

    @property
    def config(self) -> TriagemConfig:
        return self._config
