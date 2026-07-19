"""Ambiente de Triagem Adaptativa de Atendimentos — Gymnasium.

Simula um centro de atendimento com múltiplas filas de prioridades
distintas, capacidade limitada e chegada estocástica de chamados.
O agente aprende a alocar recursos entre as filas equilibrando
produtividade, prioridade e justiça.
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
    """Configuração do ambiente de triagem.

    Attributes:
        num_queues: Número de filas K.
        arrival_rates: Taxa de chegada Poisson por fila (lambda_i).
        priority_weights: Peso de prioridade por fila (maior = mais urgente).
        max_queue_size: Capacidade máxima por fila antes de descartar.
        total_capacity: Vagas simultâneas de atendimento.
        max_steps: Passos por episódio.
        reward_config: Função de recompensa ("produtividade" | "prioridade").
        wait_threshold: Passos antes do tempo de espera gerar penalidade.
        overload_threshold: Ocupação da fila que dispara overload.
        overload_patience: Passos consecutivos de overload antes de encerrar.
        penalty_referral: Custo por encaminhar chamado.
        penalty_drop: Custo por descartar chamado (fila cheia).
        delay_penalty_coeff: Escala da penalidade por atraso.
    """

    num_queues: int = 3
    arrival_rates: tuple[float, ...] = (0.3, 0.5, 0.2)
    priority_weights: tuple[float, ...] = (1.0, 2.0, 3.0)
    max_queue_size: int = 50
    total_capacity: int = 10
    max_steps: int = 100
    reward_config: str = "produtividade"
    wait_threshold: float = 3.0
    overload_threshold: float = 0.8
    overload_patience: int = 10
    penalty_referral: float = 0.5
    penalty_drop: float = 2.0
    delay_penalty_coeff: float = 0.1

    def __post_init__(self):
        n = self.num_queues
        if len(self.arrival_rates) != n:
            raise ValueError(f"arrival_rates deve ter tamanho {n}")
        if len(self.priority_weights) != n:
            raise ValueError(f"priority_weights deve ter tamanho {n}")
        if self.reward_config not in ("produtividade", "prioridade"):
            raise ValueError(f"reward_config inválido: {self.reward_config}")


class TriagemEnv(gym.Env):
    """Ambiente Gymnasium para triagem adaptativa de atendimentos.

    O agente gerencia K filas de chamados com prioridades distintas,
    capacidade limitada de atendimento e chegada estocástica. A cada
    passo decide entre atender a fila de maior prioridade, a mais longa
    ou encaminhar um chamado.

    Args:
        config: Configuração do ambiente. Usa defaults se None.
        render_mode: Modo de renderização ("human", "ansi", ou None).

    Attributes:
        observation_space: Box(2*K+3,) com tamanhos das filas, tempos de
            espera, capacidade total/usada e contador de passos.
        action_space: Discrete(K+1) com ações 0=serve_priority,
            1=serve_longest, 2..K=refer_queue[i-2].
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

        obs_dim = 2 * n + 3
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.array(
                [cfg.max_queue_size] * n
                + [cfg.max_steps * cfg.wait_threshold] * n
                + [cfg.total_capacity, cfg.total_capacity, cfg.max_steps],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n + 1)

        self._queue_sizes: np.ndarray | None = None
        self._avg_wait_times: np.ndarray | None = None
        self._total_capacity: int = cfg.total_capacity
        self._used_capacity: int = 0
        self._step: int = 0
        self._overload_counter: int = 0
        self._total_arrivals: int = 0
        self._total_served: int = 0
        self._rng: np.random.Generator | None = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reinicia o ambiente para o estado inicial.

        Esvazia todas as filas, zera tempos de espera, libera capacidade
        e reseta o contador de passos.

        Args:
            seed: Semente aleatória para reprodutibilidade.
            options: Não utilizado, mantido para compatibilidade com a API.

        Returns:
            Observação inicial e dicionário info.
        """
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
        self._total_arrivals = 0
        self._total_served = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Executa um passo de simulação.

        Pipeline: processa ação → chegadas Poisson → atualiza espera →
        libera capacidade → calcula recompensa → verifica término.

        Args:
            action: Índice da ação (0=serve_priority, 1=serve_longest,
                2..K=refer_queue[i-2]).

        Returns:
            (observação, recompensa, terminated, truncated, info).
            Truncated é sempre False (sem limite externo de tempo).
        """
        assert self._queue_sizes is not None, "chame reset() antes de step()"
        assert self.action_space.contains(action), f"Ação inválida {action}"

        cfg = self._config
        n = cfg.num_queues
        reward = 0.0
        terminated = False
        queues_served = np.zeros(n, dtype=np.int32)

        reward += self._process_action(action, queues_served)
        reward += self._simulate_arrivals()
        self._update_metrics()
        reward += self._compute_reward(queues_served)

        self._step += 1
        if self._step >= cfg.max_steps:
            terminated = True

        queue_load = self._queue_sizes / cfg.max_queue_size
        if np.all(queue_load > cfg.overload_threshold):
            self._overload_counter += 1
            if self._overload_counter >= cfg.overload_patience:
                terminated = True
        else:
            self._overload_counter = 0

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self) -> Optional[str]:
        """Renderiza o estado atual como interface textual.

        Returns:
            String ANSI se render_mode="ansi", None caso contrário
            (imprime na stdout para render_mode="human").
        """
        assert self._queue_sizes is not None, "chame reset() antes de render()"
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
            lines.append(
                f"{sep}  Fila {i} (prioridade {cfg.priority_weights[i]:.0f}):  "
                f"{bar} {size:02d} chamados  {sep}"
            )
            lines.append(f"{sep}    ⏱ Espera média: {wait:5.1f} min{pad:>27}{sep}")

        lines.append(mid)
        cap_filled = min(
            bar_len,
            int(bar_len * self._used_capacity / cfg.total_capacity),
        )
        cap_bar = "■" * cap_filled + "░" * (bar_len - cap_filled)
        lines.append(
            f"{sep}  Capacidade: {cap_bar} "
            f"{self._used_capacity:02d}/{cfg.total_capacity:02d}{pad:>18}{sep}"
        )
        lines.append(bot)

        output = "\n".join(lines)
        if self.render_mode == "ansi":
            return output
        if self.render_mode == "human":
            print(output)
        return None

    def _get_obs(self) -> np.ndarray:
        """Constrói o vetor de observação a partir do estado interno."""
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
        """Constrói o dicionário info com o estado atual."""
        assert self._queue_sizes is not None
        return {
            "queue_sizes": self._queue_sizes.copy(),
            "avg_wait_times": self._avg_wait_times.copy(),
            "used_capacity": self._used_capacity,
            "step": self._step,
            "total_arrivals": self._total_arrivals,
            "total_served": self._total_served,
        }

    def _process_action(self, action: int, queues_served: np.ndarray) -> float:
        """Processa a ação do agente e retorna a contribuição à recompensa.

        Args:
            action: Índice da ação.
            queues_served: Array de serviço por fila (mutado in-place).

        Returns:
            Contribuição de recompensa (tipicamente negativa, penalidades).
        """
        cfg = self._config
        n = cfg.num_queues
        can_serve = self._used_capacity < cfg.total_capacity
        r = 0.0

        if action in (0, 1) and not can_serve:
            r -= 0.5
        elif action == 0:
            served = self._serve_highest_priority()
            if served is not None:
                queues_served[served] = 1
                self._used_capacity += 1
            else:
                r -= 0.3
        elif action == 1:
            served = self._serve_longest_queue()
            if served is not None:
                queues_served[served] = 1
                self._used_capacity += 1
            else:
                r -= 0.3
        else:
            queue_idx = action - 2
            if queue_idx < n and self._queue_sizes[queue_idx] > 0:
                self._queue_sizes[queue_idx] -= 1
                self._total_served += 1
                r -= cfg.penalty_referral
            else:
                r -= 0.3

        return r

    def _simulate_arrivals(self) -> float:
        """Simula chegada de novos chamados (Poisson) para cada fila.

        Returns:
            Penalidade acumulada por chamados descartados (fila cheia).
        """
        cfg = self._config
        penalty = 0.0
        for i in range(cfg.num_queues):
            if self._rng is None:
                continue
            arrival = self._rng.poisson(cfg.arrival_rates[i])
            self._total_arrivals += arrival
            for _ in range(arrival):
                if self._queue_sizes[i] < cfg.max_queue_size:
                    self._queue_sizes[i] += 1
                else:
                    penalty -= cfg.penalty_drop
        return penalty

    def _update_metrics(self) -> None:
        """Atualiza tempos de espera e libera capacidade."""
        self._avg_wait_times = np.where(
            self._queue_sizes > 0, self._avg_wait_times + 1.0, 0.0
        )
        self._used_capacity = max(0, self._used_capacity - 1)

    def _serve_highest_priority(self) -> int | None:
        """Atende um chamado da fila de maior prioridade.

        Returns:
            Índice da fila atendida, ou None se todas estão vazias.
        """
        assert self._queue_sizes is not None
        cfg = self._config
        candidates = [i for i in range(cfg.num_queues) if self._queue_sizes[i] > 0]
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda i: (cfg.priority_weights[i], self._queue_sizes[i]),
        )
        self._queue_sizes[best] -= 1
        self._total_served += 1
        return best

    def _serve_longest_queue(self) -> int | None:
        """Atende um chamado da fila com mais chamados pendentes.

        Returns:
            Índice da fila atendida, ou None se todas estão vazias.
        """
        assert self._queue_sizes is not None
        candidates = [
            i for i in range(self._config.num_queues) if self._queue_sizes[i] > 0
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda i: self._queue_sizes[i])
        self._queue_sizes[best] -= 1
        self._total_served += 1
        return best

    def _compute_reward(self, queues_served: np.ndarray) -> float:
        """Calcula a recompensa conforme a função configurada.

        Dois modos:
            "produtividade": +1 por chamado atendido.
            "prioridade":   +peso_prioridade por chamado atendido.

        Ambos os modos subtraem penalidades por atraso quando o tempo de
        espera excede o threshold. Modo prioridade escala penalidades
        pelo peso da fila.

        Args:
            queues_served: Array binário indicando quais filas foram atendidas.

        Returns:
            Valor da recompensa calculada.
        """
        cfg = self._config
        reward = 0.0

        if cfg.reward_config == "produtividade":
            reward += float(queues_served.sum())
        elif cfg.reward_config == "prioridade":
            reward += float(np.dot(queues_served, cfg.priority_weights))

        for i in range(cfg.num_queues):
            wait = float(self._avg_wait_times[i])
            if wait > cfg.wait_threshold:
                if cfg.reward_config == "prioridade":
                    penalty = (
                        cfg.priority_weights[i]
                        * cfg.delay_penalty_coeff
                        * (wait - cfg.wait_threshold)
                    )
                else:
                    penalty = cfg.delay_penalty_coeff * (wait - cfg.wait_threshold)
                reward -= penalty

        return reward

    @property
    def config(self) -> TriagemConfig:
        """Retorna a configuração do ambiente."""
        return self._config
