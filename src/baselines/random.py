"""Baseline com política aleatória.

Seleciona ações uniformemente do espaço de ações do ambiente.
"""

from __future__ import annotations

import random as _random

import numpy as np


def select_action(
    observation: np.ndarray, action_space_size: int, rng: _random.Random
) -> int:
    """Seleciona ação aleatória uniforme.

    Args:
        observation: Vetor de observação (não utilizado, mantido para
            interface uniforme entre baselines).
        action_space_size: Número de ações disponíveis.
        rng: Gerador de números aleatórios.

    Returns:
        Índice da ação selecionada.
    """
    return rng.randint(0, action_space_size - 1)
