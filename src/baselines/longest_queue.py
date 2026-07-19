"""Baseline de fila mais longa.

Sempre atende a fila com mais chamados pendentes (ação 1).
"""

from __future__ import annotations

import numpy as np


def select_action(observation: np.ndarray, action_space_size: int) -> int:
    """Seleciona ação de fila mais longa.

    Args:
        observation: Vetor de observação (não utilizado).
        action_space_size: Número total de ações (não utilizado).

    Returns:
        Sempre retorna 1 (serve_longest).
    """
    del observation, action_space_size
    return 1
