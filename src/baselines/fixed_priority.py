"""Baseline de prioridade fixa.

Sempre atende a fila de maior prioridade (ação 0), que internamente
seleciona a fila com maior peso de prioridade e chamados pendentes.
"""

from __future__ import annotations

import numpy as np


def select_action(observation: np.ndarray, action_space_size: int) -> int:
    """Seleciona ação de prioridade fixa.

    Args:
        observation: Vetor de observação (não utilizado).
        action_space_size: Número total de ações (não utilizado).

    Returns:
        Sempre retorna 0 (serve_priority).
    """
    del observation, action_space_size
    return 0
