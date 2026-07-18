# Agentes — Triagem Adaptativa de Atendimentos

## Visão Geral

Este documento descreve os algoritmos de RL, baselines e configurações experimentais utilizados no projeto.

---

## Algoritmos

### PPO (Proximal Policy Optimization)

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `policy` | `MlpPolicy` | Estado é vetor numérico, não imagem |
| `learning_rate` | 3e-4 | Padrão SB3, bem estabelecido para contínuo |
| `n_steps` | 2048 | Tamanho do rollout |
| `batch_size` | 64 | Tamanho do mini-batch |
| `n_epochs` | 10 | Épocas de otimização por update |
| `gamma` | 0.99 | Fator de desconto |
| `gae_lambda` | 0.95 | Fator GAE |
| `clip_range` | 0.2 | Clipping PPO |
| `ent_coef` | 0.01 | Coeficiente de entropia |
| `verbose` | 1 | Logging básico |

### DQN (Deep Q-Network)

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `policy` | `MlpPolicy` | Estado vetorial |
| `learning_rate` | 1e-3 | Taxa de aprendizado |
| `buffer_size` | 50_000 | Tamanho do replay buffer |
| `batch_size` | 32 | Tamanho do batch de treino |
| `tau` | 1.0 | Taxa de atualização da target network |
| `gamma` | 0.99 | Fator de desconto |
| `exploration_fraction` | 0.1 | Fração do treino dedicada à exploração |
| `exploration_final_eps` | 0.02 | Epsilon mínimo |
| `train_freq` | 4 | Frequência de treino (passos) |
| `verbose` | 1 | Logging básico |

---

## Baselines

| Baseline | Descrição | Estratégia |
|---|---|---|
| **Aleatório** | Ações uniformemente aleatórias | `random.choice(actions)` |
| **Prioridade Fixa** | Sempre atender a fila de maior prioridade | Ordenar filas por prioridade decrescente |
| **Fila Mais Longa** | Sempre atender a fila com mais chamados | Ordenar filas por tamanho decrescente |

---

## Configurações Experimentais

| Config | Algoritmo | Reward Func | Descrição |
|---|---|---|---|
| **A** | PPO | `reward_produtividade` | Recompensa focada em quantidade de chamados resolvidos |
| **B** | PPO | `reward_prioridade` | Recompensa com peso maior para chamados críticos |
| **C** | DQN | `reward_produtividade` | DQN com mesma reward de A para comparação entre algoritmos |

### Seed Handling

```python
import numpy as np

SEEDS = [42, 123, 256, 789, 1024]

def set_seed(seed: int):
    np.random.seed(seed)
    # + seeds para gymnasium, torch, stable-baselines3
```

---

## Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| `total_timesteps` | 200_000 |
| Número de sementes | 5 |
| Episódios de avaliação | 100 |
| Frequência de avaliação | 10_000 passos |
| Salvamento de modelo | Final do treino |

---

## Métricas de Avaliação

- Recompensa média por episódio
- Taxa de sucesso (chamados resolvidos / total)
- Número médio de passos por episódio
- Custo acumulado (penalidades)
- Desvio padrão entre sementes
- Desempenho em semente não vista (seed surpresa)
