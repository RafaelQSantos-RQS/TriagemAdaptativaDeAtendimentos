---
section: "04"
title: "Especificação dos Agentes"
domain: "agentes"
tags: [agente, ppo, dqn, algoritmo, hiperparâmetro, baseline, heurística, treinamento, avaliação]
version: "1.0"
date: "2026-07-18"
---

# 04 — Especificação dos Agentes

## 4.1 Agentes RL

O sistema deve treinar agentes usando dois algoritmos distintos:

### PPO (Proximal Policy Optimization)

**Tipo**: Policy Gradient com clipping
**Library**: Stable-Baselines3

Requisitos:
- Policy: MLP (feedforward) — entrada é vetor, não imagem
- Deve suportar espaços de observação contínuos e ação discreta
- Implementar controle de clipping para estabilidade
- Suportar GAE (Generalized Advantage Estimation) para redução de variância

### DQN (Deep Q-Network)

**Tipo**: Value-based com replay buffer
**Library**: Stable-Baselines3

Requisitos:
- Policy: MLP (feedforward)
- Deve implementar experience replay para quebrar correlação temporal
- Deve usar target network para estabilidade
- Implementar epsilon-greedy para exploração com decay

## 4.2 Hiperparâmetros

### PPO

| Parâmetro | Valor Esperado | Propósito |
|---|---|---|
| learning_rate | 3e-4 | Taxa de aprendizado padrão SB3 |
| n_steps | 2048 | Tamanho do rollout antes de cada update |
| batch_size | 64 | Mini-batch size para SGD |
| n_epochs | 10 | Épocas de otimização por update |
| gamma | 0.99 | Fator de desconto |
| gae_lambda | 0.95 | Parâmetro GAE |
| clip_range | 0.2 | Clipping para estabilidade |
| ent_coef | 0.01 | Coeficiente de entropia (exploração) |

### DQN

| Parâmetro | Valor Esperado | Propósito |
|---|---|---|
| learning_rate | 1e-3 | Taxa de aprendizado |
| buffer_size | 50_000 | Capacidade do replay buffer |
| batch_size | 32 | Batch de treino |
| gamma | 0.99 | Fator de desconto |
| exploration_fraction | 0.1 | Fração do treino com exploração |
| exploration_final_eps | 0.02 | Epsilon final |
| train_freq | 4 | Treinar a cada N passos |

## 4.3 Baselines

O sistema deve implementar **pelo menos 3 baselines**:

### Baseline 1: Aleatório

**Estratégia**: Selecionar ação uniformemente aleatória em cada passo.

```
a_t ~ Uniform(A)
```

Propósito: Estabelecer o limite inferior de desempenho. Qualquer agente treinado deve superar esta baseline.

### Baseline 2: Prioridade Fixa

**Estratégia**: Sempre atender a fila de maior prioridade que tenha chamados pendentes.

```
a_t = argmax(priority[i] × I(queue_sizes[i] > 0))
```

Propósito: Simular uma política ingênua focada apenas em criticidade.

### Baseline 3: Fila Mais Longa

**Estratégia**: Sempre atender a fila com mais chamados aguardando.

```
a_t = argmax(queue_sizes[i])
```

Propósito: Simular uma política ingênua focada apenas em volume.

## 4.4 Configurações Experimentais

O sistema deve comparar **3 configurações**:

| Config | Algoritmo | Reward | Descrição |
|---|---|---|---|
| A | PPO | Produtividade | Agente PPO treinado para maximizar quantidade |
| B | PPO | Prioridade | Agente PPO treinado para priorizar críticos |
| C | DQN | Produtividade | Agente DQN para comparar algoritmos |

## 4.5 Requisitos dos Agentes

1. Todos os agentes devem usar as mesmas 5 sementes para treino
2. Os agentes treinados devem ser salvos em formato compatível com SB3 (.zip)
3. Deve ser possível carregar e avaliar agentes pré-treinados
4. O seed handling deve cobrir: numpy, random, gymnasium, torch, SB3
