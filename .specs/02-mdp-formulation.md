---
section: "02"
title: "Formulação MDP"
domain: "modelagem-rl"
tags: [mdp, estado, ação, recompensa, transição, markov, observation_space, action_space, reward_function]
version: "1.0"
date: "2026-07-18"
---

# 02 — Formulação MDP

O problema de triagem adaptativa é formalizado como um **Processo de Decisão de Markov (MDP)** definido pela tupla **⟨S, A, T, R, γ⟩**.

## 2.1 Espaço de Estados (S)

O estado representa a situação atual do sistema de atendimento. Deve conter informações suficientes para que o agente tome decisões informadas.

### Variáveis de Estado

| Variável | Tipo | Descrição |
|---|---|---|
| `queue_sizes[i]` | Inteiro | Número de chamados aguardando na fila i |
| `avg_wait_times[i]` | Contínuo | Tempo médio de espera dos chamados na fila i |
| `total_capacity` | Inteiro | Capacidade total de atendimento simultâneo |
| `used_capacity` | Inteiro | Quantos atendimentos estão ocorrendo agora |
| `step_in_episode` | Inteiro | Passo atual dentro do episódio |

### Requisitos do Estado

1. **Observabilidade completa**: O agente vê todo o estado (não há informação oculta)
2. **Normalizado**: Valores devem estar em intervalos conhecidos (importante para redes neurais)
3. **Suficiente para Markov**: Dado o estado atual, o histórico passado não adiciona informação relevante para a transição

### Exemplo de Estado (K=3 filas)

```
Estado: [5, 3, 8,          # queue_sizes: 5, 3, 8 chamados
         2.1, 0.5, 4.3,     # avg_wait_times: 2.1, 0.5, 4.3 min
         10,                 # total_capacity: 10
         6,                  # used_capacity: 6
         25]                 # step: 25 de 100
```

## 2.2 Espaço de Ações (A)

O conjunto de ações que o agente pode executar em cada passo.

### Ações Disponíveis

| Ação | Nome | Efeito Esperado |
|---|---|---|
| 0 | `serve_priority` | Atender a fila de maior prioridade que tiver chamados |
| 1 | `serve_longest` | Atender a fila com mais chamados aguardando |
| 2..K | `refer_queue[i-2]` | Encaminhar a fila i-2 para outro setor (remove 1 chamado) |

### Requisitos das Ações

1. Cada ação deve ter um efeito observável no estado
2. Ações inválidas (ex.: atender fila vazia) devem ser tratadas sem quebrar o ambiente
3. O espaço de ações deve ser discreto

## 2.3 Função de Transição (T)

Define como o estado evolui de `s_t` para `s_{t+1}` após executar `a_t`.

### Dinâmica do Ambiente

Em cada passo:

1. **Chegada de chamados**: Cada fila i tem probabilidade p_i de receber 1 novo chamado. A distribuição de chegadas segue Poisson com taxa λ_i.
2. **Execução da ação**:
   - Ação 0 ou 1: 1 chamado é removido da fila selecionada
   - Ação 2..K: 1 chamado é removido da fila (encaminhado)
3. **Atualização de espera**: O tempo de espera de chamados não atendidos aumenta
4. **Atualização de capacidade**: Recursos são liberados ou alocados

### Parâmetros de Transição

| Parâmetro | Descrição | Valor Sugerido |
|---|---|---|
| K | Número de filas | 3 |
| λ_i | Taxa de chegada da fila i | [0.3, 0.5, 0.2] |
| priority_i | Prioridade da fila i (1=baixa, 3=alta) | [1, 2, 3] |
| MAX_QUEUE | Tamanho máximo por fila | 50 |
| MAX_CAPACITY | Capacidade total | 10 |
| MAX_STEPS | Passos por episódio | 100 |

### Estocasticidade

- A chegada de chamados é estocástica (Poisson)
- Opcional: tempo de atendimento pode ser estocástico

## 2.4 Função de Recompensa (R)

Define o sinal de reforço que guia o aprendizado do agente.

### Configuração A — Reward Produtividade

```
r = Σ (chamados_resolvidos) × 1.0
  - Σ penalidade_por_atraso(i)
  - 0.5 se ação for encaminhamento
```

Foco em quantidade de chamados atendidos. Penalidades leves para evitar abandono de filas.

### Configuração B — Reward Prioridade

```
r = Σ (chamados_resolvidos_i) × priority_weight_i
  - Σ penalidade_priorizada(i)
  - 0.5 se ação for encaminhamento
```

Foco em atender chamados críticos. Prioridades mais altas têm peso maior.

### Requisitos da Recompensa

1. Deve ser calculável a cada passo (não apenas no final do episódio)
2. Deve diferenciar claramente ações boas de ações ruins
3. Suportar múltiplas configurações para comparação experimental

## 2.5 Critérios de Término

Um episódio termina quando:

| Condição | Descrição |
|---|---|
| Limite de passos | `step ≥ MAX_STEPS` (fim do turno simulado) |
| Sobrecarga crítica | Todas as filas acima de 80% por >10 passos consecutivos |

## 2.6 Fator de Desconto

γ = 0.99 — Valor padrão bem estabelecido para problemas de horizonte finito com ~100 passos.
