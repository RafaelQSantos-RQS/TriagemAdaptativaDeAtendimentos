---
title: Triagem Adaptativa de Atendimentos
subtitle: Trabalho 4 - Aprendizagem por Reforço
author: Grupo 4 - Brunna Moura, Carlos Cruz, Rafael Santos, Victor Bitencourt
date: 20 de julho de 2026
institute: Universidade do Estado da Bahia (UNEB)
theme: Madrid
colortheme: whale
lang: pt-BR
---

# Introdução

## O Problema
- Gestão de filas em sistemas de atendimento.
- Recursos limitados vs. Chegadas estocásticas.
- Conflito: Produtividade x Prioridade x Justiça.

## Por que RL?
- Tomada de decisão sequencial sob incerteza.
- Impacto de longo prazo (penalidades por espera acumulam).
- Necessidade de políticas adaptativas que superem heurísticas simples.

# Modelagem RL

## Formulação MDP
- **Estados ($S$):** Tamanhos das filas, tempos de espera, capacidade disponível (normalizados).
- **Ações ($A$):**
  - `serve_priority`: Atender maior prioridade.
  - `serve_longest`: Atender fila mais longa.
  - `refer_queue[i]`: Encaminhar chamado da fila $i$.
- **Recompensas ($R$):**
  - **Config A:** Foco em produtividade (+1 por chamado).
  - **Config B:** Foco em prioridade (peso da fila).
- **Episódios:** Turnos de 100 passos.

# Implementação

## Ambiente Gymnasium
- Classe `TriagemEnv` implementando o contrato `gym.Env`.
- Dinâmica de transição via distribuição de Poisson para chegadas.
- Renderização textual para depuração e visualização.
- Rigorosamente testado (83+ testes unitários).

# Experimentos

## Protocolo Experimental
- **Algoritmos:** PPO e DQN (Stable-Baselines3).
- **Baselines:** Aleatório, Prioridade Fixa, Fila Mais Longa.
- **Reprodutibilidade:** 5 sementes (42, 123, 256, 789, 1024).
- **Treino:** 200.000 passos por configuração.

# Resultados

## Curvas de Aprendizado
![Curvas de Aprendizado](./docs/assignment/learning_curves_comparison.png)

## Comparativo de Desempenho
![Comparação Agentes vs Baselines](./docs/assignment/agent_vs_baselines_reward.png)

## Tabela de Desempenho
| Configuração | Reward | Taxa de Sucesso |
|:---|:---:|:---:|
| **PPO Produtividade (A)** | **-78.67** | **91.76%** |
| PPO Prioridade (B) | -136.28 | 91.78% |
| DQN Produtividade (C) | -112.65 | 91.21% |
| Prioridade Fixa | -137.19 | 91.78% |
| Fila Mais Longa | -416.31 | 91.78% |
| Aleatório | -418.93 | 88.91% |

# Análise Qualitativa

## Chamados Resolvidos por Fila
![Análise por Fila](./docs/assignment/resolved_by_queue.png)

## Sucessos e Falhas
- **Sucesso:** Equilíbrio entre filas, evitando picos de espera.
- **Falha:** Sobrecarga estocástica extrema onde o recurso limitado impede a recuperação das filas.
- **Generalização:** Queda de apenas 7% em semente não vista (Boa generalização).

# Conclusão

## Conclusões
- O agente PPO superou significativamente as heurísticas de volume (Fila Mais Longa) e o Aleatório.
- Superou o DQN em estabilidade e reward final.
- Aprendeu a "ceder" em filas de baixa prioridade de forma estratégica.

## Trabalhos Futuros
- Ações de multitasking.
- Arquiteturas recorrentes (LSTM) para lidar com memória de espera.
- Penalidades dinâmicas baseadas em SLA (Service Level Agreement).

# Obrigado!

## Perguntas?
- Repositório: github.com/rafaelqsantos/TriagemAdaptativa
