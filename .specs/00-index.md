---
section: "00"
title: "Índice Mestre de Especificações"
domain: "geral"
tags: [index, master, toc, navegação]
version: "1.0"
date: "2026-07-18"
---

# 00 — Índice Mestre de Especificações

**Projeto:** Triagem Adaptativa de Atendimentos
**Grupo:** 4 — IA 2026.1
**Objetivo:** Sistema de Reinforcement Learning onde um agente aprende a alocar recursos de atendimento para múltiplas filas de chamados, equilibrando produtividade, prioridade e justiça.

---

## Estrutura dos Documentos

| # | Documento | Domínio | Descrição |
|---|---|---|---|
| 01 | `01-problem-definition.md` | Domínio do Problema | Contexto do sistema de triagem, motivação, justificativa para RL |
| 02 | `02-mdp-formulation.md` | Modelagem RL | Definição formal do MDP: estados, ações, reward, transições |
| 03 | `03-environment-specs.md` | Ambiente | Comportamento esperado do ambiente Gymnasium |
| 04 | `04-agent-specs.md` | Agentes | Algoritmos, hiperparâmetros, baselines |
| 05 | `05-experimental-protocol.md` | Experimentos | Protocolo: sementes, configurações, reprodutibilidade |
| 06 | `06-evaluation-metrics.md` | Métricas | Métricas de avaliação e critérios de sucesso |

---

## Tags por Documento

| Documento | Tags |
|---|---|
| 01 | `problema`, `domínio`, `contexto`, `motivação`, `rl`, `triagem` |
| 02 | `mdp`, `estado`, `ação`, `recompensa`, `transição`, `observation_space`, `action_space`, `reward_function` |
| 03 | `ambiente`, `gymnasium`, `reset`, `step`, `render`, `terminação`, `episódio` |
| 04 | `agente`, `ppo`, `dqn`, `algoritmo`, `hiperparâmetro`, `baseline`, `heurística` |
| 05 | `experimento`, `semente`, `seed`, `configuração`, `protocolo`, `reprodutibilidade` |
| 06 | `métrica`, `avaliação`, `curva`, `aprendizado`, `gráfico`, `análise`, `seed_surpresa` |

---

## Convenções

- **Numerados**: Arquivos prefixados por número de ordem (`01-`, `02-`, etc.)
- **Frontmatter YAML**: Cada documento possui metadados estruturados para parsing automatizado por agentes de IA
- **Alto nível**: Especificações descrevem **o que** o sistema deve fazer, nunca **como** implementar
- **Rastreabilidade**: Decisões referenciadas por ID (ex.: `D-001`)
