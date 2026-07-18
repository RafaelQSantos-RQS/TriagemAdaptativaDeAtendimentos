---
section: "01"
title: "Definição do Problema"
domain: "domínio-do-problema"
tags: [problema, domínio, contexto, motivação, rl, triagem, justificativa]
version: "1.0"
date: "2026-07-18"
---

# 01 — Definição do Problema

## 1.1 Contexto do Domínio

Um sistema de atendimento recebe solicitações (chamados) de diferentes tipos e prioridades. Cada chamado pertence a uma fila específica. O sistema possui capacidade limitada de atendimento simultâneo.

O agente deve decidir, a cada passo, qual ação tomar para gerenciar essas filas, buscando:
- **Produtividade**: Resolver o máximo de chamados possível
- **Prioridade**: Dar atenção proporcional à criticidade de cada chamado
- **Justiça**: Evitar que filas sejam negligenciadas por longos períodos

## 1.2 Motivação para Reinforcement Learning

O problema é adequado para RL porque:

| Característica | Justificativa |
|---|---|
| **Decisão sequencial** | Ações de hoje afetam o estado futuro das filas |
| **Trade-off temporal** | Atender um chamado agora vs. deixar recurso para chamados mais críticos |
| **Incerteza** | Chegada de chamados é estocástica (não determinística) |
| **Feedback atrasado** | Consequências de uma má decisão podem aparecer muitos passos depois |
| **Objetivos conflitantes** | Maximizar produtividade vs. priorizar casos críticos |

Não é viável definir uma política ótima manualmente devido à interação complexa entre filas, prioridades, capacidade e estocasticidade.

## 1.3 Objetivos do Sistema

O sistema deve ser capaz de:

1. Simular um ambiente de atendimento com múltiplas filas de chamados
2. Prover um agente RL que aprenda uma política de alocação de recursos
3. Comparar o desempenho do agente com baselines simples
4. Produzir análises quantitativas que demonstrem se o agente aprendeu ou não

## 1.4 Requisitos Funcionais (Alto Nível)

| ID | Requisito | Prioridade |
|---|---|---|
| RF-01 | O sistema deve simular K filas de chamados com prioridades distintas | Essencial |
| RF-02 | A chegada de chamados deve ser estocástica (não determinística) | Essencial |
| RF-03 | O sistema deve suportar múltiplas funções de recompensa | Essencial |
| RF-04 | O sistema deve treinar agentes usando PPO e DQN | Essencial |
| RF-05 | O sistema deve implementar pelo menos 2 baselines para comparação | Essencial |
| RF-06 | O sistema deve executar experimentos com 5 sementes diferentes | Essencial |
| RF-07 | O sistema deve produzir gráficos de curvas de aprendizado | Essencial |
| RF-08 | O sistema deve ser testável com seed não vista (seed surpresa) | Essencial |
| RF-09 | O ambiente deve oferecer visualização do estado atual | Essencial |

## 1.5 Restrições Conhecidas

- Deve executar em CPU (sem GPU obrigatória)
- Episódios limitados a N passos (turno de atendimento)
- Estado observável é completo (não parcial), mas futuro é incerto

## 1.6 Decisões de Design

| ID | Decisão | Alternativa | Justificativa |
|---|---|---|---|
| D-001 | K filas configurável | K fixo | Permite testar complexidade variável |
| D-002 | Prioridades estáticas por fila | Prioridades dinâmicas | Simplifica a modelagem inicial; prioridades fixas são mais previsíveis |
| D-003 | Capacidade global compartilhada | Capacidade por fila | Representa melhor um call center real com recursos compartilhados |
