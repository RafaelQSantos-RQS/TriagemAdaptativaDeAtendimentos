---
section: "06"
title: "Métricas de Avaliação"
domain: "métricas"
tags: [métrica, avaliação, curva, aprendizado, gráfico, análise, seed_surpresa, sucesso, falha]
version: "1.0"
date: "2026-07-18"
---

# 06 — Métricas de Avaliação

## 6.1 Métricas Primárias

### Recompensa Média por Episódio

A métrica principal de desempenho. Coletada durante treino (janela móvel) e avaliação (100 episódios).

Forma de análise:
- Curva de aprendizado: reward médio × timesteps de treino
- Média e desvio padrão entre as 5 sementes
- Comparação entre configurações no mesmo gráfico

### Taxa de Sucesso

Proporção de chamados resolvidos em relação ao total de chamados que chegaram:

```
sucesso = total_resolvido / total_chegadas
```

Interpretação:
- Quanto maior, mais produtivo é o agente
- Deve ser comparada com as baselines para validar aprendizado

## 6.2 Métricas Secundárias

### Passos por Episódio

Número médio de passos até o término do episódio. Episódios que terminam mais cedo podem indicar sobrecarga ou deadlock.

### Custo Acumulado

Soma total de penalidades recebidas em um episódio:
- Penalidades por atraso
- Penalidades por encaminhamento
- Penalidades por descarte (se implementado)

### Desvio Padrão Entre Sementes

Mede a estabilidade do aprendizado. Desvio padrão alto → algoritmo sensível à inicialização aleatória.

## 6.3 Análise Qualitativa

### Análise de 3 Episódios Bem-Sucedidos

Selecionar os 3 episódios com maior reward na avaliação e analisar:
- Distribuição de ações ao longo do episódio
- Como o agente equilibrou as filas
- Quais ações foram mais frequentes

### Análise de 3 Episódios com Falha

Selecionar os 3 episódios com menor reward e analisar:
- O que deu errado?
- O agente ignorou alguma fila?
- Houve overload? Deadlock?

## 6.4 Gráficos Obrigatórios

### Curva de Aprendizado

```
Eixo X: Timesteps de treino (0 a 200_000)
Eixo Y: Recompensa média por episódio (janela móvel)
Linhas: Média entre 5 seeds + banda de desvio padrão
```

Um gráfico por configuração (A, B, C), mais um comparativo com as 3 no mesmo plot.

### Comparação Agente vs Baselines

```
Gráfico de barras (ou box plot):
Eixo X: Agente PPO (A), PPO (B), DQN (C), Aleatório, Prioridade Fixa, Fila Longa
Eixo Y: Recompensa média por episódio na avaliação
Barras de erro: desvio padrão entre seeds
```

### Análise por Fila

```
Gráfico de barras empilhadas:
Eixo X: Cada fila (0, 1, 2)
Eixo Y: Quantidade de chamados resolvidos
Comparação: Agente PPO vs Baselines
```

## 6.5 Teste de Generalização (Seed Surpresa)

### Protocolo

1. Carregar cada modelo treinado (15 modelos: 3 configs × 5 seeds)
2. Executar 100 episódios com seed = 999
3. Comparar reward médio com o reward médio nas seeds de treino

### Critério de Sucesso

| Resultado | Interpretação |
|---|---|
| Reward surpresa ≈ Reward treino (±5%) | Agente generaliza bem |
| Reward surpresa < Reward treino (5-15%) | Generalização moderada |
| Reward surpresa ≪ Reward treino (>15%) | Agente overfitou nas seeds de treino |

## 6.6 Tabela Resumo Final

| Configuração | Reward Médio | Taxa Sucesso | Passos/Ep | Custo Acum | Std Dev |
|---|---|---|---|---|---|
| A — PPO Produtividade | — | — | — | — | — |
| B — PPO Prioridade | — | — | — | — | — |
| C — DQN Produtividade | — | — | — | — | — |
| Aleatório | — | — | — | — | — |
| Prioridade Fixa | — | — | — | — | — |
| Fila Mais Longa | — | — | — | — | — |

## 6.7 Critérios de Sucesso do Projeto

O projeto é considerado bem-sucedido se:

1. [ ] Agente PPO (A e/ou B) supera todas as baselines com significância estatística
2. [ ] Curvas de aprendizado mostram tendência de melhoria (não são planas)
3. [ ] Desvio padrão entre seeds é aceitável (< 20% da média)
4. [ ] Seed surpresa não causa degradação severa (> 15% de queda)
5. [ ] É possível identificar e explicar o comportamento aprendido pelo agente
6. [ ] Análise de sucessos e falhas revela padrões coerentes com a modelagem
