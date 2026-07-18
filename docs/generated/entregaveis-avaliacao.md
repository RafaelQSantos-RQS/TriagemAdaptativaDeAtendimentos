---
title: "Trabalho 4 — Aprendizagem por Reforço: Entregáveis e Critérios de Avaliação"
subtitle: "Entregáveis e Avaliação"
author: "Disciplina de Inteligência Artificial"
date: "2026.1"
course: "Inteligência Artificial — 2026.1"
assignment: "Trabalho 4"
institution: "Universidade do Estado da Bahia (UNEB)"
lang: pt-BR
tags:
  - avaliação
  - entregáveis
  - relatório
  - apresentação
  - código-fonte
deliverables:
  - type: "Relatório em PDF"
    deadline: "2026-07-20T23:59"
  - type: "Código-fonte"
    deadline: "2026-07-20T23:59"
  - type: "Slides da apresentação"
    deadline: "2026-07-20T23:59"
  - type: "Demonstração"
    deadline: "2026-07-20T23:59"
presentations:
  - date: "2026-07-21"
    groups: [1, 2, 3]
  - date: "2026-07-23"
    groups: [4, 5, 6]
total_score: 8.0
---

# Trabalho 4: Aprendizagem por Reforço — Entregáveis e Critérios de Avaliação

## Entregáveis

Cada grupo deverá entregar no Teams, até **20/07 às 23:59**:

### 1. Relatório em PDF

O relatório deve conter:

- descrição do problema;
- justificativa de por que o problema é adequado para RL;
- modelagem do ambiente:
  - estados;
  - ações;
  - recompensas;
  - episódios;
  - critérios de término;
- descrição da implementação do ambiente Gymnasium;
- descrição dos algoritmos usados;
- descrição das baselines;
- protocolo experimental;
- resultados;
- análise crítica;
- limitações e melhorias futuras;
- referências.

### 2. Código-fonte

O código deve conter:

- ambiente implementado;
- scripts de treinamento;
- scripts de avaliação;
- instruções claras de execução;
- arquivo requirements.txt ou equivalente;
- configuração de sementes;
- modelos treinados ou instrução clara para reproduzir o treino.

### 3. Slides da Apresentação

Slides objetivos, focados em:

- problema;
- modelagem;
- ambiente;
- agentes;
- experimentos;
- resultados;
- falhas e conclusões.

### 4. Demonstração

Cada grupo deverá mostrar:

- o ambiente funcionando;
- ao menos uma execução de baseline;
- ao menos uma execução do agente treinado;
- gráficos ou tabela de desempenho.

## Protocolo Experimental Mínimo

Cada grupo deverá apresentar, no mínimo:

- 5 sementes diferentes;
- pelo menos 2 configurações comparadas;
- baseline simples;
- curva de recompensa média durante o treinamento;
- média e desvio padrão de desempenho final;
- análise de pelo menos 3 episódios bem-sucedidos e 3 episódios com falha;
- discussão sobre recompensa, estabilidade e generalização.

Métricas possíveis:

- recompensa média por episódio;
- taxa de sucesso;
- número médio de passos;
- custo acumulado;
- taxa de colisão ou falha;
- uso médio de recurso;
- tempo de treinamento;
- desempenho em semente não vista.

## Critérios de Avaliação Sugeridos

Total: **8,0 pontos (exercício anterior do Wumpus vale 2,0 pontos)**

- Modelagem do problema de RL e função de recompensa: **1,0**
- Implementação do ambiente Gymnasium e complexidade do domínio: **1,0**
- Treinamento dos agentes e comparação com baseline: **1,0**
- Protocolo experimental, métricas e reprodutibilidade: **1,0**
- Análise crítica dos resultados, falhas e limitações: **1,0**
- Código-fonte, organização e instruções de execução: **0,5**
- Apresentação oral, demonstração e arguição: **1,5**
- Participação crítica nas apresentações dos colegas: **1,0**

## Apresentações

As apresentações ocorrerão nos dias:

**21/07: Grupos 1, 2 e 3** e **23/07: Grupos 4, 5 e 6**

Cada grupo terá até **20 minutos** para apresentação e demonstração, seguidos de arguição.

Todos os integrantes devem estar preparados para responder perguntas. Durante a arguição, o professor poderá perguntar sobre:

- modelagem de estados e ações;
- escolha da recompensa;
- algoritmo utilizado;
- parâmetros de treinamento;
- interpretação dos gráficos;
- falhas do agente;
- diferença entre baseline e agente treinado;
- comportamento em semente não vista.

## Participação Crítica nas Apresentações dos Colegas

Cada grupo da plateia deverá entregar uma ficha curta ao final de cada dia de apresentação, contendo:

- uma observação técnica relevante sobre uma apresentação assistida;
- uma pergunta feita ou que poderia ter sido feita;
- uma comparação entre a solução observada e a própria solução do grupo.

Além disso, um grupo da plateia poderá ser sorteado para fazer a primeira pergunta em cada apresentação.

## Regras para Uso de IA Generativa

Ferramentas de IA generativa podem ser usadas como apoio para estudo, revisão de texto, organização de código ou discussão de alternativas.

No entanto, o grupo continua responsável por tudo o que entregar.

Não será aceito que o grupo apresente código, modelagem ou análise que não consiga explicar.

O relatório deve conter uma seção curta indicando:

- se o grupo usou IA generativa;
- em que etapas usou;
- como revisou e validou o que foi gerado.

## Restrições Importantes

Não serão aceitos como trabalho suficiente:

- uso direto de ambiente pronto comum, como CartPole, FrozenLake, Taxi, MountainCar, LunarLander ou Atari;
- cópia de tutorial sem adaptação substancial;
- simples execução de algoritmo sem modelagem própria;
- ambiente sem incerteza, sem recompensa atrasada e sem necessidade real de política;
- ausência de baseline;
- ausência de análise experimental;
- ausência de explicação sobre a função de recompensa.