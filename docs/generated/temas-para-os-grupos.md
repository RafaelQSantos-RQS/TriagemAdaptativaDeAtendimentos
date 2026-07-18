---
title: "Trabalho 4 — Aprendizagem por Reforço: Temas Sorteados"
subtitle: "Temas para os Grupos"
author: "Disciplina de Inteligência Artificial"
date: "2026.1"
course: "Inteligência Artificial — 2026.1"
assignment: "Trabalho 4"
institution: "Universidade do Estado da Bahia (UNEB)"
lang: pt-BR
tags:
  - grupos
  - temas
  - projetos
  - reinforcement-learning
groups_link: "https://uneboffice365.sharepoint.com/:b:/r/sites/IntelignciaArtificial2026.1/Material%20de%20Aula/Grupos-2026.1.pdf?csf=1&web=1&e=d1WmRU"
groups:
  - number: 1
    title: "Gestão de estoque perecível com demanda estocástica"
  - number: 2
    title: "Controle de bateria em microgrid com energia solar"
  - number: 3
    title: "Escalonamento de tarefas em edge/cloud"
  - number: 4
    title: "Triagem adaptativa de atendimentos"
  - number: 5
    title: "Robô coletor com bateria, recarga e alvos dinâmicos"
  - number: 6
    title: "Posicionamento tático simplificado em futebol de robôs"
---

# Trabalho 4: Aprendizagem por Reforço — Temas Sorteados

> **Relação dos grupos:** [Acesse o PDF](https://uneboffice365.sharepoint.com/:b:/r/sites/IntelignciaArtificial2026.1/Material%20de%20Aula/Grupos-2026.1.pdf?csf=1&web=1&e=d1WmRU)

## **Grupo 1 - Gestão de estoque perecível com demanda estocástica**

### Contexto

Um pequeno centro de distribuição vende produtos perecíveis. Todos os dias, o agente precisa decidir quanto comprar para atender a uma demanda incerta. Se comprar pouco, perde vendas. Se comprar muito, há desperdício.

### O que modelar

- Estado: estoque atual, idade dos produtos, demanda recente, dia da semana ou padrão de consumo.
- Ações: quantidade a comprar ou política de reposição discretizada.
- Recompensa: lucro, penalidade por falta, penalidade por desperdício, custo de estoque.
- Episódio: sequência de vários dias simulados.

### O que torna o problema interessante

O agente precisa aprender o equilíbrio entre risco de falta e risco de desperdício. A melhor ação no curto prazo pode não ser a melhor no acumulado.

### Comparações sugeridas

- baseline de reposição fixa;
- DQN ou PPO;
- duas estratégias de recompensa;
- demanda mais estável vs demanda mais variável.

## **Grupo 2 - Controle de bateria em microgrid com energia solar**

### Contexto

Uma microgrid possui painéis solares, bateria e conexão com a rede elétrica. O agente deve decidir quando carregar, descarregar ou comprar energia da rede, considerando demanda, geração solar e preço variável.

### O que modelar

- Estado: carga da bateria, demanda atual, geração solar estimada, preço da energia, horário.
- Ações: carregar, descarregar, manter, comprar da rede ou vender excedente, conforme simplificação adotada.
- Recompensa: economia financeira, penalidade por falta de energia, penalidade por desgaste da bateria.
- Episódio: um dia, uma semana ou um ciclo simulado de operação.

### O que torna o problema interessante

Há conflito entre economizar agora e preservar bateria para momentos futuros. Isso força o aluno a pensar em recompensa atrasada e política de longo prazo.

### Comparações sugeridas

- baseline que sempre carrega quando há sol;
- PPO ou DQN com ações discretizadas;
- recompensa com e sem custo de degradação da bateria;
- cenários com preços fixos vs preços variáveis.

## **Grupo 3 - Escalonamento de tarefas em edge/cloud**

### Contexto

Um sistema recebe tarefas computacionais ao longo do tempo. Cada tarefa pode ser executada localmente, enviada para a nuvem ou descartada. Cada escolha afeta latência, custo e uso de energia.

### O que modelar

- Estado: fila de tarefas, carga local, latência estimada, custo de nuvem, nível de bateria ou energia.
- Ações: executar localmente, enviar para nuvem, adiar, descartar ou priorizar tarefa.
- Recompensa: atendimento dentro do prazo, penalidade por atraso, custo computacional, penalidade por descarte.
- Episódio: sequência de requisições simuladas.

### O que torna o problema interessante

O agente precisa lidar com filas, prioridades e trade-off entre custo e desempenho. É um domínio próximo de Sistemas de Informação e Engenharia de Software.

### Comparações sugeridas

- baseline FIFO;
- baseline "sempre nuvem" ou "sempre local";
- PPO, A2C ou DQN;
- diferentes pesos para latência e custo.

## **Grupo 4 - Triagem adaptativa de atendimentos**

### Contexto

Um sistema de atendimento recebe solicitações de diferentes tipos e prioridades. O agente deve decidir qual solicitação atender, encaminhar ou deixar em espera, buscando reduzir atrasos e priorizar casos críticos.

### O que modelar

- Estado: quantidade de chamados por fila, tempo médio de espera, prioridade dos chamados, capacidade de atendimento.
- Ações: atender uma fila, encaminhar um tipo de chamado, redistribuir capacidade ou manter política atual.
- Recompensa: chamados resolvidos, penalidade por atraso, penalidade maior para casos críticos, custo de encaminhamento.
- Episódio: turno de atendimento simulado.

### O que torna o problema interessante

O agente precisa equilibrar produtividade, prioridade e justiça entre filas. Pode aprender políticas ruins se a recompensa for mal especificada.

### Comparações sugeridas

- baseline por prioridade fixa;
- baseline por fila mais longa;
- PPO ou DQN;
- recompensa que privilegia quantidade vs recompensa que privilegia prioridade.

## **Grupo 5 - Robô coletor com bateria, recarga e alvos dinâmicos**

### Contexto

Um robô opera em um ambiente em grade. Ele precisa coletar objetos, evitar obstáculos, administrar bateria e decidir quando retornar para recarregar.

### O que modelar

- Estado: posição do robô, nível de bateria, posição dos objetos, posição da estação de recarga, obstáculos.
- Ações: mover em quatro direções, coletar, recarregar, esperar.
- Recompensa: coleta bem-sucedida, penalidade por movimento, penalidade por colisão, penalidade por ficar sem bateria.
- Episódio: termina quando coleta todos os objetos, fica sem bateria ou atinge limite de passos.

### O que torna o problema interessante

É próximo ao Wumpus, mas deve ser mais complexo: múltiplos objetivos, recurso limitado, recarga e objetos em posições variáveis.

### Comparações sugeridas

- baseline que vai sempre ao objeto mais próximo;
- DQN;
- PPO se as ações forem mantidas discretas;
- ambiente com objetos fixos vs objetos sorteados.

## **Grupo 6 - Posicionamento tático simplificado em futebol de robôs**

### Contexto

Um agente controla um jogador em um campo simplificado. O objetivo é aprender a se posicionar para receber passe, bloquear adversário ou finalizar jogada, dependendo da modelagem escolhida.

### O que modelar

- Estado: posição do agente, posição da bola, posição de adversário(s), distância ao gol, posse ou não da bola.
- Ações: mover em direções discretas, aproximar da bola, afastar, chutar, marcar, dependendo da simplificação.
- Recompensa: aproximação útil, receber passe, evitar perda de bola, finalizar, bloquear avanço adversário.
- Episódio: jogada curta, ataque, defesa ou situação de bola parada.

### O que torna o problema interessante

Esse tema aproxima RL de robótica e multiagentes, mas em uma versão controlada. É importante simplificar o suficiente para treinar em CPU, mas manter o desafio de tomada de decisão sequencial.

### Comparações sugeridas

- baseline que sempre corre para a bola;
- DQN ou PPO;
- recompensa focada em bola vs recompensa focada em posicionamento;
- adversário parado vs adversário com política simples.