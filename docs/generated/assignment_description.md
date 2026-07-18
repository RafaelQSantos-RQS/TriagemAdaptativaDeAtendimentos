---
title: "Trabalho 4 — Aprendizagem por Reforço: Modelagem, Ambiente e Análise Experimental"
subtitle: "Descrição da Atividade"
author: "Disciplina de Inteligência Artificial"
date: "2026.1"
course: "Inteligência Artificial — 2026.1"
assignment: "Trabalho 4"
institution: "Universidade do Estado da Bahia (UNEB)"
lang: pt-BR
tags:
  - aprendizagem-por-reforço
  - gymnasium
  - stable-baselines3
  - reinforcement-learning
  - python
  - ia
abstract: |
  Quarto e último trabalho prático da disciplina de Inteligência Artificial.
  Foco em Aprendizagem por Reforço: modelagem de problemas como tarefas de RL,
  implementação de ambientes Gymnasium, treinamento de agentes com Stable-Baselines3
  e análise experimental comparativa.
---

# Trabalho 4 — Aprendizagem por Reforço: Modelagem, Ambiente e Análise Experimental
Bem-vindos(as) ao quarto e último trabalho prático da disciplina de Inteligência Artificial.
--------------------------------------------------------------------------------------------

Neste trabalho, o foco será **Aprendizagem por Reforço**. Diferente dos trabalhos anteriores, aqui o objetivo não é aprender a partir de exemplos rotulados, nem apenas recuperar informação de uma base documental. O desafio é construir um agente que aprende por interação com um ambiente, recebendo recompensas ou penalidades pelas consequências de suas ações.

A proposta é que cada grupo projete um problema, implemente um ambiente compatível com Gymnasium, treine um ou mais agentes de aprendizagem por reforço e analise criticamente os resultados obtidos.

Não será suficiente apenas executar um exemplo pronto de CartPole, FrozenLake, Taxi, LunarLander, MountainCar ou algum tutorial comum disponível na internet. O trabalho deve envolver **modelagem própria do ambiente**, definição clara de estados, ações e recompensas, e uma análise experimental consistente.

O exemplo do Mundo do Wumpus apresentado em sala serve como referência de complexidade mínima. O trabalho de cada grupo deve ir além dele, seja por maior riqueza do ambiente, múltiplos objetivos, incerteza, restrições operacionais, reward design mais elaborado ou comparação experimental mais cuidadosa.

Objetivos de aprendizagem
-------------------------

Ao final deste trabalho, espera-se que os grupos sejam capazes de:

*   modelar um problema como tarefa de Aprendizagem por Reforço;
*   definir estados, ações, recompensas, episódios e critérios de término;
*   implementar um ambiente compatível com Gymnasium;
*   treinar agentes usando Stable-Baselines3 ou outra biblioteca apropriada;
*   comparar o desempenho do agente treinado com uma baseline simples;
*   executar experimentos com múltiplas sementes e configurações;
*   analisar curvas de aprendizado, estabilidade, falhas e comportamento aprendido;
*   discutir criticamente se o agente realmente aprendeu uma política útil.

Organização dos grupos
======================

A turma será organizada nos mesmos **6 grupos de 4 pessoas**.

Cada grupo receberá um projeto diferente.

Todos os projetos deverão ser implementados em Python e deverão usar, preferencialmente:

*   gymnasium para implementação do ambiente;
*   stable-baselines3 para treinamento de agentes;
*   numpy, pandas, matplotlib ou bibliotecas equivalentes para análise dos resultados.

Outras bibliotecas de RL poderão ser usadas, desde que o grupo justifique a escolha.

Regras gerais obrigatórias
==========================

1\. Ambiente próprio
--------------------

Cada grupo deverá implementar um ambiente próprio compatível com Gymnasium.

O ambiente deverá conter, no mínimo:

*   observation\_space;
*   action\_space;
*   método reset();
*   método step();
*   critério de término do episódio;
*   função de recompensa;
*   algum tipo de visualização simples, como render() textual, gráfico, grid ou animação.

Não é obrigatório fazer uma visualização bonita, mas deve ser possível entender o comportamento do agente.

2\. Complexidade mínima
-----------------------

O ambiente não pode ser trivial. Deve conter pelo menos três dos seguintes elementos:

*   múltiplos objetivos;
*   restrições de recurso;
*   incerteza ou eventos estocásticos;
*   penalidades por ações ruins;
*   recompensa atrasada;
*   necessidade de planejamento de vários passos;
*   conflitos entre curto e longo prazo;
*   estados parcialmente informativos;
*   dinâmica que muda entre episódios;
*   custo de ação;
*   risco de fracasso ou término antecipado.

3\. Baseline obrigatória
------------------------

Cada grupo deverá comparar o agente treinado com pelo menos uma baseline simples, por exemplo:

*   agente aleatório;
*   agente guloso simples;
*   regra heurística definida pelo grupo;
*   política manual razoável.

O objetivo é mostrar se o agente treinado realmente aprendeu algo útil.

4\. Comparação experimental obrigatória
---------------------------------------

Além da baseline, cada grupo deve comparar pelo menos **duas configurações**.

Exemplos:

*   DQN vs PPO;
*   PPO vs A2C;
*   DQN com duas arquiteturas de rede;
*   mesma técnica com duas funções de recompensa;
*   mesmo agente com diferentes hiperparâmetros;
*   política treinada em ambiente simples vs ambiente mais difícil;
*   treinamento com e sem determinada informação no estado.

5\. Múltiplas sementes
----------------------

Cada grupo deverá executar os experimentos com pelo menos **5 sementes diferentes**.

A documentação do Stable-Baselines3 alerta que resultados totalmente reprodutíveis não são garantidos entre diferentes plataformas, versões de PyTorch ou CPU/GPU, mas recomenda controle de semente para melhorar a reprodutibilidade em uma mesma configuração.

6\. Demonstração com semente surpresa
-------------------------------------

Na apresentação, o grupo deverá estar preparado para executar a política treinada em uma instância ou semente escolhida na hora pelo professor.

Isso não é para “pegar o grupo de surpresa”, mas para verificar se a solução não foi ajustada apenas para um caso fixo.
