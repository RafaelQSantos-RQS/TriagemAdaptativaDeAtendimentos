# Trabalho 4 - Aprendizagem por Reforço: Modelagem, Ambiente e Análise Experimental
## Grupo 4 - Triagem adaptativa de atendimentos
Brunna Moura, Carlos Cruz, Rafael Queiroz, Victor Bitencourt

## Descrição do problema
O projeto foca no desenvolvimento de um sistema de triagem adaptativa de atendimentos. Em um cenário onde solicitações de diferentes tipos e prioridades chegam a um centro de atendimento, o sistema deve gerenciar a distribuição dessas solicitações para minimizar atrasos e garantir o atendimento prioritário aos casos críticos, respeitando a capacidade limitada de processamento.

## Justificativa para RL
A Aprendizagem por Reforço é adequada para este problema devido à natureza estocástica da chegada de novas solicitações e à necessidade de tomada de decisão sequencial. O agente deve aprender a equilibrar trade-offs complexos entre produtividade (taxa de resolução), prioridade (atendimento a casos críticos) e justiça (tempo médio de espera nas filas), onde a decisão ótima no curto prazo pode não ser a melhor para o desempenho de longo prazo do sistema.

## Modelagem do ambiente
### Estados
Descreva o espaço de observação.

### Ações
Descreva o espaço de ação.

### Recompensas
Defina a função de recompensa.

### Episódios e Critérios de término
Como um episódio começa e termina?

## Descrição da implementação Gymnasium
Explique como o ambiente foi implementado.

## Descrição dos algoritmos usados
Descreva os algoritmos de RL utilizados (ex: PPO, DQN) e por que foram escolhidos.

## Descrição das baselines
Descreva as baselines utilizadas (ex: aleatório, prioridade fixa).

## Protocolo experimental
Detalhe o experimento: sementes, hiperparâmetros, configurações comparadas.

## Resultados
Apresente os resultados: gráficos de recompensa, tabelas comparativas.

## Análise crítica
O agente aprendeu uma política útil? Quais são os pontos fortes e fracos do comportamento aprendido?

## Limitações e melhorias futuras
O que poderia ser melhorado?

## Seção de uso de IA generativa
Informe se e como a IA foi utilizada (revisão de texto, estrutura, código).

## Referências
Liste as referências bibliográficas.
