---
section: "03"
title: "Especificação do Ambiente"
domain: "ambiente"
tags: [ambiente, gymnasium, reset, step, render, terminação, episódio, observação, configuração]
version: "1.0"
date: "2026-07-18"
---

# 03 — Especificação do Ambiente

## 3.1 Contrato Gymnasium

O ambiente deve implementar a interface `gymnasium.Env`, expondo:

### Métodos Obrigatórios

| Método | Comportamento Esperado |
|---|---|
| `reset(seed, options)` | Reinicia o sistema para o início de um novo turno. Retorna observação inicial e info dict. Deve aceitar seed para reprodutibilidade. |
| `step(action)` | Executa um passo de simulação. Retorna (obs, reward, terminated, truncated, info). |
| `render()` | Exibe o estado atual do sistema em formato legível. Pode ser textual ou gráfico. |

### Atributos Obrigatórios

| Atributo | Comportamento Esperado |
|---|---|
| `observation_space` | Espaço contínuo (Box) conforme especificado na seção 02 |
| `action_space` | Espaço discreto (Discrete) conforme especificado na seção 02 |

## 3.2 Comportamento do reset()

O método `reset()` deve:

1. Inicializar todas as filas como vazias (queue_sizes = 0)
2. Zerar tempos de espera (avg_wait_times = 0)
3. Definir capacidade como disponível (used_capacity = 0)
4. Inicializar step counter = 0
5. Retornar a observação do estado inicial

Deve aceitar parâmetros opcionais para configuração diferente do default.

## 3.3 Comportamento do step(action)

O método `step(action)` deve executar, em ordem:

1. **Validação**: Verificar se ação é válida para o espaço de ações
2. **Processamento da ação**: Aplicar o efeito da ação escolhida no estado
3. **Chegada de novos chamados**: Amostrar de distribuição Poisson para cada fila
4. **Atualização de métricas**: Tempo de espera, capacidade usada
5. **Cálculo da recompensa**: Aplicar função de recompensa configurada
6. **Verificação de término**: Checar condições de fim de episódio
7. **Retorno**: (obs, reward, terminated, truncated, info)

Terminated = True quando condição de fim é atingida.
Truncated = False (não usamos limite externo).

## 3.4 Configurações do Ambiente

O ambiente deve ser configurável através de um dicionário de parâmetros:

| Parâmetro | Tipo | Default | Descrição |
|---|---|---|---|
| `num_queues` | int | 3 | Número de filas K |
| `arrival_rates` | list[float] | [0.3, 0.5, 0.2] | Taxas de chegada por fila (λ_i) |
| `priority_weights` | list[float] | [1.0, 2.0, 3.0] | Pesos de prioridade por fila |
| `max_queue_size` | int | 50 | Capacidade máxima por fila |
| `total_capacity` | int | 10 | Atendimentos simultâneos máximos |
| `max_steps` | int | 100 | Passos por episódio |
| `reward_config` | str | "produtividade" | Qual função de recompensa usar |

## 3.5 Visualização (render)

O ambiente deve oferecer pelo menos um modo de visualização:

### Modo Textual (render())

Exemplo de saída esperada:

```
╔════════════════════════════════════════════════════╗
║           TRIAGEM ADAPTATIVA — TURNO 25           ║
╠════════════════════════════════════════════════════╣
║ Fila 0 (prioridade 1):  ████████░░ 08 chamados    ║
║   ⏱ Espera média: 2.1 min                         ║
║ Fila 1 (prioridade 2):  █████░░░░░ 05 chamados    ║
║   ⏱ Espera média: 0.5 min                         ║
║ Fila 2 (prioridade 3):  ██████████ 10 chamados    ║
║   ⏱ Espera média: 4.3 min                         ║
╠════════════════════════════════════════════════════╣
║ Capacidade: ■■■■■■░░░░ 06/10                       ║
╚════════════════════════════════════════════════════╝
```

## 3.6 Elementos de Complexidade

O ambiente deve implementar obrigatoriamente **pelo menos 3** dos seguintes elementos:

| Elemento | Status | Como é atendido |
|---|---|---|
| Múltiplos objetivos | ✅ Obrigatório | Produtividade + prioridade + justiça |
| Restrições de recurso | ✅ Obrigatório | Capacidade limitada de atendimento |
| Incerteza/estocasticidade | ✅ Obrigatório | Chegada Poisson de chamados |
| Penalidades por ações ruins | ✅ Obrigatório | Atraso gera penalidade |
| Recompensa atrasada | ✅ Obrigatório | Atraso acumula, afeta reward futuro |
| Conflito curto vs longo prazo | ✅ Obrigatório | Atender agora vs. preservar recurso |
| Risco de fracasso antecipado | Opcional | Overload pode encerrar episódio |

## 3.7 Comportamento em Casos Limite

| Cenário | Comportamento Esperado |
|---|---|
| Tentar atender fila vazia | Reward reduzido, estado não muda |
| Fila atingir MAX_QUEUE | Novo chamado descartado, penalidade extra |
| Capacidade total exhausted | Ações de atendimento são bloqueadas |
| Episódio terminar | reset() deve ser chamado antes do próximo step |
