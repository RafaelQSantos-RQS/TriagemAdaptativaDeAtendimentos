---
section: "05"
title: "Protocolo Experimental"
domain: "experimentos"
tags: [experimento, semente, seed, configuração, protocolo, reprodutibilidade, baseline, hiperparâmetro]
version: "1.0"
date: "2026-07-18"
---

# 05 — Protocolo Experimental

## 5.1 Reprodutibilidade

### Controle de Sementes

Cada execução deve ser controlada por uma semente mestre que afeta todas as fontes de aleatoriedade:

```python
SEEDS_MASTER = [42, 123, 256, 789, 1024]
```

A semente mestre deve propagar para:
- `numpy.random.seed()`
- `random.seed()`
- `gymnasium.Env.reset(seed=...)`
- `torch.manual_seed()`
- `stable_baselines3` seed parameter

### Estrutura de Diretórios

```
experiments/
└── {config}/
    ├── seed_042/
    │   ├── train.log           # Log do treinamento
    │   ├── eval_results.json   # Métricas de avaliação
    │   ├── model.zip           # Agente treinado
    │   └── learning_curve.png  # Curva de aprendizado
    ├── seed_123/
    └── ...
```

## 5.2 Parâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| total_timesteps | 200_000 |
| Número de sementes | 5 |
| Episódios de avaliação | 100 (após treino) |
| Frequência de logging | A cada 10_000 timesteps |
| Salvamento de modelo | Final do treino |

## 5.3 Configurações Experimentais

### Três configurações obrigatórias

1. **Config A**: PPO + reward produtividade
2. **Config B**: PPO + reward prioridade
3. **Config C**: DQN + reward produtividade

Cada configuração deve ser executada com as 5 sementes, totalizando **15 execuções de treino**.

## 5.4 Baselines

As baselines não requerem treinamento, mas devem ser avaliadas nas mesmas condições:

- Cada baseline executa 100 episódios por semente (5 sementes)
- Usar as mesmas seeds dos experimentos para consistency
- Coletar as mesmas métricas dos agentes treinados

## 5.5 Coleta de Dados

Durante o treinamento, registrar a cada logging step:
- Reward médio dos últimos episódios
- Episódios completados
- Loss do policy/value network (quando disponível)
- KL divergence (para PPO)
- Exploration rate (para DQN)

Durante a avaliação, registrar por episódio:
- Reward total
- Total de chamados resolvidos
- Total de encaminhamentos
- Passos até término
- Se houve overload crítico

## 5.6 Seed Surpresa

Armazenar **uma semente não usada** (ex.: 999) para testar generalização:

1. Carregar cada modelo treinado
2. Executar 100 episódios com seed = 999
3. Comparar desempenho com as 5 sementes de treino
4. Se desempenho na seed surpresa for similar → boa generalização
