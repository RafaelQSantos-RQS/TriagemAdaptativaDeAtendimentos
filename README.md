# 🏥 TriagemAdaptativaDeAtendimentos

**Grupo 4 — Trabalho 4: Aprendizagem por Reforço**

Sistema de triagem adaptativa de atendimentos usando Reinforcement Learning. Um agente inteligente decide como alocar recursos de atendimento para diferentes filas de chamados, equilibrando produtividade, prioridade e justiça entre solicitações.

---

## 📋 Sobre o Problema

Um sistema de atendimento recebe solicitações de diferentes tipos e prioridades. O agente deve decidir **qual solicitação atender, encaminhar ou deixar em espera**, buscando reduzir atrasos e priorizar casos críticos.

### O que modelar

| Componente | Descrição |
|---|---|
| **Estado** | Quantidade de chamados por fila, tempo médio de espera, prioridade dos chamados, capacidade de atendimento |
| **Ações** | Atender fila de maior prioridade, atender fila mais longa, encaminhar chamado de uma fila específica |
| **Recompensa** | Chamados resolvidos (+1 ou +peso), penalidade por atraso, penalidade por fila cheia, custo de encaminhamento |
| **Episódio** | Turno de atendimento simulado (100 passos) |

### Desafio

O agente precisa aprender a equilibrar **produtividade × prioridade × justiça** entre filas. Uma recompensa mal especificada pode levar a políticas indesejadas (ex.: ignorar chamados de baixa prioridade ou deixar filas longas morrerem).

---

## 👥 Equipe — Grupo 4

| Membro | Papel |
|---|---|
| Rafael Santos | Desenvolvimento |
| — | — |
| — | — |
| — | — |

---

## 🛠️ Stack Tecnológica

| Tecnologia | Finalidade |
|---|---|
| **Python 3.11+** | Linguagem principal |
| **Gymnasium** | Implementação do ambiente RL |
| **Stable-Baselines3** | Algoritmos de treinamento (PPO, DQN) |
| **NumPy** | Processamento de dados experimentais |
| **Matplotlib / Seaborn** | Visualização de curvas de aprendizado |
| **Ruff** | Linter e formatador |
| **pytest** | Testes automatizados |

---

## 📁 Estrutura do Projeto

```
TriagemAdaptativaDeAtendimentos/
├── .specs/                            # Especificações técnicas (MDP, ambiente,
│   ├── 00-index.md                    #   agentes, protocolo experimental)
│   ├── 01-problem-definition.md
│   ├── 02-mdp-formulation.md
│   ├── 03-environment-specs.md
│   ├── 04-agent-specs.md
│   ├── 05-experimental-protocol.md
│   └── 06-evaluation-metrics.md
├── docs/
│   ├── generated/
│   │   ├── assignment_description.md
│   │   ├── temas-para-os-grupos.md
│   │   └── entregaveis-avaliacao.md
│   └── professor/                     # Documentos originais (.docx)
├── src/
│   ├── environment/                   # Ambiente Gymnasium
│   │   ├── __init__.py                #   Registra "TriagemAdaptativa-v0"
│   │   └── triagem_env.py            #   Implementação do ambiente
│   ├── agents/                        # Algoritmos de treinamento
│   │   ├── train.py                   #   Script de treino (PPO/DQN)
│   │   └── eval.py                    #   Avaliação de agentes treinados
│   ├── baselines/                     # Baselines para comparação
│   │   ├── random.py                  #   Ações uniformemente aleatórias
│   │   ├── fixed_priority.py          #   Sempre atende fila de maior prioridade
│   │   ├── longest_queue.py           #   Sempre atende fila com mais chamados
│   │   └── run.py                     #   Avaliação de todas as baselines
│   ├── analysis/                      # Análise experimental
│   │   └── (em desenvolvimento)
│   └── utils/                         # Utilitários
│       └── (em desenvolvimento)
├── tests/                             # Testes automatizados
│   ├── conftest.py                    #   Fixtures compartilhadas
│   ├── test_environment.py            #   Testes do ambiente
│   ├── test_baselines.py              #   Testes das baselines
│   └── test_info_counters.py          #   Testes dos contadores do info dict
├── models/                            # Modelos treinados (.zip)
│   └── config_{label}/seed_{NNN}/
├── experiments/                       # Resultados e logs
│   ├── config_{label}/seed_{NNN}/     #   Logs de treino
│   ├── baselines/                     #   Resultados das baselines
│   └── results/                       #   Resultados da avaliação de agentes
├── run_pipeline.sh                    # Pipeline completo (treino + avaliação)
├── AGENTS.md                          # Hiperparâmetros e configurações
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 Instalação e Setup

### Pré-requisitos

- Python 3.11 ou superior
- [uv](https://docs.astral.sh/uv/) (recomendado) ou pip

### Passos

```bash
# Clone o repositório
git clone https://github.com/rafaelqsantos/TriagemAdaptativaDeAtendimentos.git
cd TriagemAdaptativaDeAtendimentos

# Crie e ative o ambiente virtual (com uv — recomendado)
uv venv
source .venv/bin/activate

# Instale o pacote em modo editable (registra o ambiente Gymnasium)
uv pip install -e .

# Instale dependências adicionais
uv pip install -r requirements.txt
```

### Dependências Opcionais

```bash
# TensorBoard — logs de treino
uv pip install tensorboard

# Ruff — lint e formatação
uv pip install ruff
```

---

## ▶️ Como Usar

### Treinar um agente

```bash
# Treino completo (200k steps) com seed única
python -m src.agents.train --seed 42

# Todas as 5 sementes (42, 123, 256, 789, 1024)
python -m src.agents.train

# Algoritmo e configuração específicos
python -m src.agents.train --algo ppo --config A --seed 42 --total-timesteps 50000

# Com TensorBoard
python -m src.agents.train --seed 42 --tensorboard tb_logs/
```

### Executar testes

```bash
# Todos os testes
uv run pytest tests/ -v

# Apenas testes específicos
uv run pytest tests/ -k "test_reset"

# Com cobertura
uv run pytest tests/ --cov=src.environment
```

### Verificar lint e formatação

```bash
# Verificar lint
uv run ruff check .

# Formatar código
uv run ruff format .

# Verificação completa
uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -q
```

### Avaliar baselines

```bash
# Avaliar todas as baselines (100 episódios × 5 seeds = 1500 episódios)
uv run python -m src.baselines.run

# Teste rápido (10 episódios)
uv run python -m src.baselines.run --episodes 10
```

### Avaliar agentes treinados

```bash
# Avaliar todas as configs (100 episódios × 5 seeds × 3 configs = 1500 episódios)
uv run python -m src.agents.eval

# Teste rápido
uv run python -m src.agents.eval --episodes 10

# Apenas uma config
uv run python -m src.agents.eval --config A
```

### Pipeline completo

```bash
# Treina todas as configs (A, B, C) + avalia baselines
bash run_pipeline.sh
```

### Visualizar o ambiente

```bash
# Usando gymnasium.make (registrado)
uv run python -c "
import gymnasium as gym
import src.environment  # noqa: F401
env = gym.make('TriagemAdaptativa-v0', render_mode='human')
env.reset()
for _ in range(5):
    env.step(env.action_space.sample())
"

# Modo ANSI (saída textual)
uv run python -c "
import gymnasium as gym
import src.environment
env = gym.make('TriagemAdaptativa-v0', render_mode='ansi')
env.reset()
print(env.render())
"
```

---

## 📊 Protocolo Experimental

### Algoritmos

- **PPO** (Proximal Policy Optimization) — `MlpPolicy`, learning_rate=3e-4, n_steps=2048
- **DQN** (Deep Q-Network) — `MlpPolicy`, learning_rate=1e-3, buffer_size=50_000

### Baselines

- **Aleatório**: ações uniformemente aleatórias
- **Prioridade Fixa**: sempre atender a fila de maior prioridade
- **Fila Mais Longa**: sempre atender a fila com mais chamados

### Configurações

| Config | Algoritmo | Função de Recompensa |
|---|---|---|
| **A** | PPO | Produtividade (+1 por chamado atendido) |
| **B** | PPO | Prioridade (peso da fila por chamado) |
| **C** | DQN | Produtividade (para comparação PPO × DQN) |

### Reprodutibilidade

- **5 sementes**: 42, 123, 256, 789, 1024
- **200.000 timesteps** por seed
- **Avaliação** a cada 10.000 passos (10 episódios durante treino, 100 episódios na avaliação final)
- **3 configurações**: A (PPO + produtividade), B (PPO + prioridade), C (DQN + produtividade)
- **3 baselines**: aleatório, prioridade fixa, fila mais longa (100 episódios × 5 seeds cada)
- Diretórios: `models/config_{label}/seed_{NNN}/`, `experiments/config_{label}/seed_{NNN}/`, `experiments/baselines/`

> Detalhes completos em [`AGENTS.md`](AGENTS.md) e [`.specs/05-experimental-protocol.md`](.specs/05-experimental-protocol.md)

---

## ✅ Status do Projeto

| Épico | Status | Descrição |
|---|---|---|
| **EPIC 0** — Especificações | ✅ Concluído | Definição do problema, MDP, ambiente, agente, protocolo |
| **EPIC 1** — Ambiente | ✅ Concluído | TriagemEnv com 83 testes, gymnasium-registrado, edge cases |
| **EPIC 2** — Treinamento | ✅ Concluído | Treino PPO/DQN, baselines (aleatório, prioridade fixa, fila mais longa), avaliação de agentes |
| **EPIC 3** — Análise | 📋 Planejado | Curvas de aprendizado, seed surpresa, gráficos comparativos |

---

## 📐 Documentação de Desenvolvimento

| Documento | Descrição |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Algoritmos, hiperparâmetros, baselines e configurações experimentais |
| [`.specs/00-index.md`](.specs/00-index.md) | Índice master de todas as especificações |
| [`.specs/01-problem-definition.md`](.specs/01-problem-definition.md) | Definição do problema e justificativa RL |
| [`.specs/02-mdp-formulation.md`](.specs/02-mdp-formulation.md) | Formulação MDP: estados, ações, recompensas |
| [`.specs/03-environment-specs.md`](.specs/03-environment-specs.md) | Especificação do ambiente Gymnasium |
| [`.specs/04-agent-specs.md`](.specs/04-agent-specs.md) | Especificação dos agentes e baselines |
| [`.specs/05-experimental-protocol.md`](.specs/05-experimental-protocol.md) | Protocolo experimental e reprodutibilidade |
| [`.specs/06-evaluation-metrics.md`](.specs/06-evaluation-metrics.md) | Métricas de avaliação e critérios de sucesso |

> **Nota**: Todos os documentos em `.specs/` possuem frontmatter YAML com tags para busca automatizada.

---

## 📚 Documentação de Apoio

| Documento | Descrição |
|---|---|
| [`assignment_description.md`](docs/generated/assignment_description.md) | Descrição completa da atividade, regras e objetivos |
| [`temas-para-os-grupos.md`](docs/generated/temas-para-os-grupos.md) | Temas de todos os grupos (nosso é o Grupo 4) |
| [`entregaveis-avaliacao.md`](docs/generated/entregaveis-avaliacao.md) | Entregáveis, prazos, critérios de avaliação |

---

## 📅 Cronograma

| Data | Evento |
|---|---|
| **20/07 — 23:59** | Entrega (relatório PDF + código + slides) |
| **23/07** | Apresentação do Grupo 4 |
| **23/07** | Participação crítica nas apresentações dos colegas |

---

## 📄 Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais informações.
