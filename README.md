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
| **Python 3.12+** | Linguagem principal |
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
│   ├── analysis/                      # Análises e relatórios experimentais
│   │   ├── learning_curves.py
│   │   ├── agent_vs_baselines.py
│   │   ├── queue_analysis.py
│   │   ├── qualitative_analysis.py
│   │   ├── surprise_seed.py
│   │   └── summary_table.py
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
│   ├── results/                       #   Resultados da avaliação de agentes
│   └── analysis/                      #   Gráficos, tabelas e relatórios
├── run_analysis.ps1                  # Todas as análises no PowerShell
├── run_analysis.sh                   # Todas as análises no Bash/Git Bash
├── run_pipeline.sh                    # Pipeline completo (treino + avaliação)
├── AGENTS.md                          # Hiperparâmetros e configurações
├── README.md
├── pyproject.toml                     # Dependências e versão do Python
├── uv.lock                            # Versões reproduzíveis das dependências
└── LICENSE
```

---

## 🚀 Instalação e Setup

### Pré-requisitos

- Python 3.12 ou superior
- [uv](https://docs.astral.sh/uv/)

### Passos

```bash
# Clone o repositório
git clone https://github.com/rafaelqsantos/TriagemAdaptativaDeAtendimentos.git
cd TriagemAdaptativaDeAtendimentos

# Crie o ambiente e instale as dependências do pyproject.toml
uv sync
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
# Avaliar todas as configs com reward comum para comparação
uv run python -m src.agents.eval --evaluation-reward produtividade

# Teste rápido
uv run python -m src.agents.eval --episodes 10

# Apenas uma config
uv run python -m src.agents.eval --config A
```

### Gerar curvas de aprendizado

Após o término dos treinamentos das configurações A, B e C:

```bash
uv run python -m src.analysis.learning_curves
```

O comando usa as cinco seeds, gera um gráfico individual por configuração,
um comparativo entre A/B/C e um CSV com as médias e desvios padrão em
`experiments/analysis/learning_curves/`.

Durante o treinamento, uma prévia com as seeds já concluídas pode ser gerada com:

```bash
uv run python -m src.analysis.learning_curves --allow-partial
```

### Comparar agentes com as baselines

Após o treino, avalie todos os agentes com a mesma função de recompensa usada
pelas baselines e gere o gráfico comparativo:

```bash
uv run python -m src.agents.eval --evaluation-reward produtividade
uv run python -m src.analysis.agent_vs_baselines
```

O gráfico de barras e o CSV agregado são salvos em
`experiments/analysis/agent_vs_baselines/`. Cada barra mostra o reward médio
entre as cinco seeds; as barras de erro mostram o desvio padrão entre seeds.

### Analisar chamados resolvidos por fila

As avaliações de agentes e baselines também registram a média de chamados
resolvidos por episódio em cada fila. Gere a análise com:

```bash
uv run python -m src.analysis.queue_analysis
```

O gráfico com um painel por fila e o CSV agregado são salvos em
`experiments/analysis/by_queue/`.

### Gerar análise qualitativa

A análise qualitativa usa, por padrão, a Config A com o modelo da seed 123,
seleciona os três maiores sucessos e as três piores falhas e registra a
distribuição de ações:

```bash
uv run python -m src.analysis.qualitative_analysis
```

Os resumos, trajetórias passo a passo, tabela de ações e `artifact.json` do
relatório são salvos em `experiments/analysis/qualitative/`.

### Avaliar uma seed surpresa

Para avaliar os 15 checkpoints em uma seed de ambiente não vista e comparar
com os resultados das seeds de treinamento:

```bash
uv run python -m src.analysis.surprise_seed --surprise-seed 999
```

Na apresentação, substitua `999` pela seed solicitada. A seed informada gera
100 seeds de episódio distintas, compartilhadas por todos os modelos e sem
interseção com as seeds de treinamento. Os CSVs, a lista auditável de seeds,
o `artifact.json` e, quando o empacotador local estiver disponível, o relatório
HTML são salvos em
`experiments/analysis/surprise_seed/seed_<seed>/`.

Para uma verificação rápida somente da Config A:

```bash
uv run python -m src.analysis.surprise_seed --surprise-seed 999 --config A --episodes 10
```

O critério de generalização segue
[as métricas especificadas](.specs/06-evaluation-metrics.md): queda de até 5%
é boa generalização, de 5% a 15% é moderada e acima de 15% é degradação
severa. Como os rewards de referência podem ser negativos, a queda relativa
usa o valor absoluto do reward de referência como denominador.

### Gerar a tabela-resumo final

Depois de avaliar agentes e baselines com 100 episódios por seed:

```bash
uv run python -m src.analysis.summary_table
```

O comando gera `summary_table.csv` e `summary_table.md` em
`experiments/analysis/summary/`. Resultados atuais:

| Configuração | Reward médio | Taxa de sucesso | Passos/ep | Custo acum. | Std dev entre seeds |
|---|---:|---:|---:|---:|---:|
| A — PPO Produtividade | -78.67 | 91.76% | 100.00 | 159.06 | 11.78 |
| B — PPO Prioridade | -136.28 | 91.78% | 100.00 | 227.88 | 10.36 |
| C — DQN Produtividade | -112.65 | 91.21% | 100.00 | 202.69 | 16.92 |
| Aleatório | -418.93 | 88.91% | 100.00 | 467.12 | 19.27 |
| Prioridade Fixa | -137.19 | 91.78% | 100.00 | 228.84 | 11.07 |
| Fila Mais Longa | -416.31 | 91.78% | 100.00 | 507.95 | 36.57 |

Definições utilizadas:

- **Taxa de sucesso:** total de chamados resolvidos dividido pelo total de
  chegadas, agregado com os numeradores e denominadores das cinco seeds.
- **Custo acumulado:** soma das penalidades por atraso, encaminhamento,
  descarte, ação inválida e capacidade indisponível; a tabela mostra a média
  por episódio.
- **Std dev:** desvio padrão amostral entre os cinco rewards médios, um por
  seed de treinamento.
- Na análise qualitativa, “sucesso” tem definição própria: episódio com reward
  total maior ou igual a zero, usada somente para selecionar os casos extremos.

### Gerar todas as análises sem treinar novamente

PowerShell:

```powershell
.\run_analysis.ps1 -SurpriseSeed 999
```

Bash ou Git Bash:

```bash
bash run_analysis.sh 999
```

Os scripts reavaliam os checkpoints e baselines, geram as curvas de
aprendizado, comparação por reward, análise por fila, análise qualitativa,
tabela-resumo e relatório da seed surpresa. A seed pode ser substituída pelo
valor solicitado na apresentação.

### Pipeline completo

```bash
# Treina A, B e C e depois gera todas as avaliações e análises
bash run_pipeline.sh
```

## Status dos requisitos experimentais

A implementação foi conferida contra o
[protocolo experimental](.specs/05-experimental-protocol.md) e as
[métricas de avaliação](.specs/06-evaluation-metrics.md).

| Requisito | Status | Evidência gerada |
|---|---|---|
| Curvas A, B e C com média de 5 seeds e banda de DP | Concluído | 3 gráficos individuais + `learning_curves_comparison.png` |
| Agentes vs baselines com barras de erro | Concluído | `agent_vs_baselines_reward.png` |
| Saídas registradas por fila | Concluído | `resolved_by_queue.png` + CSV por fila |
| Três sucessos e três falhas documentados | Concluído | relatório qualitativo + trajetórias passo a passo |
| Distribuição de ações | Concluído | tabela CSV e gráfico no relatório qualitativo |
| Seed surpresa nos 15 modelos | Concluído | seed 999, 100 episódios por checkpoint |
| Relatório de generalização | Concluído | `generalization_report.html` |
| Tabela-resumo real com seis métodos | Concluído | CSV e Markdown em `experiments/analysis/summary/` |
| Pelo menos quatro gráficos | Concluído | 6 arquivos PNG, além dos gráficos dos relatórios HTML |

### Conclusão atual da seed 999

- Config A: degradação de aproximadamente 7,0% — generalização moderada.
- Config B: degradação de aproximadamente 16,0% — degradação severa.
- Config C: degradação de aproximadamente 13,0% — generalização moderada.

### Limitações que devem ser citadas

- Uma única seed surpresa não demonstra generalização universal; ela testa uma
  coorte não vista específica.
- As barras de erro mostram dispersão entre seeds, mas ainda não constituem um
  teste formal de significância estatística.
- A análise por fila usa três painéis comparáveis, em vez de barras empilhadas.
  Isso evita tratar métodos concorrentes como partes de um mesmo total e mantém
  o desvio padrão visível.
- Os resultados existentes usam `total_served` como contador de saídas, que soma
  atendimentos locais e encaminhamentos. O ambiente também expõe
  `total_resolved` e `total_referred` para novas avaliações.
- A política escolhe entre duas regras de atendimento prontas, prioridade fixa e
  fila mais longa, além das ações de encaminhamento.
- A capacidade nominal é alocada e liberada no mesmo passo. Ela não cria disputa
  persistente por vagas nas execuções atuais.

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
| **EPIC 1** — Ambiente | ✅ Concluído | TriagemEnv registrado no Gymnasium, testes e contadores no `info` |
| **EPIC 2** — Treinamento | ✅ Concluído | Treino PPO/DQN, baselines (aleatório, prioridade fixa, fila mais longa), avaliação de agentes |
| **EPIC 3** — Análise | ✅ Concluído | Curvas, seed surpresa, análise qualitativa, gráficos e tabela-resumo |

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
