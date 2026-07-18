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
| **Ações** | Atender uma fila, encaminhar um tipo de chamado, redistribuir capacidade ou manter política atual |
| **Recompensa** | Chamados resolvidos, penalidade por atraso, penalidade maior para casos críticos, custo de encaminhamento |
| **Episódio** | Turno de atendimento simulado |

### Desafio

O agente precisa aprender a equilibrar **produtividade × prioridade × justiça** entre filas. Uma recompensa mal especificada pode levar a políticas indesejadas (ex.: ignorar chamados de baixa prioridade ou deixar filas longas morrerem).

---

## 👥 Equipe — Grupo 4

| Membro | Papel |
|---|---|
| *A definir* | — |
| *A definir* | — |
| *A definir* | — |
| *A definir* | — |

---

## 🛠️ Stack Tecnológica

| Tecnologia | Finalidade |
|---|---|
| **Python 3.11+** | Linguagem principal |
| **Gymnasium** | Implementação do ambiente RL |
| **Stable-Baselines3** | Algoritmos de treinamento (PPO, DQN) |
| **NumPy / Pandas** | Processamento de dados experimentais |
| **Matplotlib / Seaborn** | Visualização de curvas de aprendizado |
| **Shimmy** | Compatibilidade entre frameworks (se necessário) |

---

## 📁 Estrutura do Projeto

```
TriagemAdaptativaDeAtendimentos/
├── docs/
│   ├── generated/
│   │   ├── assignment_description.md   # Descrição oficial da atividade
│   │   ├── temas-para-os-grupos.md      # Temas sorteados para cada grupo
│   │   └── entregaveis-avaliacao.md     # Entregáveis e critérios de avaliação
│   └── professor/                       # Documentos originais (.docx)
├── src/                                 # Código-fonte (em desenvolvimento)
│   ├── env/                             # Ambiente Gymnasium
│   │   └── triagem_env.py
│   ├── agents/                          # Algoritmos e configurações de treino
│   │   ├── train.py
│   │   └── eval.py
│   ├── baselines/                       # Baseline(s) para comparação
│   │   └── heuristic.py
│   ├── analysis/                        # Análise experimental
│   │   └── plot_results.py
│   └── utils/                           # Utilitários
│       └── seeding.py
├── experiments/                         # Resultados e logs
│   └── seed_*/                          # Pastas por semente
├── models/                              # Modelos treinados (.zip)
├── notebooks/                           # Jupyter/Marimo para análise
├── requirements.txt                     # Dependências
├── README.md                            # Este arquivo
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

# Instale as dependências
uv pip install -r requirements.txt

# Ou com pip tradicional
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Como Usar

```bash
# Treinar um agente
python src/agents/train.py --algo ppo --seed 42

# Executar baseline
python src/baselines/heuristic.py --strategy priority

# Avaliar agente treinado
python src/agents/eval.py --model models/ppo_seed_42.zip --episodes 100

# Visualizar ambiente
python src/env/triagem_env.py --render human

# Gerar gráficos de análise
python src/analysis/plot_results.py --experiment experiments/seed_42/
```

---

## 📊 Protocolo Experimental

- **5 sementes** diferentes para cada configuração
- **2+ configurações** comparadas (ex.: PPO vs DQN, ou duas funções de recompensa)
- **Baseline obrigatória**: prioridade fixa ou fila mais longa
- **Análise**: curvas de aprendizado, média/desvio padrão, 3 sucessos + 3 falhas

> Detalhes completos em [`docs/generated/assignment_description.md`](docs/generated/assignment_description.md)

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

> **Nota**: Todos os documentos em `.specs/` possuem frontmatter YAML com tags para busca automatizada por agentes de IA.

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
