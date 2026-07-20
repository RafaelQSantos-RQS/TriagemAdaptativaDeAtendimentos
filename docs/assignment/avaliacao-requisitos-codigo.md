# Avaliação dos requisitos e do código

## Triagem adaptativa de atendimentos, Grupo 4

**Data da revisão:** 20 de julho de 2026  
**Objetivo:** apoiar a decisão do grupo sobre mudanças no código antes da apresentação.  
**Escopo:** enunciado do professor, arquivos `.specs`, ambiente, agentes, baselines, protocolo experimental, resultados e instruções de execução. Este documento não avalia slides nem qualidade da apresentação oral.

## Parecer

O projeto atende a maior parte dos requisitos formais. A equipe implementou um ambiente Gymnasium próprio, três configurações de agentes, três baselines, cinco sementes, curvas de aprendizado, avaliação final, análise qualitativa e teste com seed surpresa.

Dois problemas afetam a interpretação do ambiente:

1. a capacidade de atendimento não restringe as execuções padrão;
2. o código inclui encaminhamentos no contador de chamados resolvidos.

Uma correção na capacidade, no tempo de espera ou no espaço de ações mudaria o MDP. A equipe precisaria treinar os 15 agentes e gerar todos os resultados de novo. Como a entrega ocorre hoje, a equipe escolheu preservar o ambiente usado nos experimentos, explicar as limitações e aplicar apenas ajustes que não alteram os modelos.

## Decisão aplicada

A equipe executou a Opção B. O ambiente agora expõe `total_resolved`, `total_referred`, contadores por fila e `terminated_by_overload` no `info`. Os campos legados continuam disponíveis para preservar os scripts e os resultados existentes. A mudança não altera observações, ações, transições ou recompensa.

O README registra Python 3.12, instalação com `uv sync`, conclusão das análises e as limitações de capacidade, encaminhamento e ações heurísticas. O relatório e os slides usam “taxa de saída registrada” ao interpretar os números gerados com `total_served`.

## Requisitos do professor

| Requisito | Situação | Evidência | Risco |
|---|---|---|---|
| Ambiente próprio compatível com Gymnasium | Atende | `src/environment/triagem_env.py` | Baixo |
| `observation_space`, `action_space`, `reset()` e `step()` | Atende | classe `TriagemEnv` | Baixo |
| Critérios de término | Atende | horizonte de 100 passos e sobrecarga | Baixo |
| Visualização | Atende | modos `ansi` e `human` | Baixo |
| Incerteza | Atende | chegadas Poisson por fila | Baixo |
| Recompensa atrasada e penalidades | Atende | atraso, descarte, encaminhamento e ação inválida | Baixo |
| Restrição de recurso | Parcial | capacidade existe no estado, mas não limita episódios iniciados em zero | Alto |
| Baseline simples | Atende | aleatória, prioridade fixa e fila mais longa | Baixo |
| Duas configurações | Atende | A, B e C | Baixo |
| Cinco sementes | Atende | 42, 123, 256, 789 e 1024 | Baixo |
| Curva de recompensa média | Atende | quatro gráficos de aprendizado | Baixo |
| Média e desvio padrão final | Atende | CSV e tabela-resumo | Baixo |
| Três sucessos e três falhas | Atende | relatório qualitativo e trajetórias | Baixo |
| Discussão de generalização | Atende | seed surpresa 999 | Baixo |
| Scripts de treino e avaliação | Atende | `src/agents/train.py` e `src/agents/eval.py` | Baixo |
| Dependências | Atende com ressalva | `pyproject.toml` e `uv.lock` existem; README cita arquivo ausente | Médio |
| Modelos ou instrução de reprodução | Atende com ressalva | pipeline existe; checkpoints não estão no repositório | Médio |

## Pontos fortes

### Ambiente e testes

O ambiente passa pelo verificador do Gymnasium. Os testes cobrem reset, step, espaços, recompensas, seeds, término, renderização e casos limite. Durante a revisão:

- 91 testes do ambiente, baselines e contadores passaram;
- nove testes das rotinas de análise passaram;
- o Ruff não encontrou erros.

O Windows bloqueou a DLL do PyTorch durante a coleta da suíte completa. Esse bloqueio ocorreu no computador usado para a revisão e não comprova falha no projeto.

### Treinamento

O script configura PPO e DQN com Stable-Baselines3. A equipe propagou a seed para `random`, NumPy, PyTorch, Gymnasium e Stable-Baselines3. Cada configuração usa 200.000 passos, cinco sementes e avaliações a cada 10.000 passos.

As configurações cobrem duas comparações:

- A contra B avalia duas recompensas com PPO;
- A contra C compara PPO e DQN sob a recompensa de produtividade.

### Baselines e análises

As três baselines correspondem ao enunciado do Grupo 4. Os scripts produzem:

- curvas de aprendizado com média e desvio entre seeds;
- comparação entre agentes e baselines;
- quantidade registrada por fila;
- tabela-resumo;
- análise de sucessos e falhas;
- teste de generalização.

A Configuração A obteve o maior reward na avaliação comum, -78,67. Ela superou a prioridade fixa, -137,19, a fila mais longa, -416,31, e a política aleatória, -418,93.

## Problemas encontrados

### 1. A capacidade não limita as execuções padrão

**Arquivos:** `src/environment/triagem_env.py`, linhas 319 a 332 e 367 a 372.

O ambiente incrementa `_used_capacity` quando atende um chamado. No mesmo passo, `_update_metrics()` reduz esse valor em uma unidade. Como o episódio começa com capacidade usada igual a zero e cada ação atende no máximo um chamado, o valor retorna a zero.

Uma execução controlada com seed 42 produziu:

```text
total_capacity = 1  -> reward 26,7; 91 atendimentos; used_capacity = {0}
total_capacity = 10 -> reward 26,7; 91 atendimentos; used_capacity = {0}
```

**Impacto:** a observação contém capacidade, mas o agente não enfrenta uma restrição de recurso. O professor pode questionar esse ponto porque o tema cita capacidade de atendimento.

**Correção adequada:** atribuir duração aos atendimentos e liberar cada vaga após o término do serviço.

**Custo da correção:** alto. A mudança altera transições, observações visitadas e recompensas. Todos os modelos precisam de novo treino.

### 2. Encaminhamento aumenta `total_served`

**Arquivo:** `src/environment/triagem_env.py`, linhas 335 a 341.

Ao encaminhar, o ambiente remove um chamado, incrementa `_total_served` e incrementa `_served_by_queue`. A recompensa aplica o custo de encaminhamento e não concede o bônus de atendimento.

O cálculo de sucesso usa:

```text
success_rate = total_served / total_arrivals
```

A métrica agrega atendimento local e encaminhamento. O nome “taxa de sucesso” sugere resolução, mas o numerador mede saídas provocadas pelo agente.

**Impacto:** as comparações permanecem consistentes porque todos os métodos usam a mesma regra. A interpretação da métrica exige cuidado.

**Correção adequada:** criar `total_referred` e reservar `total_resolved` para atendimento local.

**Custo da correção:** baixo se a equipe adicionar apenas contadores. Os modelos não precisam de novo treino. Os CSVs e tabelas precisam de nova avaliação para publicar as métricas separadas.

### 3. O agente escolhe entre as próprias heurísticas das baselines

**Arquivo:** `src/environment/triagem_env.py`, método `_process_action()`.

As ações 0 e 1 executam prioridade fixa e fila mais longa. O agente não escolhe uma fila de atendimento por índice. Ele escolhe qual heurística controla o atendimento naquele passo.

Essa modelagem define uma política de alto nível e pode ser defendida dessa forma. Ela limita as estratégias que PPO e DQN conseguem descobrir. A Configuração B convergiu para a prioridade fixa, e a quantidade por fila coincide com essa baseline.

**Correção adequada:** oferecer ações `atender_fila_0`, `atender_fila_1` e `atender_fila_2`.

**Custo da correção:** alto. O novo espaço de ações exige novo treino e invalida os checkpoints atuais.

### 4. O espaço não oferece encaminhamento da fila 2

O ambiente usa `Discrete(4)`. As ações 2 e 3 encaminham as filas 0 e 1. A fila 2, que possui prioridade alta, não recebe ação de encaminhamento.

O grupo pode defender a escolha se casos críticos não puderem ser transferidos. O código e as especificações não registram essa justificativa.

**Correção adequada:** documentar a restrição do domínio ou adicionar a quinta ação.

**Custo da correção:** documentar não afeta os modelos. Adicionar a ação muda a saída das redes e exige novo treino.

### 5. `avg_wait_times` não calcula a média individual

**Arquivo:** `src/environment/triagem_env.py`, linhas 367 a 371.

O ambiente incrementa um contador enquanto a fila permanece ocupada e o zera quando ela esvazia. O valor representa a idade contínua da fila. Ele não acompanha a idade de cada chamado.

Chamados novos podem herdar um contador alto quando entram em uma fila antiga. A recompensa usa esse valor para calcular atraso.

**Correção adequada:** guardar a idade de cada chamado ou manter soma de idades e quantidade.

**Custo da correção:** alto. A mudança altera estado e recompensa, portanto exige novo treino.

### 6. As próprias `.specs` pedem métricas que o código não salva

O arquivo `.specs/05-experimental-protocol.md` pede, por episódio:

- total de encaminhamentos;
- ocorrência de overload;
- reward e passos;
- chamados resolvidos.

O `info` não expõe total de encaminhamentos nem um campo de overload. `eval.py` agrega os episódios e grava uma linha por configuração e seed.

O protocolo mínimo do professor não obriga esses campos. A diferença existe entre o código e a especificação criada pelo grupo.

### 7. Instruções de execução contêm inconsistências

O README apresenta três problemas:

- declara Python 3.11+, enquanto `pyproject.toml` exige Python 3.12+;
- manda instalar `requirements.txt`, mas esse arquivo não existe;
- marca a etapa de análise como planejada, embora o repositório contenha as análises.

O `pyproject.toml` atende ao requisito de arquivo equivalente a `requirements.txt`. A equipe deve orientar o uso de `uv sync`.

### 8. O repositório não inclui checkpoints

O `.gitignore` exclui `model.zip`, `best_model.zip` e `evaluations.npz`. O professor aceita instruções de reprodução no lugar dos modelos, mas a demonstração exige uma execução de agente treinado.

O grupo precisa confirmar que pelo menos um integrante possui os checkpoints usados na avaliação. Um clone limpo não consegue executar `src.agents.eval` sem treinar antes.

### 9. A comparação não demonstra significância estatística

Os gráficos usam um desvio padrão entre cinco seeds. O projeto não aplica teste pareado, intervalo de confiança ou bootstrap.

A Configuração B supera a prioridade fixa por 0,91 ponto, diferença menor que os desvios entre seeds. O grupo não deve afirmar que B superou essa baseline com significância. A conclusão segura é que B reproduziu a política de prioridade fixa.

## Opções para o grupo

### Opção A: preservar o código experimental

O grupo mantém ambiente, modelos e resultados. O relatório e a apresentação explicam as limitações.

**Vantagens:** preserva os 15 treinos, os gráficos e a análise. Reduz o risco antes da apresentação.

**Desvantagens:** o professor pode descontar pontos na modelagem da capacidade e questionar a taxa de sucesso.

### Opção B: aplicar ajustes sem mudar o MDP (escolhida)

O grupo corrige a documentação e prepara a demonstração. Também adiciona contadores ao `info` sem alterar transições ou recompensa.

Mudanças possíveis:

- corrigir versão do Python e instalação no README;
- atualizar o status das análises;
- documentar que encaminhamento conta como saída;
- documentar que a política escolhe heurísticas;
- criar um comando curto de demonstração com `render_mode="human"`;
- adicionar `total_resolved`, `total_referred` e `terminated_by_overload` ao `info`.

Os ajustes de documentação não exigem nova avaliação. Os novos contadores receberam testes. O grupo só precisa gerar novos CSVs se quiser publicar taxas separadas nesta entrega.

### Opção C: corrigir a modelagem

O grupo implementa duração de serviço, espera por chamado e ações diretas por fila.

Essa opção exige:

1. alterar o ambiente e os testes;
2. treinar A, B e C em cinco sementes;
3. avaliar os 15 modelos;
4. gerar baselines, gráficos, tabelas e análise qualitativa;
5. atualizar relatório e slides.

O prazo torna essa opção arriscada. Resultados antigos não podem representar o ambiente corrigido.

## Matriz de impacto

| Mudança | Altera o MDP | Exige treino | Exige novos resultados | Recomendação para agora |
|---|---:|---:|---:|---|
| Corrigir README | Não | Não | Não | Fazer |
| Atualizar status do projeto | Não | Não | Não | Fazer |
| Preparar script de demonstração | Não | Não | Não | Fazer se houver checkpoint |
| Documentar encaminhamento como saída | Não | Não | Não | Fazer |
| Adicionar contador de encaminhamento | Não | Não | Sim, para publicar a métrica | Opcional |
| Adicionar indicador de overload | Não | Não | Sim, para publicar a métrica | Opcional |
| Corrigir capacidade | Sim | Sim | Sim | Adiar |
| Calcular espera individual | Sim | Sim | Sim | Adiar |
| Criar ações diretas por fila | Sim | Sim | Sim | Adiar |
| Adicionar encaminhamento da fila 2 | Sim | Sim | Sim | Adiar ou justificar |

## Recomendação

O grupo deve escolher a Opção B com escopo curto:

1. preservar o ambiente, os modelos e os números apresentados;
2. corrigir o README;
3. garantir um checkpoint local para a demonstração;
4. testar o comando da baseline e do agente no computador da apresentação;
5. explicar capacidade, encaminhamento e espera como limitações;
6. evitar afirmações de significância estatística.

Não recomendamos alterar capacidade, recompensa, observação ou espaço de ações antes da apresentação. Essas mudanças invalidariam a evidência experimental que sustenta o trabalho.

## Respostas para a arguição

### Por que a capacidade não muda o resultado?

O ambiente atual trata cada atendimento como uma operação concluída dentro do passo. Ele aloca e libera uma unidade na mesma transição. A equipe identificou essa simplificação e propõe duração de serviço como melhoria. A correção exige novo treino, por isso os experimentos mantiveram a versão avaliada.

### A taxa de sucesso mede resolução?

Ela mede chamados retirados das filas por atendimento ou encaminhamento. O nome do contador não separa as duas operações. Uma versão futura publicará resolução local e encaminhamento em métricas distintas.

### Por que o agente escolhe prioridade ou fila longa?

A equipe modelou o agente como um seletor de políticas operacionais. Ele decide qual regra aplica a cada estado. Essa abordagem reduz o espaço de ações, mas limita a descoberta de estratégias por fila.

### O agente superou as baselines?

A Configuração A superou as três baselines em reward médio. A Configuração C também obteve reward maior. A Configuração B apresentou resultado próximo da prioridade fixa, sem evidência de diferença estatística.

### O agente generalizou?

Na seed surpresa 999, A perdeu 7,0% e C perdeu 13,0%, classificadas como generalização moderada pelo critério do projeto. B perdeu 16,0% e recebeu classificação de degradação severa. Uma seed surpresa testa uma coorte e não demonstra generalização para todas as distribuições.

## Checklist antes da apresentação

- [ ] Confirmar onde estão os arquivos `model.zip`.
- [ ] Executar uma baseline do início ao fim.
- [ ] Executar um agente treinado do início ao fim.
- [ ] Conferir renderização no computador da apresentação.
- [ ] Abrir os gráficos sem depender de internet.
- [ ] Corrigir as instruções do README.
- [ ] Definir quem explicará a recompensa.
- [ ] Definir quem explicará as limitações.
- [ ] Evitar chamar encaminhamento de resolução local.
- [ ] Evitar afirmar significância estatística.

## Conclusão

O código atende o núcleo da atividade e sustenta a apresentação dos experimentos. A capacidade sem efeito persistente e a mistura entre atendimento e encaminhamento reduzem a precisão da modelagem. Corrigir esses pontos agora exigiria repetir o protocolo.

O grupo deve preservar o ambiente usado nos resultados, ajustar a documentação e preparar a demonstração com um checkpoint local. Após a apresentação, a equipe pode criar uma segunda versão do ambiente e repetir os experimentos com capacidade persistente, espera por chamado e ações diretas por fila.
