# Modelagem de Problemas de Otimização

## Introdução

Este documento apresenta a modelagem matemática detalhada de três problemas de otimização, com foco na definição clara de variáveis, restrições e funções objetivo.

## 1. Problema de Roteamento de Veículos (VRP)

### Descrição
Uma empresa de logística precisa otimizar a entrega de pacotes em várias localidades. O problema envolve determinar rotas eficientes para uma frota de veículos, considerando:
- Múltiplos pontos de entrega
- Capacidade limitada dos veículos
- Prazos de entrega
- Minimização de custos e tempo

### Representação do Problema
O problema pode ser representado como um grafo G = (V, E), onde:
- V: conjunto de vértices (pontos de entrega e depósito)
- E: conjunto de arestas (rotas possíveis)
- Cada vértice v ∈ V tem uma demanda dv e janela de tempo [av, bv]
- Cada aresta (i,j) ∈ E tem um custo cij e tempo de viagem tij

### Variáveis do Problema
1. **Variáveis de Decisão**:
   - `xijk ∈ {0,1}`: 1 se o veículo k usa a aresta (i,j), 0 caso contrário
   - `yik ∈ {0,1}`: 1 se o ponto i é atendido pelo veículo k, 0 caso contrário
   - `sik ≥ 0`: Horário de início do serviço no ponto i pelo veículo k
   - `lik ≥ 0`: Carga do veículo k após visitar o ponto i

2. **Parâmetros**:
   - `N`: Conjunto de pontos de entrega
   - `K`: Conjunto de veículos disponíveis
   - `Q`: Capacidade de cada veículo
   - `di`: Demanda no ponto i
   - `[ai, bi]`: Janela de tempo para entrega no ponto i
   - `cij`: Custo de viagem entre pontos i e j
   - `tij`: Tempo de viagem entre pontos i e j

### Restrições
1. **Atendimento da Demanda**:
   ```
   ∑k∈K yik = 1, ∀i∈N  # Cada ponto deve ser atendido exatamente uma vez
   ```

2. **Capacidade dos Veículos**:
   ```
   ∑i∈N di × yik ≤ Q, ∀k∈K  # Não exceder capacidade do veículo
   ```

3. **Conservação de Fluxo**:
   ```
   ∑j∈V xijk = yik, ∀i∈N, ∀k∈K  # Entrada = Saída para cada ponto
   ∑i∈V xijk = yjk, ∀j∈N, ∀k∈K
   ```

4. **Janelas de Tempo**:
   ```
   ai ≤ sik ≤ bi, ∀i∈N, ∀k∈K  # Respeitar horários de entrega
   sik + tij - M(1-xijk) ≤ sjk, ∀i,j∈V, ∀k∈K  # Sequenciamento temporal
   ```

### Funções Objetivo
1. **Minimizar Custo Total**:
   ```
   min ∑k∈K ∑i∈V ∑j∈V cij × xijk
   ```

2. **Minimizar Tempo Total**:
   ```
   min ∑k∈K ∑i∈N (sik - s0k)  # s0k é o tempo de saída do depósito
   ```

## 2. Problema de Planejamento de Produção

### Descrição
Uma fábrica de eletrônicos precisa planejar sua produção considerando:
- Múltiplos produtos
- Capacidade limitada de máquinas e trabalhadores
- Demanda variável ao longo do tempo
- Custos de produção, estoque e mão de obra

### Representação do Problema
O problema pode ser modelado como um problema de programação linear multi-período:
- Horizonte de planejamento dividido em períodos
- Recursos compartilhados entre produtos
- Balanço entre produção e demanda
- Restrições de capacidade e armazenamento

### Variáveis do Problema
1. **Variáveis de Decisão**:
   - `xit ≥ 0`: Quantidade produzida do produto i no período t
   - `sit ≥ 0`: Estoque do produto i no fim do período t
   - `mit ≥ 0`: Horas-máquina alocadas ao produto i no período t
   - `wit ≥ 0`: Número de trabalhadores alocados ao produto i no período t

2. **Parâmetros**:
   - `I`: Conjunto de produtos
   - `T`: Conjunto de períodos
   - `pi`: Tempo de produção unitário do produto i
   - `ci`: Custo unitário de produção do produto i
   - `hi`: Custo unitário de estoque do produto i
   - `dit`: Demanda do produto i no período t
   - `M`: Número total de máquinas
   - `W`: Número total de trabalhadores
   - `H`: Horas disponíveis por período
   - `S`: Capacidade máxima de armazenamento

### Restrições
1. **Balanço de Estoque**:
   ```
   sit = si(t-1) + xit - dit, ∀i∈I, ∀t∈T
   ```

2. **Capacidade de Produção**:
   ```
   ∑i∈I pi × xit ≤ H × min(M, wit), ∀t∈T
   ```

3. **Limitação de Recursos**:
   ```
   ∑i∈I wit ≤ W, ∀t∈T
   ∑i∈I sit ≤ S, ∀t∈T
   ```

### Funções Objetivo
1. **Maximizar Lucro**:
   ```
   max ∑t∈T ∑i∈I [(pi - ci)xit - hi × sit]
   ```

2. **Minimizar Custos**:
   ```
   min ∑t∈T ∑i∈I [ci × xit + hi × sit + wi × wit]
   ```

## 3. Problema de Alocação de Recursos em Projetos

### Descrição
Uma empresa de construção precisa alocar recursos entre múltiplos projetos, considerando:
- Diferentes tipos de trabalhadores e equipamentos
- Prazos específicos para cada projeto
- Restrições orçamentárias
- Requisitos mínimos de recursos por projeto

### Representação do Problema
O problema pode ser modelado como um problema de programação linear multi-projeto:
- Projetos executados em paralelo
- Recursos compartilhados entre projetos
- Precedência entre atividades
- Progresso medido em percentual de conclusão

### Variáveis do Problema
1. **Variáveis de Decisão**:
   - `xijt ≥ 0`: Número de trabalhadores tipo i no projeto j no período t
   - `yijt ≥ 0`: Número de equipamentos tipo i no projeto j no período t
   - `zjt ∈ [0,1]`: Percentual de conclusão do projeto j no período t
   - `pjt ∈ {0,1}`: 1 se o projeto j está ativo no período t

2. **Parâmetros**:
   - `I`: Tipos de recursos (trabalhadores/equipamentos)
   - `J`: Conjunto de projetos
   - `T`: Períodos de planejamento
   - `Wi`: Número total de trabalhadores tipo i
   - `Ei`: Número total de equipamentos tipo i
   - `Bj`: Orçamento do projeto j
   - `Dj`: Prazo do projeto j
   - `rij`: Requisito mínimo do recurso i no projeto j
   - `cit`: Custo do recurso i no período t

### Restrições
1. **Disponibilidade de Recursos**:
   ```
   ∑j∈J xijt ≤ Wi, ∀i∈I, ∀t∈T
   ∑j∈J yijt ≤ Ei, ∀i∈I, ∀t∈T
   ```

2. **Requisitos Mínimos**:
   ```
   xijt ≥ rij × pjt, ∀i∈I, ∀j∈J, ∀t∈T
   yijt ≥ rij × pjt, ∀i∈I, ∀j∈J, ∀t∈T
   ```

3. **Orçamento e Prazo**:
   ```
   ∑t∈T ∑i∈I (cit × xijt + cit × yijt) ≤ Bj, ∀j∈J
   zjDj = 1, ∀j∈J  # Projeto concluído no prazo
   ```

4. **Progresso do Projeto**:
   ```
   zjt ≥ zj(t-1), ∀j∈J, ∀t∈T  # Progresso não decresce
   zjt ≤ pjt, ∀j∈J, ∀t∈T  # Progresso só ocorre se projeto ativo
   ```

### Funções Objetivo
1. **Minimizar Custo Total**:
   ```
   min ∑t∈T ∑j∈J ∑i∈I (cit × xijt + cit × yijt)
   ```

2. **Maximizar Velocidade de Conclusão**:
   ```
   max ∑j∈J ∑t∈T (zjt - zj(t-1))
   ```

3. **Minimizar Variação de Alocação**:
   ```
   min ∑t∈T ∑j∈J ∑i∈I (|xijt - xij(t-1)| + |yijt - yij(t-1)|)
   ```