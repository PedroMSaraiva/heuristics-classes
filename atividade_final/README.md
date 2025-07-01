# Atividade Final — Organização e Avaliação de Modelos

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Em%20Desenvolvimento-yellow)

## Estrutura de Pastas

- `data/csv/`: arquivos de dados originais.
- `results/`: resultados numéricos (csv) de cada experimento.
    - `results/baseline/`: baseline_results.csv
    - `results/genetic/`: genetic_results.csv
    - `results/comparacoes/`: gráficos de comparação entre experimentos
- `plots/`: gráficos gerados, organizados por experimento e tipo.
    - `plots/baseline/`: gráficos do baseline
    - `plots/genetic/`: gráficos do genetic
    - `plots/exploratorio/`: histogramas, matriz de correlação, pairplot, nulos
- `src/`: funções utilitárias centralizadas (métricas, data loading, plots).

## Como Executar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Execute os scripts principais:
   ```bash
   # ======== EXECUÇÕES INDIVIDUAIS ========
   
   # Baseline (execução única)
   python baseline_mlr.py
   
   # Algoritmo genético com parâmetros padrão (execução única)
   python genetic_mlr.py
   
   # Otimização de hiperparâmetros (NOVO!)
   python genetic_mlr.py --optimize --trials 50
   
   # Script auxiliar para otimização (recomendado)
   python optimize_hyperparams.py quick    # 30 trials
   python optimize_hyperparams.py full     # 100 trials
   python optimize_hyperparams.py best     # usar melhores parâmetros salvos
   
   # ======== ANÁLISE ESTATÍSTICA (NOVO!) ========
   
   # Execução completa: 30 iterações + análise comparativa
   python run_statistical_analysis.py
   
   # Executar apenas 30 iterações do Baseline MLR
   python run_statistical_analysis.py --baseline-only
   
   # Executar apenas 30 iterações do Genetic MLR (com melhores parâmetros)
   python run_statistical_analysis.py --genetic-only
   
   # Executar apenas análise dos resultados existentes
   python run_statistical_analysis.py --analysis-only
   
   # Scripts individuais de múltiplas execuções
   python baseline_mlr_multiple_runs.py    # 30 execuções do baseline
   python genetic_mlr_multiple_runs.py     # 30 execuções do genético
   python analyze_multiple_runs.py         # análise com boxplots
   
   # ======== OUTROS ========
   
   # Comparação entre modelos (execução única)
   python compare_results.py results/baseline/baseline_results.csv results/genetic/genetic_results.csv
   
   # Visualização exploratória
   python visualize_dataset.py
   ```

## Algoritmo Genético para Seleção de Variáveis

O algoritmo genético (AG) é utilizado para selecionar automaticamente o subconjunto de features mais relevante para a regressão linear múltipla. Cada indivíduo da população é um vetor binário, onde cada bit indica se a feature correspondente é usada (1) ou não (0).

### 🧬 Implementação Básica

- **Modelagem do Cromossomo:**
  - Exemplo: `[1, 0, 1, 1]` (usa as features 1, 3 e 4)
- **Fitness (Corrigido):**
  - Calculado com cross-validation apenas no conjunto de treino
  - Fórmula: `(R² + 1/MSE) / 2 × penalty_features`
  - **Sem data leakage**: Validação/teste não são usados no fitness
- **Restrições:**
  - Máximo de 90 features selecionadas
  - Seleção por torneio (K=4) no crossover
  - Operadores customizados respeitam restrições

### 🎯 Otimização Inteligente de Hiperparâmetros (NOVO!)

Implementamos otimização automática usando **Optuna** (Bayesian Optimization):

- **Hiperparâmetros otimizados:**
  - `num_generations` (50-300)
  - `sol_per_pop` (20-100) 
  - `K_tournament` (2-8)
  - `keep_parents` (2-20)
  - `cv_folds` (3-10)
  - `max_features` (30-150)
  - `feature_penalty` (0.1-0.5)

- **Função objetivo:** 70% R² + 30% parcimônia
- **Estratégias:**
  - **Quick**: 30 trials (~30 min)
  - **Full**: 100 trials (~2h)
  - **Best**: usa parâmetros salvos

### ⚙️ Como Usar a Otimização

```bash
# Otimização rápida
python optimize_hyperparams.py quick

# Otimização completa  
python optimize_hyperparams.py full

# Usar melhores parâmetros encontrados
python optimize_hyperparams.py best
```

**Arquivos gerados:**
- `results/best_hyperparameters.json`: melhores parâmetros
- `results/optuna_study.pkl`: histórico completo

**Documentação completa:** [HYPERPARAMETER_OPTIMIZATION.md](HYPERPARAMETER_OPTIMIZATION.md)

## 📊 Análise Estatística com Múltiplas Execuções (NOVO!)

Para obter resultados estatisticamente robustos, implementamos scripts que executam **30 iterações** de cada algoritmo e geram análises comparativas detalhadas.

### Scripts de Múltiplas Execuções

1. **`baseline_mlr_multiple_runs.py`**
   - Executa o MLR baseline 30 vezes
   - Salva todas as métricas em formato estruturado
   - Gera estatísticas resumo (média, desvio, quartis)

2. **`genetic_mlr_multiple_runs.py`**  
   - Executa o algoritmo genético 30 vezes usando os **melhores parâmetros** encontrados
   - Inclui análise do número de features selecionadas
   - Salva fitness evolution para cada execução

3. **`analyze_multiple_runs.py`**
   - Cria **boxplots comparativos** entre baseline e genético
   - Gera plots individuais para cada métrica (R², MSE, MAE, etc.)
   - Análise específica da seleção de features (distribuição, correlação com performance)
   - Tabela de estatísticas descritivas completa

4. **`run_statistical_analysis.py`** (Script Principal)
   - Orquestra toda a análise estatística
   - Execução inteligente com estimativa de tempo
   - Verificação de pré-requisitos
   - Opções flexíveis (apenas baseline, apenas genético, apenas análise)

### Como Usar

```bash
# Análise completa (RECOMENDADO)
python run_statistical_analysis.py
# ⏱️ Tempo estimado: 70-140 minutos

# Execuções parciais
python run_statistical_analysis.py --baseline-only  # ~5-10 min
python run_statistical_analysis.py --genetic-only   # ~60-120 min  
python run_statistical_analysis.py --analysis-only  # ~2-5 min
```

### Resultados Gerados

**Estrutura de pastas criada:**
```
results/
├── multiple_runs/          # Dados das 30 execuções
│   ├── baseline_mlr_30_runs_YYYYMMDD_HHMMSS.csv
│   ├── baseline_mlr_30_runs_YYYYMMDD_HHMMSS.json
│   ├── baseline_mlr_summary_stats_YYYYMMDD_HHMMSS.json
│   ├── genetic_mlr_30_runs_YYYYMMDD_HHMMSS.csv
│   ├── genetic_mlr_30_runs_YYYYMMDD_HHMMSS.json
│   └── genetic_mlr_summary_stats_YYYYMMDD_HHMMSS.json
└── analysis_plots/         # Visualizações comparativas
    ├── comparison_boxplots_YYYYMMDD_HHMMSS.png
    ├── r2_analysis_YYYYMMDD_HHMMSS.png
    ├── mse_analysis_YYYYMMDD_HHMMSS.png
    ├── features_analysis_YYYYMMDD_HHMMSS.png
    └── statistical_summary_YYYYMMDD_HHMMSS.csv
```

**Tipos de análise:**
- **Boxplots comparativos**: Distribuição de todas as métricas
- **Violin plots**: Distribuição detalhada das performances  
- **Análise de features**: Número ótimo, correlação com R²
- **Estatísticas descritivas**: Média, desvio, quartis, min/max
- **Teste de significância**: Comparação estatística entre algoritmos

### Vantagens da Análise Estatística

✅ **Base estatística robusta** (30 execuções vs 1 execução)  
✅ **Quantificação da variabilidade** dos algoritmos  
✅ **Comparação objetiva** com intervalos de confiança  
✅ **Identificação de outliers** e comportamentos anômalos  
✅ **Análise de estabilidade** dos algoritmos  
✅ **Visualizações profissionais** para relatórios  

## Métricas Calculadas

- **R²**: Coeficiente de determinação.
- **MSE**: Mean Squared Error.
- **MAE**: Mean Absolute Error.
- **Bias**: Média dos resíduos, conforme fórmula:
  \[
  Bias = \frac{\sum_{i=1}^n (\hat{y}_i - y_i)}{n}
  \]
- **RMSE**: Root Mean Squared Error.
- **SE**: Standard Error, conforme fórmula:
  \[
  SE = \sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2 - \frac{(\sum_{i=1}^n (\hat{y}_i - y_i))^2}{n}}{n-1}}
  \]

## Visualizações

- Histogramas, matriz de correlação, pairplot, contagem de nulos (plots/exploratorio).
- Gráficos de real vs previsto, resíduos, histogramas de resíduos (plots/baseline, plots/genetic).

## Comparação

- Use `compare_results.py` para comparar métricas entre experimentos.
- Gráficos de barras são salvos em `results/comparacoes/`.

## Centralização de Funções

- Todas as métricas estão em `src/metrics.py`.
- Scripts importam `compute_metrics` para garantir consistência.
- Caminhos de resultados e plots são padronizados.

---

## Observações

- Centralize funções para evitar duplicidade e facilitar manutenção.
- Siga a estrutura de pastas para manter o projeto organizado.

# Heuristics Classes — Atividade Final

Esta pasta contém a **Parte 1 da atividade final** do curso de Heurística e Modelagem Multiobjetivo. Nela foi criado um **baseline** de Regressão Linear Múltipla (MLR) para os dados fornecidos, sem qualquer pré‑processamento ou seleção de variáveis.

---

## 1. Objetivo

- Aplicar um modelo de **Multiple Linear Regression** para prever a variável alvo (`target`) a partir das features disponíveis (`wl` e `input`) nos dados.
- Gerar métricas de avaliação básicas (baseline) que servirão de referência para comparações futuras.

## 2. Metodologia

1. **Carregamento dos dados**
   - `all_data_matlab.csv`: contém colunas `wl`, `inputCalibration`, `targetCalibration`, `inputValidation`, `targetTest`, etc.
   - `all_data_IDRC.csv`: lista de valores de referência (ordem de validação).
2. **Preparação dos conjuntos**
   - **Treino (calibração)**: linhas onde `inputCalibration` e `targetCalibration` não são nulos.
   - **Validação**: linhas indicadas pelos IDs do arquivo `all_data_IDRC.csv`. Essa partição usa `inputValidation` vs. referência do IDRC.
   - **Teste**: resto dos dados com `inputTest` e `targetTest`.
3. **Ajuste do modelo**
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```
4. **Cálculo das métricas (baseline)**
   - **MSE** (Mean Squared Error)
   - **MAE** (Mean Absolute Error)
   - **R²** (coeficiente de determinação)
5. **Geração de saída**
   - `baseline_results.csv`: contém as métricas para **validação** e **teste**.

---

## 3. Como executar

```bash
# Instalar dependências (incluindo scikit-learn)
pip install -r requirements.txt

# Entrar na pasta da atividade final
cd atividade_final

# Rodar o script de baseline
python baseline_mlr.py
```

O script irá imprimir no console e salvar em `baseline_results.csv`:

- Valores de MSE, MAE e R² para os conjuntos de **Validação** e **Teste**.

---

## 4. Resultados do baseline

| Conjunto    |    MSE    |    MAE    |     R²     |
|-------------|-----------|-----------|------------|
| **Validação** | 1.09754   | 0.74191   | –0.08798   |
| **Teste**     | 1.35652   | 0.88702   | –0.02204   |

> **Observação**: R² negativo indica que a regressão linear simples está pior que uma previsão da média dos valores. Esse baseline é uma linha de base para comparação com melhorias futuras.

---

## 5. Próximos Passos

1. **Normalização/Escala** das features (`wl` e `input`).
2. **Engenharia de Features** (interações, polinômios, análise de correlação).
3. **Modelos regularizados** (`Ridge`, `Lasso`) para evitar overfitting.
4. **Seleção de Variáveis** (`SelectKBest`, `RFE`).
5. **Reavaliação** das métricas para verificar ganhos em MSE, MAE e R².

---