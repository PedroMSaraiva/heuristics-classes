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
   ```
   python baseline_mlr.py
   python genetic_mlr.py
   python compare_results.py results/baseline/baseline_results.csv results/genetic/genetic_results.csv
   python visualize_dataset.py
   ```

## Algoritmo Genético para Seleção de Variáveis

O algoritmo genético (AG) é utilizado para selecionar automaticamente o subconjunto de features mais relevante para a regressão linear múltipla. Cada indivíduo da população é um vetor binário, onde cada bit indica se a feature correspondente é usada (1) ou não (0).

- **Modelagem do Cromossomo:**
  - Exemplo: `[1, 0, 1, 1]` (usa as features 1, 3 e 4)
- **Fitness:**
  - Calculado como: `1/(MSE) + 1/(MAE) + max(0, R²)`
  - Quanto maior o fitness, melhor a solução.
- **Parâmetros principais:**
  - `num_generations`: número de gerações (ex: 30)
  - `sol_per_pop`: tamanho da população (ex: 10)
  - `mutation_percent_genes`: taxa de mutação (ex: 30%)
  - `crossover_type`: tipo de cruzamento (ex: 'single_point')
  - `stop_criteria`: critério de parada (ex: 'reach_10')
- **Biblioteca:** PyGAD
- **Como alterar parâmetros:**
  - No arquivo `genetic_mlr.py`, edite a função `run_genetic_feature_selection`.
  - Exemplo:
    ```python
    ga_instance = pygad.GA(
        num_generations=50,  # Altere aqui
        sol_per_pop=20,      # Altere aqui
        ...
    )
    ```

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