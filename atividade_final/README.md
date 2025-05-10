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