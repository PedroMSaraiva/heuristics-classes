# Apresentação — Atividade Final de Heurísticas

## Objetivo

Este projeto visa comparar abordagens de Regressão Linear Múltipla (MLR) para calibração e predição de dados experimentais, utilizando baseline tradicional e seleção de variáveis via Algoritmo Genético.

## Principais Scripts

- `baseline_mlr.py`: Executa o baseline de MLR.
- `genetic_mlr.py`: Executa MLR com seleção genética de features.
- `visualize_dataset.py`: Gera visualizações exploratórias do dataset.
- `compare_results.py`: Compara métricas entre experimentos.

## Estrutura de Pastas

```
atividade_final/
├── data/csv/           # Dados originais
├── results/
│   ├── baseline_results.csv
│   ├── genetic_results.csv
│   ├── comparacoes/    # Gráficos de comparação
│   ├── baseline_plots/ # Plots do baseline
│   ├── genetic_plots/  # Plots do genetic
│   └── exploratory_plots/ # Plots exploratórios
├── src/                # Funções utilitárias
├── README.md           # Documentação detalhada
├── README_APRESENTACAO.md # Esta apresentação
```

## Principais Métricas

- **R²**: Coeficiente de determinação
- **MSE**: Erro quadrático médio
- **MAE**: Erro absoluto médio
- **Bias**: Viés das previsões
- **RMSE**: Raiz do erro quadrático médio
- **SE**: Erro padrão

## Como Navegar

- Resultados numéricos: `results/baseline_results.csv`, `results/genetic_results.csv`
- Plots de cada experimento: `results/baseline_plots/`, `results/genetic_plots/`
- Visualização exploratória: `results/exploratory_plots/`
- Comparação de experimentos: `results/comparacoes/`

## Execução Rápida

```bash
python baseline_mlr.py
python genetic_mlr.py
python visualize_dataset.py
python compare_results.py results/baseline_results.csv results/genetic_results.csv
```

---

Projeto desenvolvido para a disciplina de Heurística e Modelagem Multiobjetivo. 