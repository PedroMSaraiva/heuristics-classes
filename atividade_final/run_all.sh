#!/bin/bash
# Executa toda a pipeline do projeto de heurísticas

set -e

# 1. Visualização exploratória do dataset
echo "[1/4] Gerando visualizações exploratórias..."
python visualize_dataset.py

# 2. Baseline de Regressão Linear Múltipla
echo "[2/4] Executando baseline MLR..."
python baseline_mlr.py

# 3. Regressão Linear Múltipla com seleção genética de features
echo "[3/4] Executando genetic MLR..."
python genetic_mlr.py

# 4. Comparação dos resultados (baseline vs genetic)
echo "[4/4] Comparando resultados..."
python compare_results.py results/baseline_results.csv results/genetic_results.csv

echo "Pipeline completa!" 