import sys
from pathlib import Path
sys.path.append('src')

import numpy as np
from genetic_mlr import (
    load_data, prepare_datasets, run_genetic_feature_selection, 
    hyperparams, plot_genetic_evolution
)
import matplotlib.pyplot as plt

def test_improved_genetic():
    """Testa o algoritmo genético melhorado com parâmetros reduzidos para velocidade."""
    print("🧬 TESTE DO ALGORITMO GENÉTICO MELHORADO")
    
    # Carrega dados
    CALIBRATION_CSV = Path('data/csv_new/calibration.csv')
    IDRC_CSV = Path('data/csv_new/idrc_validation.csv')
    OUTPUT_PLOTS = Path('results/test_plots')
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    
    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols, scaler = prepare_datasets(df, idrc_df)
    
    print(f"Dados carregados: {X_train.shape}")
    
    # Parâmetros de teste rápido
    test_params = {
        "num_generations": 30,     # Reduzido para teste rápido
        "sol_per_pop": 20,         # População pequena
        "K_tournament": 4,
        "keep_parents": 4,         # Elitismo moderado
        "cv_folds": 3,             # CV rápido
        "max_features": 60,        # Limite menor
        "feature_penalty": 0.1,    # Penalização suave
        "is_optimization": False   # Modo normal
    }
    
    print(f"Parâmetros do teste: {test_params}")
    
    # Executa algoritmo genético
    print("\n🚀 Executando algoritmo genético...")
    selected_features, ga_instance = run_genetic_feature_selection(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, test_params
    )
    
    # Análise dos resultados
    print(f"\n📊 RESULTADOS:")
    print(f"Features selecionadas: {len(selected_features)}")
    print(f"Melhor fitness: {ga_instance.best_solutions_fitness[-1]:.6f}")
    print(f"Fitness inicial: {ga_instance.best_solutions_fitness[0]:.6f}")
    print(f"Melhoria total: {ga_instance.best_solutions_fitness[-1] - ga_instance.best_solutions_fitness[0]:.6f}")
    
    # Verifica evolução
    fitness_evolution = ga_instance.best_solutions_fitness
    generations_with_improvement = 0
    for i in range(1, len(fitness_evolution)):
        if fitness_evolution[i] > fitness_evolution[i-1]:
            generations_with_improvement += 1
    
    improvement_rate = generations_with_improvement / (len(fitness_evolution) - 1)
    print(f"Taxa de gerações com melhoria: {improvement_rate:.2%}")
    
    # Gera gráfico de evolução
    plot_genetic_evolution(ga_instance, OUTPUT_PLOTS)
    print(f"Gráfico salvo em: {OUTPUT_PLOTS}/genetic_evolution.png")
    
    # Análise da estagnação
    last_10_fitness = fitness_evolution[-10:]
    stagnation = len(set(last_10_fitness)) == 1
    
    if stagnation:
        print("⚠️  PROBLEMA: Algoritmo ainda está estagnado nas últimas 10 gerações")
        print("Necessário fazer mais ajustes")
    else:
        print("✅ SUCESSO: Algoritmo está evoluindo!")
        print(f"Últimas 10 gerações tiveram {len(set(last_10_fitness))} valores únicos")
    
    # Estatísticas detalhadas
    print(f"\n📈 ANÁLISE DETALHADA:")
    print(f"Fitness mínimo: {min(fitness_evolution):.6f}")
    print(f"Fitness máximo: {max(fitness_evolution):.6f}")
    print(f"Média das últimas 5 gerações: {np.mean(fitness_evolution[-5:]):.6f}")
    print(f"Desvio padrão da evolução: {np.std(fitness_evolution):.6f}")
    
    return selected_features, ga_instance, improvement_rate

if __name__ == "__main__":
    selected_features, ga_instance, improvement_rate = test_improved_genetic()
    
    if improvement_rate > 0.1:  # Pelo menos 10% das gerações melhoraram
        print("\n🎉 TESTE PASSOU! Algoritmo está funcionando.")
    else:
        print("\n❌ TESTE FALHOU. Ainda há problemas de estagnação.") 