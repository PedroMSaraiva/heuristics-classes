import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('src')
from genetic_mlr import load_data, prepare_datasets, hyperparams, fitness_func, custom_mutation_func, custom_crossover_func
import matplotlib.pyplot as plt

def diagnose_genetic_algorithm():
    """Diagnóstica problemas no algoritmo genético."""
    print('=== DIAGNÓSTICO DO ALGORITMO GENÉTICO ===')
    
    # Carrega dados
    CALIBRATION_CSV = Path('data/csv_new/calibration.csv')
    IDRC_CSV = Path('data/csv_new/idrc_validation.csv')
    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols, scaler = prepare_datasets(df, idrc_df)
    
    # Configura função de fitness
    fitness_func.X_train = X_train
    fitness_func.y_train = y_train
    fitness_func.max_features = 70
    fitness_func.cv_folds = 3
    fitness_func.feature_penalty = 0.2
    fitness_func.is_optimization = False
    
    print(f'Dados: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}')
    
    # =========== TESTE 1: VARIABILIDADE DO FITNESS ===========
    print('\n1. TESTE DE VARIABILIDADE DO FITNESS:')
    solutions = []
    fitness_values = []
    
    # Gera 20 soluções aleatórias diferentes
    np.random.seed(42)
    for i in range(20):
        solution = np.zeros(X_train.shape[1], dtype=int)
        n_features = np.random.randint(10, 70)
        selected_indices = np.random.choice(X_train.shape[1], size=n_features, replace=False)
        solution[selected_indices] = 1
        
        fitness = fitness_func(None, solution, i)
        solutions.append(solution)
        fitness_values.append(fitness)
        print(f'  Solução {i:2d}: {n_features:2d} features, fitness = {fitness:.6f}')
    
    print(f'\nEstatísticas do fitness:')
    print(f'  Média: {np.mean(fitness_values):.6f}')
    print(f'  Variância: {np.var(fitness_values):.8f}')
    print(f'  Desvio padrão: {np.std(fitness_values):.8f}')
    print(f'  Range: {np.max(fitness_values) - np.min(fitness_values):.8f}')
    print(f'  Min: {np.min(fitness_values):.6f}, Max: {np.max(fitness_values):.6f}')
    
    # =========== TESTE 2: CACHE DA FUNÇÃO DE FITNESS ===========
    print('\n2. TESTE DO CACHE:')
    # Testa mesma solução duas vezes
    test_solution = solutions[0]
    fitness1 = fitness_func(None, test_solution, 0)
    fitness2 = fitness_func(None, test_solution, 0)
    print(f'  Primeira chamada: {fitness1:.6f}')
    print(f'  Segunda chamada (cache): {fitness2:.6f}')
    print(f'  Cache funcionando: {fitness1 == fitness2}')
    
    # =========== TESTE 3: OPERADORES GENÉTICOS ===========
    print('\n3. TESTE DOS OPERADORES GENÉTICOS:')
    
    # População inicial pequena para teste
    population = np.array(solutions[:6])
    print(f'População original:')
    original_fitness = []
    for i, sol in enumerate(population):
        fit = fitness_func(None, sol, i)
        original_fitness.append(fit)
        print(f'  Ind {i}: {np.sum(sol):2d} features, fitness = {fit:.6f}')
    
    # Teste de mutação
    print(f'\nTeste de Mutação:')
    custom_mutation_func.max_features = 70
    mutated = custom_mutation_func(population.copy(), None)
    mutated_fitness = []
    mutations_count = 0
    for i, sol in enumerate(mutated):
        fit = fitness_func(None, sol, i)
        mutated_fitness.append(fit)
        changed = not np.array_equal(sol, population[i])
        if changed:
            mutations_count += 1
        print(f'  Ind {i}: {np.sum(sol):2d} features, fitness = {fit:.6f} (mudou: {changed})')
    
    print(f'  Taxa de mutação efetiva: {mutations_count}/{len(population)} = {mutations_count/len(population):.2%}')
    print(f'  Melhoria média no fitness: {np.mean(mutated_fitness) - np.mean(original_fitness):.6f}')
    
    # Teste de crossover
    print(f'\nTeste de Crossover:')
    custom_crossover_func.max_features = 70
    offspring_size = (4, X_train.shape[1])  # 4 filhos
    children = custom_crossover_func(population, offspring_size, None)
    children_fitness = []
    for i, child in enumerate(children):
        fit = fitness_func(None, child, i)
        children_fitness.append(fit)
        print(f'  Filho {i}: {np.sum(child):2d} features, fitness = {fit:.6f}')
    
    print(f'  Fitness médio dos pais: {np.mean(original_fitness):.6f}')
    print(f'  Fitness médio dos filhos: {np.mean(children_fitness):.6f}')
    print(f'  Melhoria no crossover: {np.mean(children_fitness) - np.mean(original_fitness):.6f}')
    
    # =========== TESTE 4: DISTRIBUIÇÃO INICIAL ===========
    print('\n4. TESTE DA POPULAÇÃO INICIAL:')
    
    # Simula geração de população inicial
    correlations = np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])
    initial_population = []
    MAX_FEATURES = 70
    sol_per_pop = 40
    
    min_features = max(5, MAX_FEATURES // 10)
    med_features = max(min_features + 1, MAX_FEATURES // 2)
    high_features = max(med_features + 1, int(MAX_FEATURES * 0.8))
    
    for i in range(sol_per_pop):
        probs = correlations / correlations.sum()
        
        if i < sol_per_pop // 3:
            num_features_to_select = np.random.randint(high_features, MAX_FEATURES + 1)
        elif i < 2 * sol_per_pop // 3:
            num_features_to_select = np.random.randint(med_features, high_features + 1)
        else:
            num_features_to_select = np.random.randint(min_features, med_features + 1)
        
        num_features_to_select = min(num_features_to_select, X_train.shape[1])
        
        solution = np.zeros(X_train.shape[1], dtype=int)
        if num_features_to_select > 0:
            selected_indices = np.random.choice(X_train.shape[1], size=num_features_to_select, 
                                              p=probs, replace=False)
            solution[selected_indices] = 1
        
        initial_population.append(solution)
    
    # Estatísticas da população inicial
    initial_features_counts = [np.sum(solution) for solution in initial_population]
    initial_fitness = [fitness_func(None, solution, i) for i, solution in enumerate(initial_population)]
    
    print(f'  Estatísticas da população inicial:')
    print(f'    Média de features: {np.mean(initial_features_counts):.1f}')
    print(f'    Min/Max features: {np.min(initial_features_counts)}/{np.max(initial_features_counts)}')
    print(f'    Desvio padrão features: {np.std(initial_features_counts):.1f}')
    print(f'    Fitness médio inicial: {np.mean(initial_fitness):.6f}')
    print(f'    Desvio padrão fitness: {np.std(initial_fitness):.6f}')
    print(f'    Range fitness: {np.max(initial_fitness) - np.min(initial_fitness):.6f}')
    
    # =========== TESTE 5: PROBLEMAS IDENTIFICADOS ===========
    print('\n5. POSSÍVEIS PROBLEMAS IDENTIFICADOS:')
    
    issues = []
    
    # Problema 1: Variabilidade baixa do fitness
    if np.std(fitness_values) < 0.01:
        issues.append("⚠️  PROBLEMA: Variabilidade do fitness muito baixa!")
        issues.append("   Solução: Aumentar diferenciação entre soluções")
    
    # Problema 2: Cache muito agressivo
    if hasattr(fitness_func, 'cache') and len(fitness_func.cache) > 500:
        issues.append("⚠️  PROBLEMA: Cache muito grande pode estar impedindo exploração")
        issues.append("   Solução: Limitar ou limpar cache periodicamente")
    
    # Problema 3: Mutação muito baixa
    if mutations_count / len(population) < 0.5:
        issues.append("⚠️  PROBLEMA: Taxa de mutação efetiva muito baixa")
        issues.append("   Solução: Aumentar probabilidade de mutação ou força da mutação")
    
    # Problema 4: Crossover não melhora fitness
    if np.mean(children_fitness) <= np.mean(original_fitness):
        issues.append("⚠️  PROBLEMA: Crossover não está gerando melhoria")
        issues.append("   Solução: Melhorar seleção de pais ou operador de crossover")
    
    # Problema 5: Fitness inicial muito homogêneo
    if np.std(initial_fitness) < 0.005:
        issues.append("⚠️  PROBLEMA: População inicial muito homogênea")
        issues.append("   Solução: Aumentar diversidade na população inicial")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ Nenhum problema óbvio detectado nos operadores genéticos")
    
    # =========== RECOMENDAÇÕES ===========
    print('\n6. RECOMENDAÇÕES PARA MELHORAR A EVOLUÇÃO:')
    print("• Aumentar taxa de mutação de 0.1 para 0.2-0.3")
    print("• Implementar mutação adaptativa baseada na diversidade")
    print("• Usar crossover uniforme ao invés de dois pontos")
    print("• Adicionar operador de busca local ocasional")
    print("• Implementar restart quando fitness estagnar")
    print("• Reduzir penalização por número de features")
    print("• Aumentar pressão seletiva no torneio (K maior)")

if __name__ == "__main__":
    diagnose_genetic_algorithm() 