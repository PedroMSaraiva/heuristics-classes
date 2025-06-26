import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from metrics import compute_metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import numpy as np
import pygad
from src.logging_utils import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import scipy.stats as stats
import optuna
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

logger = get_logger('genetic_mlr_simplified')

# --- Configuração Simplificada ---
class GAConfig:
    """Configuração simplificada para o algoritmo genético."""
    
    def __init__(self):
        self.params = {
            "num_generations": 80,
            "sol_per_pop": 40,
            "mutation_probability": [0.05, 0.02],  # [alta, baixa] baseado no número de features
            "crossover_probability": 0.8,
            "max_features": 70,
            "feature_penalty": 0.15
        }
        
    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

# Instância global
ga_config = GAConfig()

# Configurações de arquivo
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'calibration.csv'
IDRC_CSV = BASE_DIR / 'data' / 'csv_new' / 'idrc_validation.csv'
OUTPUT_CSV = BASE_DIR / 'results' / 'genetic_results_simplified.csv'
OUTPUT_PLOTS = BASE_DIR / 'results' / 'genetic_plots_simplified'
OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Carrega e prepara todos os dados necessários."""
    logger.info("Carregando e preparando dados...")
    
    # Carrega dados
    calib_df = pd.read_csv(CALIBRATION_CSV)
    idrc_df = pd.read_csv(IDRC_CSV)
    test_df = pd.read_csv(BASE_DIR / 'data' / 'csv_new' / 'test.csv')
    validation_df = pd.read_csv(BASE_DIR / 'data' / 'csv_new' / 'validation.csv')
    
    # Prepara features
    feature_columns = [col for col in calib_df.columns if col != 'target']
    
    X_train = calib_df[feature_columns]
    y_train = calib_df['target']
    X_test = test_df[feature_columns]
    y_test = test_df['target']
    X_val = validation_df[feature_columns]
    y_val = idrc_df['reference']
    
    # Imputação e normalização
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))
    X_val = scaler.transform(imputer.transform(X_val))
    
    logger.info(f"Dados preparados: X_train={X_train.shape}, X_test={X_test.shape}, X_val={X_val.shape}")
    return X_train, y_train, X_test, y_test, X_val, y_val, feature_columns

def create_fitness_function(X_train, y_train, max_features=70, feature_penalty=0.15):
    """Cria função de fitness otimizada usando closures."""
    
    # Cache para evitar recálculos
    fitness_cache = {}
    correlations = np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])
    
    def fitness_func(ga_instance, solution, solution_idx):
        selected = np.where(solution > 0.5)[0]
        
        # Validações básicas
        if len(selected) == 0:
            return 0.001
        if len(selected) > max_features:
            return 0.001
        
        # Cache lookup
        solution_key = tuple(solution)
        if solution_key in fitness_cache:
            return fitness_cache[solution_key]
        
        try:
            # Cross-validation no conjunto de treino
            X_selected = X_train[:, selected]
            model = LinearRegression()
            
            # Métricas de CV
            r2_scores = cross_val_score(model, X_selected, y_train, cv=3, scoring='r2', n_jobs=1)
            mse_scores = -cross_val_score(model, X_selected, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
            
            mean_r2 = r2_scores.mean()
            mean_mse = mse_scores.mean()
            
            # Fitness combinado com bonificação para parcimônia
            if mean_r2 < 0:
                fitness_score = 0.01
            else:
                r2_component = max(0, mean_r2)
                mse_component = 1.0 / (mean_mse + 1e-6)
                fitness_score = 0.6 * r2_component + 0.4 * mse_component
                
                # Bonificação para soluções parcimoniosas e eficazes
                if len(selected) <= 30 and r2_component > 0.5:
                    fitness_score *= 1.2
                elif len(selected) <= 20 and r2_component > 0.3:
                    fitness_score *= 1.5
            
            # Penalização suave por número de features
            penalty = 1.0 - (len(selected) / max_features) * feature_penalty * 0.5
            final_fitness = fitness_score * penalty
            
            # Cache com limite
            if len(fitness_cache) < 1000:
                fitness_cache[solution_key] = final_fitness
            
            return final_fitness
            
        except Exception as e:
            logger.warning(f"Erro no fitness: {e}")
            return 0.001
    
    return fitness_func, correlations

def create_smart_initial_population(num_features, pop_size, correlations, max_features):
    """Cria população inicial inteligente baseada em correlações."""
    population = []
    probs = correlations / correlations.sum()
    
    # Distribuição de features: 1/3 alta, 1/3 média, 1/3 baixa densidade
    for i in range(pop_size):
        solution = np.zeros(num_features, dtype=int)
        
        if i < pop_size // 3:
            n_features = np.random.randint(max_features//2, max_features + 1)
        elif i < 2 * pop_size // 3:
            n_features = np.random.randint(max_features//4, max_features//2 + 1)
        else:
            n_features = np.random.randint(5, max_features//4 + 1)
        
        n_features = min(n_features, num_features)
        
        if n_features > 0:
            selected_indices = np.random.choice(num_features, size=n_features, p=probs, replace=False)
            solution[selected_indices] = 1
        
        population.append(solution)
    
    return np.array(population)

def adaptive_mutation_func(offspring, ga_instance):
    """Mutação adaptativa que usa funcionalidades nativas do PyGAD."""
    max_features = ga_config.get_params()["max_features"]
    
    for i in range(offspring.shape[0]):
        current_features = np.sum(offspring[i])
        
        # Taxa de mutação adaptativa
        if current_features <= 30:
            mutation_rate = 0.02  # Conservador para boas soluções
        elif current_features <= 50:
            mutation_rate = 0.05  # Moderado
        else:
            mutation_rate = 0.15  # Agressivo para simplificar
        
        # Aplica mutação bit-flip nativa do PyGAD
        for j in range(offspring.shape[1]):
            if np.random.random() < mutation_rate:
                offspring[i, j] = 1 - offspring[i, j]
        
        # Controle de restrições simples e eficaz
        current_features = np.sum(offspring[i])
        if current_features > max_features:
            # Remove features aleatoriamente
            selected_indices = np.where(offspring[i] == 1)[0]
            excess = current_features - max_features
            to_remove = np.random.choice(selected_indices, size=excess, replace=False)
            offspring[i, to_remove] = 0
        elif current_features == 0:
            # Adiciona uma feature aleatória
            offspring[i, np.random.randint(0, offspring.shape[1])] = 1
    
    return offspring

def run_simplified_genetic_algorithm(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Executa o algoritmo genético simplificado."""
    logger.info("🧬 Iniciando algoritmo genético simplificado")
    
    params = ga_config.get_params()
    num_features = X_train.shape[1]
    
    # Cria função de fitness e população inicial
    fitness_func, correlations = create_fitness_function(
        X_train, y_train, params["max_features"], params["feature_penalty"]
    )
    
    initial_population = create_smart_initial_population(
        num_features, params["sol_per_pop"], correlations, params["max_features"]
    )
    
    # Callback para restart automático (simplificado)
    def on_generation(ga_instance):
        generation = ga_instance.generations_completed
        
        # Restart simples quando estagnar
        if generation >= 10 and generation % 15 == 0:
            recent_fitness = ga_instance.best_solutions_fitness[-5:]
            if len(set(recent_fitness)) == 1:  # Estagnação
                logger.info(f"[Gen {generation}] 🔄 Restart: diversificando população")
                
                # Mantém top 20%, regenera o resto
                current_pop = ga_instance.population
                keep_count = len(current_pop) // 5
                
                # Regenera com mais diversidade
                new_pop = create_smart_initial_population(
                    num_features, len(current_pop) - keep_count, correlations, params["max_features"]
                )
                
                # Combina elite + nova população
                ga_instance.population = np.vstack([current_pop[:keep_count], new_pop])
    
    # ✅ CONFIGURAÇÃO PYGAD SIMPLIFICADA
    ga_instance = pygad.GA(
        num_generations=params["num_generations"],
        num_parents_mating=params["sol_per_pop"] // 2,
        fitness_func=fitness_func,
        sol_per_pop=params["sol_per_pop"],
        num_genes=num_features,
        initial_population=initial_population,
        gene_type=int,
        gene_space=[0, 1],
        
        # Seleção e crossover nativos do PyGAD
        parent_selection_type="tournament",
        K_tournament=4,
        crossover_type="two_points",
        crossover_probability=params["crossover_probability"],
        
        # Mutação customizada
        mutation_type=adaptive_mutation_func,
        
        # Elitismo nativo
        keep_elitism=4,
        
        # Callback para restart
        on_generation=on_generation,
        
        # Critérios de parada otimizados
        stop_criteria=["reach_30", "saturate_50"]
    )
    
    logger.info("🚀 Executando algoritmo genético...")
    ga_instance.run()
    
    # Resultado
    solution, solution_fitness, _ = ga_instance.best_solution()
    selected_features = np.where(solution > 0.5)[0]
    selected_names = [feature_cols[i] for i in selected_features]
    
    logger.info(f"✅ Algoritmo concluído:")
    logger.info(f"  Fitness: {solution_fitness:.6f}")
    logger.info(f"  Features selecionadas: {len(selected_features)}")
    logger.info(f"  Nomes: {selected_names}")
    
    return selected_features, ga_instance, selected_names

def evaluate_and_save_results(X_train, y_train, X_test, y_test, X_val, y_val, 
                            selected_features, feature_cols, ga_instance):
    """Avalia modelo final e salva resultados."""
    logger.info("📊 Avaliando modelo final...")
    
    # Treina modelo final
    model = LinearRegression()
    model.fit(X_train[:, selected_features], y_train)
    
    # Predições
    y_pred_test = model.predict(X_test[:, selected_features])
    y_pred_val = model.predict(X_val[:, selected_features])
    
    # Métricas
    metrics_test = compute_metrics(y_test, y_pred_test)
    metrics_val = compute_metrics(y_val, y_pred_val)
    
    # Salva resultados
    results = pd.DataFrame([
        {'Conjunto': 'Validação', **metrics_val},
        {'Conjunto': 'Teste', **metrics_test},
    ])
    results.to_csv(OUTPUT_CSV, index=False)
    
    # Gráficos essenciais
    plot_evolution(ga_instance)
    plot_feature_importance(model, selected_features, feature_cols)
    
    logger.info(f"✅ Resultados salvos em {OUTPUT_CSV}")
    logger.info(f"📈 Gráficos salvos em {OUTPUT_PLOTS}")
    
    return metrics_test, metrics_val

def plot_evolution(ga_instance):
    """Plota evolução do algoritmo genético."""
    plt.figure(figsize=(10, 6))
    plt.plot(ga_instance.best_solutions_fitness, 'b-', linewidth=2, label='Melhor Fitness')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Algoritmo Genético (Simplificado)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "genetic_evolution_simplified.png", dpi=300)
    plt.close()

def plot_feature_importance(model, selected_features, feature_names):
    """Plota importância das features."""
    importance = np.abs(model.coef_)
    importance = 100 * importance / importance.sum()
    
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in selected_features],
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importância (%)')
    plt.title(f'Importância das Features (Simplificado)\nTotal: {len(selected_features)} features')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "feature_importance_simplified.png", dpi=300)
    plt.close()

def main():
    """Função principal simplificada."""
    logger.info("🚀 Iniciando pipeline de Seleção Genética de Features (Simplificado)")
    
    # Carrega e prepara dados
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols = load_and_prepare_data()
    
    # Executa algoritmo genético
    selected_features, ga_instance, selected_names = run_simplified_genetic_algorithm(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    )
    
    # Avalia e salva resultados
    metrics_test, metrics_val = evaluate_and_save_results(
        X_train, y_train, X_test, y_test, X_val, y_val, 
        selected_features, feature_cols, ga_instance
    )
    
    # Relatório final
    logger.info("\n📊 RELATÓRIO FINAL (SIMPLIFICADO):")
    logger.info(f"R² Validação: {metrics_val['R2']:.4f}")
    logger.info(f"R² Teste: {metrics_test['R2']:.4f}")
    logger.info(f"Features selecionadas: {len(selected_features)}")
    logger.info(f"Features: {selected_names}")
    
    return {
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'selected_features': selected_names,
        'num_features': len(selected_names)
    }

if __name__ == "__main__":
    results = main() 