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
import optuna
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = get_logger('genetic_mlr')

# --- Configuração de Hiperparâmetros ---
class HyperparameterConfig:
    """Classe para gerenciar configuração de hiperparâmetros do algoritmo genético."""
    
    def __init__(self):
        """Inicializa com configuração padrão."""
        self.default_params = {
            "num_generations": 80,   # Reduzido para velocidade
            "sol_per_pop": 40,       # Reduzido para velocidade
            "K_tournament": 4,
            "keep_parents": 6,       # Reduzido
            "cv_folds": 4,
            "max_features": 70,      # Mais conservador
            "feature_penalty": 0.2
        }
        
        self.search_space = {
            "num_generations": (30, 150),  # Reduzido para velocidade
            "sol_per_pop": (20, 60),       # Reduzido para velocidade
            "K_tournament": (2, 6),
            "keep_parents": (2, 15),
            "cv_folds": (3, 5),            # Reduzido para velocidade
            "max_features": (30, 120),     # Mais conservador
            "feature_penalty": (0.1, 0.4)
        }
        
        self.best_params = None
        self.best_score = None
        
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna parâmetros padrão."""
        return self.default_params.copy()
    
    def get_search_space(self) -> Dict[str, Tuple]:
        """Retorna espaço de busca para otimização."""
        return self.search_space.copy()
    
    def update_best_params(self, params: Dict[str, Any], score: float):
        """Atualiza os melhores parâmetros encontrados."""
        self.best_params = params.copy()
        self.best_score = score
        
    def save_best_params(self, filepath: Path):
        """Salva os melhores parâmetros em arquivo JSON."""
        if self.best_params:
            with open(filepath, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_score': self.best_score
                }, f, indent=2)
                
    def load_best_params(self, filepath: Path):
        """Carrega os melhores parâmetros de arquivo JSON."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.best_params = data.get('best_params')
                self.best_score = data.get('best_score')

# Instância global da configuração
hyperparams = HyperparameterConfig()

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'calibration.csv'
TEST_CSV = BASE_DIR / 'data' / 'csv_new' / 'test.csv'
VALIDATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'validation.csv'
IDRC_CSV = BASE_DIR / 'data' / 'csv_new' / 'idrc_validation.csv'
WL_CSV = BASE_DIR / 'data' / 'csv_new' / 'wl.csv'
OUTPUT_CSV = BASE_DIR / 'results' / 'genetic_results.csv'
OUTPUT_PLOTS = BASE_DIR / 'results' / 'genetic_plots'
OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

def load_data(calib_path: Path, idrc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os dados de calibração e IDRC.

    Args:
        calib_path (Path): Caminho para o CSV de calibração.
        idrc_path (Path): Caminho para o CSV de validação IDRC.

    Returns:
        tuple: DataFrames de calibração e IDRC.
    """
    logger.info(f"Carregando dados de calibração de: {calib_path}")
    df = pd.read_csv(calib_path)
    logger.info(f"Dados de calibração carregados. Shape: {df.shape}")
    logger.info(f"Carregando dados IDRC de: {idrc_path}")
    idrc_df = pd.read_csv(idrc_path)
    logger.info(f"Dados IDRC carregados. Shape: {idrc_df.shape}")
    return df, idrc_df

def prepare_datasets(df: pd.DataFrame, idrc_df: pd.DataFrame):
    """Prepara os conjuntos de treino, teste e validação, com imputação e normalização.

    Args:
        df (pd.DataFrame): Dados de calibração.
        idrc_df (pd.DataFrame): Dados de validação IDRC.

    Returns:
        tuple: Arrays e listas para treino, teste, validação, nomes das features e scaler.
    """
    # Carrega os dados de teste e validação
    test_df = pd.read_csv(TEST_CSV)
    validation_df = pd.read_csv(VALIDATION_CSV)
    
    # Seleciona apenas as colunas numéricas (excluindo a última coluna que é o target)
    feature_columns = [col for col in df.columns if col != 'target']
    
    # Prepara os dados de treino (usando todos os dados de calibração)
    X_train = df[feature_columns]
    y_train = df['target']
    
    # Prepara os dados de teste (do arquivo test.csv)
    X_test = test_df[feature_columns]
    y_test = test_df['target']
    
    # Prepara os dados de validação (do arquivo validation.csv)
    X_val = validation_df[feature_columns]
    y_val = idrc_df['reference']  # Usando a coluna 'reference' do IDRC
    
    # Imputação de dados faltantes
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_val_imputed = imputer.transform(X_val)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val, feature_columns, scaler

def adaptive_mutation_func(offspring, ga_instance):
    """Mutação adaptativa simplificada com controle de features."""
    MAX_FEATURES = getattr(adaptive_mutation_func, 'max_features', 70)
    
    for i in range(offspring.shape[0]):
        current_features = np.sum(offspring[i])
        
        # Taxa adaptativa baseada no número de features
        mutation_rate = 0.02 if current_features <= 30 else (0.05 if current_features <= 50 else 0.15)
        
        # Aplica mutação bit-flip
        mutations = np.random.random(offspring.shape[1]) < mutation_rate
        offspring[i] = np.where(mutations, 1 - offspring[i], offspring[i])
        
        # Controle de restrições
        current_features = np.sum(offspring[i])
        if current_features > MAX_FEATURES:
            excess = current_features - MAX_FEATURES
            selected_indices = np.where(offspring[i] == 1)[0]
            to_remove = np.random.choice(selected_indices, size=excess, replace=False)
            offspring[i, to_remove] = 0
        elif current_features == 0:
            offspring[i, np.random.randint(0, offspring.shape[1])] = 1
    
    return offspring

# Removemos o crossover customizado para usar o nativo do PyGAD com controle de features

def fitness_func(ga_instance, solution, solution_idx):
    """Função de fitness otimizada usando cross-validation apenas no conjunto de treino.
    
    Otimizações implementadas:
    - Cache de resultados para soluções idênticas
    - Cross-validation reduzido durante otimização
    - Penalização eficiente para restrições

    Args:
        ga_instance: Instância do GA (PyGAD).
        solution: Vetor binário de seleção de features.
        solution_idx: Índice da solução na população.

    Returns:
        float: Valor de fitness.
    """
    
    selected = np.where(solution > 0.5)[0]
    if len(selected) == 0:
        return 0.001
    
    # ✅ RESTRIÇÃO: Máximo de features
    MAX_FEATURES = getattr(fitness_func, 'max_features', 90)
    if len(selected) > MAX_FEATURES:
        return 0.001  # Penalização severa
    
    # ✅ Cache para evitar recomputação de soluções idênticas
    solution_key = tuple(solution)
    if not hasattr(fitness_func, 'cache'):
        fitness_func.cache = {}
    
    if solution_key in fitness_func.cache:
        return fitness_func.cache[solution_key]
    
    # Penalização suave para promover parcimônia
    feature_penalty = getattr(fitness_func, 'feature_penalty', 0.2)
    num_features_penalty = 1.0 - (len(selected) / MAX_FEATURES) * feature_penalty
    
    try:
        # ✅ Usa APENAS dados de treino para evitar data leakage
        X = fitness_func.X_train[:, selected]
        y = fitness_func.y_train
        
        # ✅ Otimização: CV reduzido durante busca, completo apenas no final
        cv_folds = getattr(fitness_func, 'cv_folds', 4)
        is_optimization = getattr(fitness_func, 'is_optimization', False)
        
        if is_optimization and cv_folds > 3:
            # Durante otimização, usa CV reduzido para velocidade
            actual_cv = 3
        else:
            actual_cv = cv_folds
        
        model = LinearRegression()
        
        # ✅ Cross-validation otimizado
        cv_r2_scores = cross_val_score(model, X, y, cv=actual_cv, scoring='r2', n_jobs=1)
        cv_neg_mse_scores = cross_val_score(model, X, y, cv=actual_cv, scoring='neg_mean_squared_error', n_jobs=1)
        
        # Converte MSE negativo para positivo
        cv_mse_scores = -cv_neg_mse_scores
        
        # Calcula médias das métricas de cross-validation
        mean_r2 = cv_r2_scores.mean()
        mean_mse = cv_mse_scores.mean()
        
        # ✅ Fitness com bonificação para soluções parcimoniosas
        if mean_r2 < 0:
            fitness_score = 0.01  # R² negativo = fitness muito baixo
        else:
            # Combina R² e inverso do MSE de forma mais equilibrada
            r2_component = max(0, mean_r2)  # Garante não negativo
            mse_component = 1.0 / (mean_mse + 1e-6)
            
            # Pondera os componentes: 60% R², 40% MSE
            fitness_score = 0.6 * r2_component + 0.4 * mse_component
            
            # 🎯 BONIFICAÇÃO ESPECIAL para soluções com poucas features e bom R²
            if len(selected) <= 30 and r2_component > 0.5:
                fitness_score *= 1.2  # +20% bonus para soluções excelentes e simples
            elif len(selected) <= 20 and r2_component > 0.3:
                fitness_score *= 1.5  # +50% bonus para soluções muito simples e boas
        
        # Aplica penalização por número de features (suavizada)
        # Penalização mais suave para incentivar exploração
        soft_penalty = 1.0 - (len(selected) / MAX_FEATURES) * feature_penalty * 0.5  # Reduz penalização pela metade
        combined_fitness = fitness_score * soft_penalty
        
        # ✅ Salva no cache (limita tamanho do cache)
        if len(fitness_func.cache) < 1000:
            fitness_func.cache[solution_key] = combined_fitness
        
        return combined_fitness
    except Exception as e:
        logger.warning(f"Erro no fitness (sol_idx={solution_idx}): {e}")
        return 0.001

def run_genetic_feature_selection(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, 
                                  ga_params: Dict[str, Any] = None):
    """Executa o algoritmo genético para seleção de features com parâmetros configuráveis."""
    # Usa parâmetros padrão se não fornecidos
    if ga_params is None:
        ga_params = hyperparams.get_default_params()
    
    num_features = X_train.shape[1]
    fitness_func.X_train = X_train
    fitness_func.y_train = y_train
    
    correlations = np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])
    initial_population = []
    
    # ✅ Gera população inicial com viés para mais features (configurável)
    MAX_FEATURES = ga_params.get("max_features", 90)
    sol_per_pop = ga_params.get("sol_per_pop", 60)
    # ✅ Corrige os limites da população inicial com base em MAX_FEATURES
    min_features = max(5, MAX_FEATURES // 10)  # Mínimo 5 ou 10% do máximo
    med_features = max(min_features + 1, MAX_FEATURES // 2)  # 50% do máximo
    high_features = max(med_features + 1, int(MAX_FEATURES * 0.8))  # 80% do máximo
    
    logger.info(f"Faixas de features para população inicial:")
    logger.info(f"  - Baixa: {min_features}-{med_features}")
    logger.info(f"  - Média: {med_features}-{high_features}")
    logger.info(f"  - Alta: {high_features}-{MAX_FEATURES}")
    
    for i in range(sol_per_pop):
        # Probabilidade de seleção baseada na correlação
        probs = correlations / correlations.sum()
        
        if i < sol_per_pop // 3:
            # 1/3 da população: muitas features
            num_features_to_select = np.random.randint(high_features, MAX_FEATURES + 1)
            num_features_to_select = min(num_features_to_select, num_features)  # Não pode exceder total
        elif i < 2 * sol_per_pop // 3:
            # 1/3 da população: features moderadas
            num_features_to_select = np.random.randint(med_features, high_features + 1)
            num_features_to_select = min(num_features_to_select, num_features)
        else:
            # 1/3 da população: poucas features
            num_features_to_select = np.random.randint(min_features, med_features + 1)
            num_features_to_select = min(num_features_to_select, num_features)
        
        solution = np.zeros(num_features, dtype=int)
        if num_features_to_select > 0:
            selected_indices = np.random.choice(num_features, size=num_features_to_select, 
                                              p=probs, replace=False)
            solution[selected_indices] = 1
        
        initial_population.append(solution)
    
    # ✅ Logging das estatísticas da população inicial
    initial_features_counts = [np.sum(solution) for solution in initial_population]
    logger.info(f"Estatísticas da população inicial:")
    logger.info(f"  - Média de features selecionadas: {np.mean(initial_features_counts):.1f}")
    logger.info(f"  - Min/Max features: {np.min(initial_features_counts)}/{np.max(initial_features_counts)}")
    logger.info(f"  - Soluções com > {MAX_FEATURES} features: {sum(1 for count in initial_features_counts if count > MAX_FEATURES)}")
    
    # ✅ Parâmetros configuráveis do GA
    num_parents_mating = max(2, ga_params.get("sol_per_pop", 60) // 2)
    
    logger.info(f"Parâmetros do GA:")
    logger.info(f"  - Gerações: {ga_params.get('num_generations', 150)}")
    logger.info(f"  - População: {ga_params.get('sol_per_pop', 60)}")
    logger.info(f"  - Pais para cruzamento: {num_parents_mating}")
    logger.info(f"  - Torneio K: {ga_params.get('K_tournament', 4)}")
    logger.info(f"  - Elitismo: {ga_params.get('keep_parents', 8)}")
    logger.info(f"  - Máximo features: {MAX_FEATURES}")
    
    # ✅ Configura parâmetros para funções
    adaptive_mutation_func.max_features = MAX_FEATURES
    fitness_func.max_features = MAX_FEATURES
    fitness_func.cv_folds = ga_params.get("cv_folds", 4)
    fitness_func.feature_penalty = ga_params.get("feature_penalty", 0.2)
    fitness_func.is_optimization = ga_params.get("is_optimization", False)
    
    # ✅ Limpa cache se existir
    if hasattr(fitness_func, 'cache'):
        fitness_func.cache.clear()
    
    # ✅ Configuração do algoritmo genético com operadores customizados
    # Critérios de parada otimizados para evolução real
    if ga_params.get("is_optimization", False):
        stop_criteria = ["reach_15", "saturate_20"]  # Parada mais agressiva durante otimização
        parallel_threads = 2  # Menos threads para evitar overhead
    else:
        stop_criteria = ["reach_30", "saturate_50"]  # Permite mais evolução
        parallel_threads = 4
    
    # 🔄 Callback simplificado para restart automático
    def on_generation(ga_instance):
        generation = ga_instance.generations_completed
        if generation >= 10 and generation % 15 == 0:
            recent_fitness = ga_instance.best_solutions_fitness[-5:]
            if len(set(recent_fitness)) == 1:  # Estagnação
                print(f"  [Gen {generation}] 🔄 RESTART: diversificando população...")
                # Regenera 80% da população mantendo 20% da elite
                pop_size = len(ga_instance.population)
                keep_count = pop_size // 5
                new_pop = create_diverse_population(num_features, pop_size - keep_count, correlations, MAX_FEATURES)
                ga_instance.population = np.vstack([ga_instance.population[:keep_count], new_pop])
    
    def create_diverse_population(num_features, pop_size, correlations, max_features):
        """Cria população diversificada baseada em correlações."""
        population = []
        probs = correlations / correlations.sum()
        for i in range(pop_size):
            solution = np.zeros(num_features, dtype=int)
            n_features = np.random.randint(5, max_features//2)
            if np.random.random() < 0.5:
                selected_indices = np.random.choice(num_features, size=n_features, p=probs, replace=False)
            else:
                selected_indices = np.random.choice(num_features, size=n_features, replace=False)
            solution[selected_indices] = 1
            population.append(solution)
        return np.array(population)

    # ✅ Configuração PyGAD simplificada
    ga_instance = pygad.GA(
        num_generations=ga_params.get("num_generations", 80),
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=ga_params.get("sol_per_pop", 40),
        num_genes=num_features,
        initial_population=initial_population,
        gene_type=int,
        gene_space=[0, 1],
        
        # Operadores nativos do PyGAD
        parent_selection_type="tournament",
        K_tournament=ga_params.get("K_tournament", 4),
        crossover_type="two_points",
        crossover_probability=0.8,
        mutation_type=adaptive_mutation_func,
        
        # Elitismo nativo
        keep_elitism=ga_params.get("keep_parents", 6),
        
        # Callback e critérios de parada
        on_generation=on_generation,
        stop_criteria=stop_criteria
    )
    
    logger.info("Iniciando algoritmo genético para seleção de features...")
    logger.info("✅ Usando seleção por torneio (K=4) para escolha de pais no crossover")
    ga_instance.run()
    
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    selected = np.where(solution > 0.5)[0]
    
    # Log das features selecionadas
    selected_features = [feature_cols[i] for i in selected]
    logger.info(f"Melhor solução encontrada:")
    logger.info(f"Fitness: {solution_fitness}")
    logger.info(f"Features selecionadas: {selected_features}")
    logger.info(f"Número de features selecionadas: {len(selected_features)}")
    
    return selected, ga_instance

def train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, selected_features, feature_cols, scaler):
    """Treina o modelo final e avalia nos conjuntos de teste e validação.

    Args:
        X_train, y_train, X_test, y_test, X_val, y_val: Dados e targets.
        selected_features: Índices das features selecionadas.
        feature_cols: Lista de nomes das features.
        scaler: Scaler usado.

    Returns:
        tuple: Modelo treinado, métricas, predições e nomes das features selecionadas.
    """
    selected_names = [feature_cols[i] for i in selected_features]
    logger.info(f"Treinando modelo final com features: {selected_names}")
    model = LinearRegression()
    model.fit(X_train[:, selected_features], y_train)
    y_pred_test = model.predict(X_test[:, selected_features])
    y_pred_val = model.predict(X_val[:, selected_features])
    metrics_test = compute_metrics(y_test, y_pred_test)
    metrics_val = compute_metrics(y_val, y_pred_val)
    return model, metrics_test, metrics_val, y_pred_test, y_pred_val, selected_names

def save_results_to_csv(metrics_val: dict, metrics_test: dict, output_path: Path, selected_names):
    """Salva as métricas em CSV e loga as features selecionadas.

    Args:
        metrics_val: Métricas de validação.
        metrics_test: Métricas de teste.
        output_path: Caminho do CSV de saída.
        selected_names: Lista de features selecionadas.
    """
    results = pd.DataFrame([
        {'Conjunto': 'Validação', **metrics_val},
        {'Conjunto': 'Teste', **metrics_test},
    ])
    logger.info(f"Salvando resultados em: {output_path}")
    results.to_csv(output_path, index=False)
    logger.info("Resultados salvos com sucesso.")
    logger.info(f"Features selecionadas: {selected_names}")
    logger.info('\n=== Genetic MLR Results ===')
    logger.info(f"\n{results.to_string(index=False)}")

def plot_feature_importance(model, selected_features, feature_names, output_dir):
    """Plota a importância das features selecionadas pelo algoritmo genético.

    Args:
        model: Modelo treinado.
        selected_features: Índices das features selecionadas.
        feature_names: Lista de nomes das features.
        output_dir: Pasta de saída.
    """
    # Calcula a importância das features usando os coeficientes absolutos
    importance = np.abs(model.coef_)
    # Normaliza para porcentagem
    importance = 100 * importance / importance.sum()
    
    # Cria DataFrame apenas com as features selecionadas
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in selected_features],
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importância (%)')
    plt.title(f'Importância das Features Selecionadas pelo Algoritmo Genético\nTotal de Features: {len(selected_features)}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adiciona valores nas barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / "feature_importance_genetic.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de importância das features salvo em {plot_path}")

def create_plots(ga_instance, model, selected_features, feature_cols, y_val, y_pred_val, y_test, y_pred_test):
    """Cria todos os gráficos necessários de forma simplificada."""
    
    # 1. Evolução genética
    plt.figure(figsize=(10, 6))
    plt.plot(ga_instance.best_solutions_fitness, 'b-', linewidth=2, label='Melhor Fitness')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Algoritmo Genético')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "genetic_evolution.png", dpi=300)
    plt.close()
    
    # 2. Importância das features
    importance = 100 * np.abs(model.coef_) / np.abs(model.coef_).sum()
    feature_names = [feature_cols[i] for i in selected_features]
    
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(importance)
    plt.barh(range(len(importance)), importance[sorted_idx])
    plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importância (%)')
    plt.title(f'Importância das Features (Total: {len(selected_features)})')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "feature_importance_genetic.png", dpi=300)
    plt.close()
    
    # 3. Real vs Previsto (função simplificada)
    for y_true, y_pred, nome in [(y_val, y_pred_val, 'validação'), (y_test, y_pred_test, 'teste')]:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
        plt.xlabel('Valor Real')
        plt.ylabel('Valor Previsto')
        plt.title(f'Real vs Previsto - {nome.title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOTS / f"real_vs_pred_{nome}_genetic.png", dpi=300)
        plt.close()
        
        # Histograma dos resíduos
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Resíduo')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição dos Resíduos - {nome.title()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOTS / f"hist_residuos_{nome}_genetic.png", dpi=300)
        plt.close()
    
    logger.info(f"Gráficos salvos em {OUTPUT_PLOTS}")

def optimize_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, 
                            n_trials: int = 100, timeout: int = 3600) -> Dict[str, Any]:
    """Otimiza hiperparâmetros do algoritmo genético usando Optuna.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Dados de treino, validação e teste
        feature_cols: Lista de nomes das features
        n_trials: Número de trials do Optuna
        timeout: Timeout em segundos
        
    Returns:
        Dict com melhores parâmetros encontrados
    """
    logger.info(f"🔍 Iniciando otimização de hiperparâmetros com {n_trials} trials")
    
    def objective(trial):
        """Função objetivo para o Optuna."""
        # Define o espaço de busca
        search_space = hyperparams.get_search_space()
        
        # Sugere parâmetros para este trial
        params = {
            "num_generations": trial.suggest_int("num_generations", *search_space["num_generations"]),
            "sol_per_pop": trial.suggest_int("sol_per_pop", *search_space["sol_per_pop"]),
            "K_tournament": trial.suggest_int("K_tournament", *search_space["K_tournament"]),
            "keep_parents": trial.suggest_int("keep_parents", *search_space["keep_parents"]),
            "cv_folds": trial.suggest_int("cv_folds", *search_space["cv_folds"]),
            "max_features": trial.suggest_int("max_features", *search_space["max_features"]),
            "feature_penalty": trial.suggest_float("feature_penalty", *search_space["feature_penalty"]),
            "is_optimization": True  # ✅ Ativa modo otimização rápida
        }
        
        # Atualiza função de fitness com novos parâmetros
        fitness_func.cv_folds = params["cv_folds"]
        fitness_func.feature_penalty = params["feature_penalty"]
        fitness_func.is_optimization = True  # ✅ Ativa modo otimização rápida
        
        try:
            # Executa algoritmo genético com parâmetros atuais
            selected_features, ga_instance = run_genetic_feature_selection(
                X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, params
            )
            
            # Treina modelo final e avalia
            model, metrics_test, metrics_val, _, _, _ = train_and_evaluate(
                X_train, y_train, X_test, y_test, X_val, y_val, 
                selected_features, feature_cols, None
            )
            
            # Função objetivo: maximizar R² de validação e minimizar número de features
            r2_val = metrics_val['R2']
            num_features = len(selected_features)
            
            # Score combinado: 70% R² + 30% parcimônia
            parsimony_score = 1.0 - (num_features / params["max_features"])
            combined_score = 0.7 * r2_val + 0.3 * parsimony_score
            
            # Registro das métricas no trial
            trial.set_user_attr("r2_val", r2_val)
            trial.set_user_attr("r2_test", metrics_test['R2'])
            trial.set_user_attr("num_features", num_features)
            trial.set_user_attr("mse_val", metrics_val['MSE'])
            trial.set_user_attr("mae_val", metrics_val['MAE'])
            
            logger.info(f"Trial {trial.number}: R²={r2_val:.3f}, Features={num_features}, Score={combined_score:.3f}")
            
            return combined_score
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} falhou: {e}")
            return -1.0  # Score baixo para trials que falharam
    
    # Cria estudo Optuna
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Executa otimização
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # Analisa resultados
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"🎯 Otimização concluída!")
    logger.info(f"Melhor score: {best_value:.4f}")
    logger.info(f"Melhores parâmetros: {best_params}")
    
    # Atualiza configuração global
    hyperparams.update_best_params(best_params, best_value)
    
    # Salva resultados
    results_file = BASE_DIR / 'results' / 'best_hyperparameters.json'
    hyperparams.save_best_params(results_file)
    
    # Salva estudo completo
    study_file = BASE_DIR / 'results' / 'optuna_study.pkl'
    import pickle
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    
    logger.info(f"Resultados salvos em: {results_file}")
    
    return best_params

def run_with_optimization(optimize: bool = True, n_trials: int = 50) -> Dict[str, Any]:
    """Executa o pipeline completo com ou sem otimização de hiperparâmetros.
    
    Args:
        optimize: Se True, executa otimização de hiperparâmetros
        n_trials: Número de trials para otimização
        
    Returns:
        Dict com resultados finais
    """
    logger.info("🚀 Iniciando pipeline de Regressão Linear Múltipla com Seleção Genética")
    
    # Carrega dados
    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols, scaler = prepare_datasets(df, idrc_df)
    
    best_params = None
    
    if optimize:
        # Otimiza hiperparâmetros
        best_params = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, n_trials
        )
    else:
        # Usa parâmetros padrão ou carrega melhores parâmetros salvos
        hyperparams.load_best_params(BASE_DIR / 'results' / 'best_hyperparameters.json')
        best_params = hyperparams.best_params or hyperparams.get_default_params()
        logger.info(f"Usando parâmetros: {best_params}")
    
    # Executa algoritmo genético com melhores parâmetros
    logger.info("🧬 Executando algoritmo genético com melhores parâmetros")
    selected_features, ga_instance = run_genetic_feature_selection(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, best_params
    )
    
    # Treina e avalia modelo final
    model, metrics_test, metrics_val, y_pred_test, y_pred_val, selected_names = train_and_evaluate(
        X_train, y_train, X_test, y_test, X_val, y_val, selected_features, feature_cols, scaler
    )
    
    # Salva resultados
    save_results_to_csv(metrics_val, metrics_test, OUTPUT_CSV, selected_names)
    
    # Gera visualizações simplificadas
    create_plots(ga_instance, model, selected_features, feature_cols, y_val, y_pred_val, y_test, y_pred_test)
    
    logger.info("✅ Pipeline concluído com sucesso!")
    
    return {
        'best_params': best_params,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'selected_features': selected_names,
        'num_features': len(selected_names)
    }

def main():
    """Executa o pipeline com opções de otimização de hiperparâmetros."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Regressão Linear Múltipla com Seleção Genética de Features')
    parser.add_argument('--optimize', action='store_true', 
                       help='Executa otimização de hiperparâmetros')
    parser.add_argument('--trials', type=int, default=50,
                       help='Número de trials para otimização (padrão: 50)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout em segundos para otimização (padrão: 3600)')
    
    args = parser.parse_args()
    
    if args.optimize:
        logger.info(f"🔍 Modo de otimização ativado - {args.trials} trials")
        results = run_with_optimization(optimize=True, n_trials=args.trials)
    else:
        logger.info("⚡ Modo padrão - usando parâmetros salvos ou padrão")
        results = run_with_optimization(optimize=False)
    
    # Relatório final
    logger.info("📊 RELATÓRIO FINAL:")
    logger.info(f"R² Validação: {results['metrics_val']['R2']:.4f}")
    logger.info(f"R² Teste: {results['metrics_test']['R2']:.4f}")
    logger.info(f"Features selecionadas: {results['num_features']}")
    logger.info(f"MSE Validação: {results['metrics_val']['MSE']:.4f}")
    logger.info(f"MAE Validação: {results['metrics_val']['MAE']:.4f}")
    
    return results

if __name__ == "__main__":
    main() 