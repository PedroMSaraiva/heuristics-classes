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
import json
from typing import Dict, Any, Tuple
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

logger = get_logger('genetic_mlr_multiple_runs')

# --- Configuração de Hiperparâmetros ---
class HyperparameterConfig:
    """Classe para gerenciar configuração de hiperparâmetros do algoritmo genético."""
    
    def __init__(self):
        """Inicializa com configuração padrão."""
        self.default_params = {
            "num_generations": 80,
            "sol_per_pop": 40,
            "K_tournament": 4,
            "keep_parents": 6,
            "cv_folds": 4,
            "max_features": 70,
            "feature_penalty": 0.2
        }
        
        self.best_params = None
        self.best_score = None
        
    def load_best_params(self, filepath: Path):
        """Carrega os melhores parâmetros de arquivo JSON."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.best_params = data.get('best_params')
                self.best_score = data.get('best_score')
                logger.info(f"Carregados melhores parâmetros: {self.best_params}")
        else:
            logger.warning(f"Arquivo de parâmetros não encontrado: {filepath}")
            logger.info("Usando parâmetros padrão")
            self.best_params = self.default_params.copy()

# Instância global da configuração
hyperparams = HyperparameterConfig()

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'calibration.csv'
TEST_CSV = BASE_DIR / 'data' / 'csv_new' / 'test.csv'
VALIDATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'validation.csv'
IDRC_CSV = BASE_DIR / 'data' / 'csv_new' / 'idrc_validation.csv'
WL_CSV = BASE_DIR / 'data' / 'csv_new' / 'wl.csv'

# Pasta para resultados das múltiplas execuções
MULTIPLE_RUNS_DIR = BASE_DIR / 'results' / 'multiple_runs'
MULTIPLE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(calib_path: Path, idrc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os dados de calibração e IDRC."""
    logger.debug(f"Carregando dados de calibração de: {calib_path}")
    df = pd.read_csv(calib_path)
    logger.debug(f"Carregando dados IDRC de: {idrc_path}")
    idrc_df = pd.read_csv(idrc_path)
    return df, idrc_df

def prepare_datasets(df: pd.DataFrame, idrc_df: pd.DataFrame):
    """Prepara os conjuntos de treino, teste e validação, com imputação e normalização."""
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

def fitness_func(ga_instance, solution, solution_idx):
    """Função de fitness otimizada usando cross-validation apenas no conjunto de treino."""
    
    selected = np.where(solution > 0.5)[0]
    
    if len(selected) == 0:
        return -1.0
    
    try:
        X_train_selected = fitness_func.X_train[:, selected]
        
        # Cross-validation no conjunto de treino
        cv_folds = getattr(fitness_func, 'cv_folds', 4)
        cv_scores = cross_val_score(
            fitness_func.model, X_train_selected, fitness_func.y_train, 
            cv=cv_folds, scoring='r2', n_jobs=1
        )
        
        r2_score = np.mean(cv_scores)
        
        # Penalização por número de features
        feature_penalty = getattr(fitness_func, 'feature_penalty', 0.2)
        penalty = feature_penalty * (len(selected) / len(solution))
        
        fitness = r2_score - penalty
        
        return max(fitness, -1.0)
        
    except Exception as e:
        logger.debug(f"Erro no fitness: {e}")
        return -1.0

def run_genetic_feature_selection(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, 
                                ga_params: Dict[str, Any] = None):
    """Executa seleção de features usando algoritmo genético."""
    
    if ga_params is None:
        ga_params = hyperparams.best_params or hyperparams.default_params
    
    logger.debug(f"Parâmetros GA: {ga_params}")
    
    # Configurações para a função fitness
    fitness_func.X_train = X_train
    fitness_func.y_train = y_train
    fitness_func.model = LinearRegression()
    fitness_func.cv_folds = ga_params.get('cv_folds', 4)
    fitness_func.feature_penalty = ga_params.get('feature_penalty', 0.2)
    
    # Atualiza max_features na função de mutação
    adaptive_mutation_func.max_features = ga_params.get('max_features', 70)
    
    num_features = X_train.shape[1]
    
    def create_diverse_population(num_features, pop_size, max_features):
        """Cria população inicial diversa."""
        population = []
        
        for i in range(pop_size):
            individual = np.zeros(num_features)
            
            if i < pop_size // 4:
                # 25% com poucas features (10-30)
                num_selected = np.random.randint(10, min(31, max_features + 1))
            elif i < pop_size // 2:
                # 25% com features medianas (30-60)
                num_selected = np.random.randint(30, min(61, max_features + 1))
            else:
                # 50% com features aleatórias dentro do limite
                num_selected = np.random.randint(20, max_features + 1)
            
            selected_features = np.random.choice(num_features, num_selected, replace=False)
            individual[selected_features] = 1
            population.append(individual)
        
        return np.array(population)
    
    # Cria população inicial
    initial_population = create_diverse_population(
        num_features, 
        ga_params.get('sol_per_pop', 40),
        ga_params.get('max_features', 70)
    )
    
    # Configuração do algoritmo genético
    ga_instance = pygad.GA(
        num_generations=ga_params.get('num_generations', 80),
        num_parents_mating=ga_params.get('sol_per_pop', 40) // 2,
        fitness_func=fitness_func,
        sol_per_pop=ga_params.get('sol_per_pop', 40),
        num_genes=num_features,
        gene_type=int,
        gene_space=[0, 1],
        parent_selection_type="tournament",
        K_tournament=ga_params.get('K_tournament', 4),
        keep_parents=ga_params.get('keep_parents', 6),
        crossover_type="single_point",
        mutation_type=adaptive_mutation_func,
        initial_population=initial_population,
        save_best_solutions=False,  # Desabilita para economizar memória
        suppress_warnings=True
    )
    
    # Executa o algoritmo genético
    logger.debug("Executando algoritmo genético...")
    ga_instance.run()
    
    # Obtém a melhor solução
    solution, solution_fitness, _ = ga_instance.best_solution()
    selected_features = np.where(solution > 0.5)[0]
    
    logger.debug(f"Melhor fitness: {solution_fitness:.4f}")
    logger.debug(f"Features selecionadas: {len(selected_features)}")
    
    return selected_features, ga_instance

def train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, selected_features, feature_cols, scaler):
    """Treina modelo final e avalia performance."""
    
    if len(selected_features) == 0:
        raise ValueError("Nenhuma feature foi selecionada!")
    
    # Seleciona features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    
    # Treina modelo final
    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    
    # Predições
    y_pred_test = model.predict(X_test_selected)
    y_pred_val = model.predict(X_val_selected)
    
    # Métricas
    metrics_test = compute_metrics(y_test, y_pred_test)
    metrics_val = compute_metrics(y_val, y_pred_val)
    
    # Nomes das features selecionadas
    selected_names = [feature_cols[i] for i in selected_features]
    
    return model, metrics_test, metrics_val, y_pred_test, y_pred_val, selected_names

def run_single_iteration(iteration: int, best_params: dict) -> dict:
    """Executa uma única iteração do algoritmo genético.
    
    Args:
        iteration (int): Número da iteração atual.
        best_params (dict): Melhores parâmetros para usar.
        
    Returns:
        dict: Resultados da iteração.
    """
    logger.info(f"=== Execução {iteration + 1}/30 ===")
    
    # Carrega e prepara dados
    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols, scaler = prepare_datasets(df, idrc_df)
    
    # Executa seleção genética de features
    selected_features, ga_instance = run_genetic_feature_selection(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, best_params
    )
    
    # Treina e avalia modelo final
    model, metrics_test, metrics_val, y_pred_test, y_pred_val, selected_names = train_and_evaluate(
        X_train, y_train, X_test, y_test, X_val, y_val, selected_features, feature_cols, scaler
    )
    
    # Prepara resultado da iteração
    result = {
        'iteration': iteration + 1,
        'timestamp': datetime.now().isoformat(),
        'validation': metrics_val,
        'test': metrics_test,
        'selected_features': selected_names,
        'num_features': len(selected_names),
        'best_fitness': float(ga_instance.best_solutions_fitness[-1]),
        'hyperparameters': best_params.copy()
    }
    
    logger.info(f"Iteração {iteration + 1} - Val R²: {metrics_val['R2']:.4f}, Test R²: {metrics_test['R2']:.4f}, Features: {len(selected_names)}")
    
    return result

def save_multiple_runs_results(results: list):
    """Salva os resultados de múltiplas execuções em diferentes formatos.
    
    Args:
        results (list): Lista com resultados de todas as iterações.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Salva resultados completos em JSON
    json_path = MULTIPLE_RUNS_DIR / f'genetic_mlr_30_runs_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Resultados completos salvos em: {json_path}")
    
    # 2. Cria DataFrame com métricas para análise estatística
    metrics_data = []
    for result in results:
        # Métricas de validação
        val_row = {
            'iteration': result['iteration'],
            'dataset': 'validation',
            'R2': result['validation']['R2'],
            'MSE': result['validation']['MSE'],
            'RMSE': result['validation']['RMSE'],
            'MAE': result['validation']['MAE'],
            'BIAS': result['validation']['Bias'],
            'SE': result['validation']['SE'],
            'num_features': result['num_features'],
            'best_fitness': result['best_fitness']
        }
        metrics_data.append(val_row)
        
        # Métricas de teste
        test_row = {
            'iteration': result['iteration'],
            'dataset': 'test',
            'R2': result['test']['R2'],
            'MSE': result['test']['MSE'],
            'RMSE': result['test']['RMSE'],
            'MAE': result['test']['MAE'],
            'BIAS': result['test']['Bias'],
            'SE': result['test']['SE'],
            'num_features': result['num_features'],
            'best_fitness': result['best_fitness']
        }
        metrics_data.append(test_row)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 3. Salva CSV para análise estatística
    csv_path = MULTIPLE_RUNS_DIR / f'genetic_mlr_30_runs_{timestamp}.csv'
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Métricas para análise estatística salvas em: {csv_path}")
    
    # 4. Calcula e salva estatísticas resumo
    summary_stats = {}
    
    for dataset in ['validation', 'test']:
        dataset_data = metrics_df[metrics_df['dataset'] == dataset]
        summary_stats[dataset] = {}
        
        for metric in ['R2', 'MSE', 'RMSE', 'MAE', 'BIAS', 'SE', 'num_features', 'best_fitness']:
            values = dataset_data[metric]
            summary_stats[dataset][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'median': float(values.median()),
                'q75': float(values.quantile(0.75))
            }
    
    # Salva estatísticas resumo
    summary_path = MULTIPLE_RUNS_DIR / f'genetic_mlr_summary_stats_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Estatísticas resumo salvas em: {summary_path}")
    
    # 5. Exibe relatório resumo no log
    logger.info("\n" + "="*60)
    logger.info("RELATÓRIO RESUMO - GENETIC MLR (30 EXECUÇÕES)")
    logger.info("="*60)
    
    for dataset in ['validation', 'test']:
        logger.info(f"\n{dataset.upper()}:")
        stats = summary_stats[dataset]
        logger.info(f"R² - Média: {stats['R2']['mean']:.4f} ± {stats['R2']['std']:.4f}")
        logger.info(f"    Min: {stats['R2']['min']:.4f}, Max: {stats['R2']['max']:.4f}")
        logger.info(f"    Mediana: {stats['R2']['median']:.4f}")
        logger.info(f"MSE - Média: {stats['MSE']['mean']:.4f} ± {stats['MSE']['std']:.4f}")
        logger.info(f"MAE - Média: {stats['MAE']['mean']:.4f} ± {stats['MAE']['std']:.4f}")
        logger.info(f"Features - Média: {stats['num_features']['mean']:.1f} ± {stats['num_features']['std']:.1f}")
    
    logger.info("\n" + "="*60)

def main():
    """Executa 30 iterações do algoritmo genético MLR usando os melhores parâmetros."""
    logger.info("Iniciando 30 execuções do Genetic MLR para análise estatística")
    logger.info(f"Resultados serão salvos em: {MULTIPLE_RUNS_DIR}")
    
    # Carrega os melhores parâmetros
    best_params_file = BASE_DIR / 'results' / 'best_hyperparameters.json'
    hyperparams.load_best_params(best_params_file)
    
    if hyperparams.best_params is None:
        logger.error("Não foi possível carregar os melhores parâmetros!")
        logger.error("Execute primeiro a otimização de hiperparâmetros.")
        return
    
    best_params = hyperparams.best_params
    logger.info(f"Usando melhores parâmetros: {best_params}")
    
    results = []
    
    try:
        for i in range(30):
            result = run_single_iteration(i, best_params)
            results.append(result)
            
            # Log de progresso a cada 5 iterações
            if (i + 1) % 5 == 0:
                logger.info(f"Concluídas {i + 1}/30 execuções")
        
        # Salva todos os resultados
        save_multiple_runs_results(results)
        
        logger.info("Todas as 30 execuções foram concluídas com sucesso!")
        logger.info(f"Resultados disponíveis em: {MULTIPLE_RUNS_DIR}")
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {e}")
        if results:
            logger.info(f"Salvando {len(results)} resultados parciais...")
            save_multiple_runs_results(results)
        raise

if __name__ == "__main__":
    main() 