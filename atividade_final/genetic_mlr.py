import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from metrics import compute_metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pygad
from src.logging_utils import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import scipy.stats as stats

# --- 0. Configuração do Logger ---
logger = get_logger('genetic_mlr')

# --- 1. Caminhos dos arquivos ---
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

def fitness_func(ga_instance, solution, solution_idx):
    """Função de fitness simplificada para seleção genética de features.

    Args:
        ga_instance: Instância do GA (PyGAD).
        solution: Vetor binário de seleção de features.
        solution_idx: Índice da solução na população.

    Returns:
        float: Valor de fitness.
    """
    selected = np.where(solution > 0.5)[0]
    if len(selected) == 0:
        return 0.0
    
    # Penaliza soluções com muitas features (reduzido para 0.3 para permitir mais features)
    num_features_penalty = 1.0 - (len(selected) / fitness_func.X_train.shape[1]) * 0.3
    
    try:
        # Usa as features selecionadas para treinar e avaliar o modelo
        X = fitness_func.X_train[:, selected]
        y = fitness_func.y_train
        X_val = fitness_func.X_val[:, selected]
        y_val = fitness_func.y_val
        X_test = fitness_func.X_test[:, selected]
        y_test = fitness_func.y_test
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Avalia em validação e teste
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Métricas simplificadas: R² e MSE
        r2_val = r2_score(y_val, y_pred_val)
        r2_test = r2_score(y_test, y_pred_test)
        mse_val = np.mean((y_val - y_pred_val) ** 2)
        mse_test = np.mean((y_test - y_pred_test) ** 2)
        
        # Fitness simplificado: média ponderada de R² e inverso do MSE
        # Peso maior para o conjunto de validação (0.6) vs teste (0.4)
        fitness_val = (r2_val + 1.0 / (mse_val + 1e-8)) / 2
        fitness_test = (r2_test + 1.0 / (mse_test + 1e-8)) / 2
        
        # Combina os fitness com peso maior para validação
        combined_fitness = (0.6 * fitness_val + 0.4 * fitness_test) * num_features_penalty
        
        return combined_fitness
    except Exception as e:
        logger.warning(f"Erro no fitness: {e}")
        return 0.0

def run_genetic_feature_selection(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, num_generations=50, sol_per_pop=20):
    """Executa o algoritmo genético para seleção de features."""
    num_features = X_train.shape[1]
    fitness_func.X_train = X_train
    fitness_func.y_train = y_train
    fitness_func.X_val = X_val
    fitness_func.y_val = y_val
    fitness_func.X_test = X_test
    fitness_func.y_test = y_test

    
    # Calcula importância inicial das features usando correlação com o target
    correlations = np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])
    initial_population = []
    
    # Gera população inicial mais diversificada
    for _ in range(sol_per_pop):
        # Probabilidade de seleção baseada na correlação
        probs = correlations / correlations.sum()
        # Viés para features mais correlacionadas
        solution = np.random.choice([0, 1], size=num_features, p=[0.6, 0.4])
        # Seleciona algumas features mais importantes
        important_features = np.random.choice(num_features, size=min(3, num_features), p=probs, replace=False)
        solution[important_features] = 1
        initial_population.append(solution)
    
    # Parâmetros do GA para logging
    ga_params = {
        "num_generations": 150,
        "num_parents_mating": sol_per_pop // 2,
        "sol_per_pop": 60,
        "mutation_percent_genes": 25,
        "mutation_probability": 0.25,
        "crossover_type": "two_points",
        "K_tournament": 4,
        "keep_parents": 8
    }
    
    
    # Configuração do algoritmo genético
    ga_instance = pygad.GA(
        num_generations=ga_params["num_generations"],
        num_parents_mating=ga_params["num_parents_mating"],
        fitness_func=fitness_func,
        sol_per_pop=ga_params["sol_per_pop"],
        num_genes=num_features,
        gene_type=int,
        init_range_low=0,
        init_range_high=2,
        initial_population=initial_population,
        mutation_percent_genes=ga_params["mutation_percent_genes"],
        mutation_probability=ga_params["mutation_probability"],
        crossover_type=ga_params["crossover_type"],
        gene_space=[0, 1],
        parent_selection_type="tournament",
        K_tournament=ga_params["K_tournament"],
        keep_parents=ga_params["keep_parents"],
        parallel_processing=["thread", 4],
        stop_criteria=["reach_20", "saturate_100"]
    )
    
    logger.info("Iniciando algoritmo genético para seleção de features...")
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

def plot_genetic_evolution(ga_instance, output_dir):
    """Plota a evolução do algoritmo genético.

    Args:
        ga_instance: Instância do GA (PyGAD).
        output_dir: Pasta de saída.
    """
    plt.figure(figsize=(12, 8))
    
    # Plota a evolução do melhor fitness
    plt.plot(ga_instance.best_solutions_fitness, 'b-', label='Melhor Fitness')
    
    # Calcula e plota o fitness médio para cada geração
    mean_fitness = []
    std_fitness = []
    for generation in range(len(ga_instance.best_solutions_fitness)):
        # Obtém os fitness de todos os indivíduos na geração atual
        if hasattr(ga_instance, 'population_fitness') and len(ga_instance.population_fitness) > generation:
            population_fitness = ga_instance.population_fitness[generation]
            mean_fitness.append(np.mean(population_fitness))
            std_fitness.append(np.std(population_fitness))
        else:
            # Se não houver dados de população, usa apenas o melhor fitness
            mean_fitness.append(ga_instance.best_solutions_fitness[generation])
            std_fitness.append(0)
    
    # Plota a média e o desvio padrão
    mean_fitness = np.array(mean_fitness)
    std_fitness = np.array(std_fitness)
    plt.plot(mean_fitness, 'r--', label='Fitness Médio')
    plt.fill_between(range(len(mean_fitness)), 
                    mean_fitness - std_fitness,
                    mean_fitness + std_fitness,
                    alpha=0.2, color='r', label='±1 Desvio Padrão')
    
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolução do Algoritmo Genético', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona estatísticas
    stats_text = f'Melhor Fitness: {ga_instance.best_solutions_fitness[-1]:.3f}\n'
    stats_text += f'Fitness Final Médio: {mean_fitness[-1]:.3f}\n'
    stats_text += f'Desvio Padrão Final: {std_fitness[-1]:.3f}\n'
    stats_text += f'Gerações: {len(ga_instance.best_solutions_fitness)}'
    
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / "genetic_evolution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de evolução genética salvo em {plot_path}")

def plot_real_vs_pred(y_true, y_pred, conjunto, output_dir):
    """Plota gráfico Real vs Previsto com melhorias visuais.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.
        conjunto: Nome do conjunto.
        output_dir: Pasta de saída.
    """
    plt.figure(figsize=(10, 8))
    
    # Calcula métricas para o título
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = r2_score(y_true, y_pred)
    
    # Plota os pontos
    plt.scatter(y_true, y_pred, alpha=0.6, c='blue', label='Dados')
    
    # Linha de referência y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    # Adiciona linha de regressão
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "g--", alpha=0.8, label=f'Tendência (R² = {r2:.3f})')
    
    plt.xlabel('Valor Real', fontsize=12)
    plt.ylabel('Valor Previsto', fontsize=12)
    plt.title(f'Real vs Previsto (Genético) - {conjunto}\nMSE: {mse:.3f}, R²: {r2:.3f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona texto com métricas
    plt.text(0.05, 0.95, f'MSE: {mse:.3f}\nR²: {r2:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / f"real_vs_pred_{conjunto.lower()}_genetic.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico Real vs Previsto salvo em {plot_path}")

def plot_residuals(y_true, y_pred, conjunto, output_dir):
    """Plota resíduos vs valor previsto com melhorias visuais.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.
        conjunto: Nome do conjunto.
        output_dir: Pasta de saída.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 8))
    
    # Plota os resíduos
    plt.scatter(y_pred, residuals, alpha=0.6, c='blue', label='Resíduos')
    
    # Linha de referência y=0
    plt.axhline(y=0, color='r', linestyle='--', label='Resíduo = 0')
    
    # Adiciona banda de confiança
    std_residuals = np.std(residuals)
    plt.axhline(y=2*std_residuals, color='gray', linestyle=':', alpha=0.5, label='±2σ')
    plt.axhline(y=-2*std_residuals, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel('Valor Previsto', fontsize=12)
    plt.ylabel('Resíduo', fontsize=12)
    plt.title(f'Análise de Resíduos (Genético) - {conjunto}\nDesvio Padrão: {std_residuals:.3f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona estatísticas dos resíduos
    stats_text = f'Média: {np.mean(residuals):.3f}\nDesvio: {std_residuals:.3f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / f"residuos_vs_pred_{conjunto.lower()}_genetic.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de resíduos salvo em {plot_path}")

def plot_residuals_hist(y_true, y_pred, conjunto, output_dir):
    """Plota histograma dos resíduos com melhorias visuais.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.
        conjunto: Nome do conjunto.
        output_dir: Pasta de saída.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 8))
    
    # Plota histograma com KDE
    sns.histplot(residuals, bins=30, kde=True, stat='density', color='blue', alpha=0.6)
    
    # Adiciona curva normal para comparação
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, normal, 'r--', label='Distribuição Normal', alpha=0.8)
    
    # Adiciona linhas de referência
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    plt.axvline(mean_res, color='g', linestyle='-', label=f'Média: {mean_res:.3f}')
    plt.axvline(mean_res + 2*std_res, color='r', linestyle=':', alpha=0.5, label='±2σ')
    plt.axvline(mean_res - 2*std_res, color='r', linestyle=':', alpha=0.5)
    
    plt.xlabel('Resíduo', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    plt.title(f'Distribuição dos Resíduos (Genético) - {conjunto}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona estatísticas
    stats_text = f'Média: {mean_res:.3f}\nDesvio: {std_res:.3f}\nSkewness: {stats.skew(residuals):.3f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / f"hist_residuos_{conjunto.lower()}_genetic.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Histograma dos resíduos salvo em {plot_path}")

def main():
    """Executa o pipeline de Regressão Linear Múltipla com Seleção Genética de Features."""
    logger.info("Iniciando pipeline de Regressão Linear Múltipla com Seleção Genética de Features.")
    
    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols, scaler = prepare_datasets(df, idrc_df)
    
    # Executa seleção genética de features
    selected_features, ga_instance = run_genetic_feature_selection(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    
    # Treina e avalia o modelo final
    model, metrics_test, metrics_val, y_pred_test, y_pred_val, selected_names = train_and_evaluate(
        X_train, y_train, X_test, y_test, X_val, y_val, selected_features, feature_cols, scaler)
    
    # Salva resultados
    save_results_to_csv(metrics_val, metrics_test, OUTPUT_CSV, selected_names)
    
    # Gera visualizações
    plot_genetic_evolution(ga_instance, OUTPUT_PLOTS)
    plot_feature_importance(model, selected_features, feature_cols, OUTPUT_PLOTS)
    plot_real_vs_pred(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals_hist(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_real_vs_pred(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals_hist(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    
    logger.info("Pipeline de Regressão Linear Múltipla Genética concluído.")

if __name__ == "__main__":
    main() 