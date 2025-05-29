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

# --- 0. Configuração do Logger ---
logger = get_logger('genetic_mlr')

# --- 1. Caminhos dos arquivos ---
BASE_DIR = Path(__file__).resolve().parent
CALIB_CSV = BASE_DIR / 'data' / 'csv' / 'all_data_matlab.csv'
IDRC_CSV  = BASE_DIR / 'data' / 'csv' / 'all_data_IDRC.csv'
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
    feature_cols = ['wl', 'inputCalibration', 'inputTest', 'inputValidation']
    mask_calib = df['targetCalibration'].notna()
    X_train = df.loc[mask_calib, feature_cols].copy()
    y_train = df.loc[mask_calib, 'targetCalibration']
    mask_test = df['targetTest'].notna()
    X_test = df.loc[mask_test, feature_cols].copy()
    y_test = df.loc[mask_test, 'targetTest']
    mask_val = df['inputValidation'].notna()
    X_val = df.loc[mask_val, feature_cols].copy()
    y_val = idrc_df['Value (Reference values)']
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_val_imputed = imputer.transform(X_val)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val, feature_cols, scaler

def fitness_func(ga_instance, solution, solution_idx):
    """Função de fitness para seleção genética de features.

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
    X = fitness_func.X_train[:, selected]
    y = fitness_func.y_train
    X_val = fitness_func.X_val[:, selected]
    y_val = fitness_func.y_val
    try:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        fitness = 1 / (metrics['MSE'] + 1e-8) + 1 / (metrics['MAE'] + 1e-8) + max(0, metrics['R2'])
        return fitness
    except Exception as e:
        logger.warning(f"Erro no fitness: {e}")
        return 0.0

def run_genetic_feature_selection(X_train, y_train, X_val, y_val, feature_cols, num_generations=30, sol_per_pop=10):
    """Executa o algoritmo genético para seleção de features.

    Args:
        X_train: Dados de treino.
        y_train: Target de treino.
        X_val: Dados de validação.
        y_val: Target de validação.
        feature_cols: Lista de nomes das features.
        num_generations: Número de gerações.
        sol_per_pop: Soluções por população.

    Returns:
        tuple: Índices das features selecionadas e instância do GA.
    """
    num_features = X_train.shape[1]
    fitness_func.X_train = X_train
    fitness_func.y_train = y_train
    fitness_func.X_val = X_val
    fitness_func.y_val = y_val
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_features,
        gene_type=int,
        init_range_low=0,
        init_range_high=2,
        mutation_percent_genes=30,
        mutation_type="random",
        crossover_type="single_point",
        gene_space=[0, 1],
        stop_criteria=["reach_10"]
    )
    logger.info("Iniciando algoritmo genético para seleção de features...")
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    selected = np.where(solution > 0.5)[0]
    logger.info(f"Melhor solução: {solution}, Fitness: {solution_fitness}, Features selecionadas: {[feature_cols[i] for i in selected]}")
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

def plot_real_vs_pred(y_true, y_pred, conjunto, output_dir):
    """Plota gráfico Real vs Previsto.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.
        conjunto: Nome do conjunto.
        output_dir: Pasta de saída.
    """
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='y = y_pred')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Previsto')
    plt.title(f'Real vs Previsto - {conjunto}')
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"real_vs_pred_{conjunto.lower()}_genetic.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gráfico Real vs Previsto salvo em {plot_path}")

def plot_residuals(y_true, y_pred, conjunto, output_dir):
    """Plota resíduos vs valor previsto.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.
        conjunto: Nome do conjunto.
        output_dir: Pasta de saída.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Valor Previsto')
    plt.ylabel('Resíduo')
    plt.title(f'Resíduos vs Valor Previsto - {conjunto}')
    plt.tight_layout()
    plot_path = output_dir / f"residuos_vs_pred_{conjunto.lower()}_genetic.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gráfico de resíduos salvo em {plot_path}")

def plot_residuals_hist(y_true, y_pred, conjunto, output_dir):
    """Plota histograma dos resíduos.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.
        conjunto: Nome do conjunto.
        output_dir: Pasta de saída.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel('Resíduo')
    plt.title(f'Histograma dos Resíduos - {conjunto}')
    plt.tight_layout()
    plot_path = output_dir / f"hist_residuos_{conjunto.lower()}_genetic.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Histograma dos resíduos salvo em {plot_path}")

def main():
    """Executa o pipeline de Regressão Linear Múltipla com Seleção Genética de Features."""
    logger.info("Iniciando pipeline de Regressão Linear Múltipla com Seleção Genética de Features.")
    df, idrc_df = load_data(CALIB_CSV, IDRC_CSV)
    X_train, y_train, X_test, y_test, X_val, y_val, feature_cols, scaler = prepare_datasets(df, idrc_df)
    selected_features, ga_instance = run_genetic_feature_selection(X_train, y_train, X_val, y_val, feature_cols)
    model, metrics_test, metrics_val, y_pred_test, y_pred_val, selected_names = train_and_evaluate(
        X_train, y_train, X_test, y_test, X_val, y_val, selected_features, feature_cols, scaler)
    save_results_to_csv(metrics_val, metrics_test, OUTPUT_CSV, selected_names)
    plot_real_vs_pred(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals_hist(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_real_vs_pred(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals_hist(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    logger.info("Pipeline de Regressão Linear Múltipla Genética concluído.")

if __name__ == "__main__":
    main() 