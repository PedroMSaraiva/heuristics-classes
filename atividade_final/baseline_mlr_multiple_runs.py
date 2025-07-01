import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from metrics import compute_metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.logging_utils import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
import json
from datetime import datetime

# --- 0. Configuração do Logger ---
logger = get_logger('baseline_mlr_multiple_runs')

# --- 1. Caminhos dos arquivos ---
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

def prepare_datasets(df: pd.DataFrame, idrc_df: pd.DataFrame) -> tuple:
    """Prepara os conjuntos de treino, teste e validação, com normalização.

    Args:
        df (pd.DataFrame): Dados de calibração.
        idrc_df (pd.DataFrame): Dados de validação IDRC.

    Returns:
        tuple: Arrays e listas para treino, teste, validação e scaler.
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
    y_val = idrc_df['reference']
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val

def train_mlr_model(X_train, y_train) -> LinearRegression:
    """Treina o modelo de Regressão Linear Múltipla.

    Args:
        X_train: Dados de treino.
        y_train: Target de treino.

    Returns:
        LinearRegression: Modelo treinado.
    """
    logger.debug("Treinando o modelo LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.debug("Modelo treinado com sucesso.")
    return model

def run_single_iteration(iteration: int) -> dict:
    """Executa uma única iteração do algoritmo baseline MLR com bootstrap sampling.
    
    Args:
        iteration (int): Número da iteração atual.
        
    Returns:
        dict: Resultados da iteração.
    """
    logger.info(f"=== Execução {iteration + 1}/30 ===")
    
    # Define seed para reprodutibilidade dentro da iteração
    np.random.seed(iteration + 42)  # +42 para evitar seed=0
    
    # Carrega e prepara dados
    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val = prepare_datasets(df, idrc_df)
    
    # Bootstrap sampling nos dados de treino
    n_samples = len(X_train_scaled)
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    X_train_bootstrap = X_train_scaled[bootstrap_indices]
    y_train_bootstrap = y_train.iloc[bootstrap_indices].reset_index(drop=True)
    
    # Treina modelo com dados bootstrap
    model = train_mlr_model(X_train_bootstrap, y_train_bootstrap)
    
    # Gera predições
    y_pred_val = model.predict(X_val_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calcula métricas
    metrics_val = compute_metrics(y_val, y_pred_val)
    metrics_test = compute_metrics(y_test, y_pred_test)
    
    # Prepara resultado da iteração
    result = {
        'iteration': iteration + 1,
        'timestamp': datetime.now().isoformat(),
        'validation': metrics_val,
        'test': metrics_test,
        'model_coefficients': model.coef_.tolist(),
        'model_intercept': float(model.intercept_),
        'bootstrap_indices': bootstrap_indices.tolist()  # Para reprodutibilidade
    }
    
    logger.info(f"Iteração {iteration + 1} - Val R²: {metrics_val['R2']:.4f}, Test R²: {metrics_test['R2']:.4f}")
    
    return result

def save_multiple_runs_results(results: list):
    """Salva os resultados de múltiplas execuções em diferentes formatos.
    
    Args:
        results (list): Lista com resultados de todas as iterações.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Salva resultados completos em JSON
    json_path = MULTIPLE_RUNS_DIR / f'baseline_mlr_30_runs_{timestamp}.json'
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
            'SE': result['validation']['SE']
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
            'SE': result['test']['SE']
        }
        metrics_data.append(test_row)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 3. Salva CSV para análise estatística
    csv_path = MULTIPLE_RUNS_DIR / f'baseline_mlr_30_runs_{timestamp}.csv'
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Métricas para análise estatística salvas em: {csv_path}")
    
    # 4. Calcula e salva estatísticas resumo
    summary_stats = {}
    
    for dataset in ['validation', 'test']:
        dataset_data = metrics_df[metrics_df['dataset'] == dataset]
        summary_stats[dataset] = {}
        
        for metric in ['R2', 'MSE', 'RMSE', 'MAE', 'BIAS', 'SE']:
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
    summary_path = MULTIPLE_RUNS_DIR / f'baseline_mlr_summary_stats_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Estatísticas resumo salvas em: {summary_path}")
    
    # 5. Exibe relatório resumo no log
    logger.info("\n" + "="*60)
    logger.info("RELATÓRIO RESUMO - BASELINE MLR (30 EXECUÇÕES)")
    logger.info("="*60)
    
    for dataset in ['validation', 'test']:
        logger.info(f"\n{dataset.upper()}:")
        stats = summary_stats[dataset]
        logger.info(f"R² - Média: {stats['R2']['mean']:.4f} ± {stats['R2']['std']:.4f}")
        logger.info(f"    Min: {stats['R2']['min']:.4f}, Max: {stats['R2']['max']:.4f}")
        logger.info(f"    Mediana: {stats['R2']['median']:.4f}")
        logger.info(f"MSE - Média: {stats['MSE']['mean']:.4f} ± {stats['MSE']['std']:.4f}")
        logger.info(f"MAE - Média: {stats['MAE']['mean']:.4f} ± {stats['MAE']['std']:.4f}")
    
    logger.info("\n" + "="*60)

def main():
    """Executa 30 iterações do algoritmo baseline MLR."""
    logger.info("Iniciando 30 execuções do Baseline MLR para análise estatística")
    logger.info(f"Resultados serão salvos em: {MULTIPLE_RUNS_DIR}")
    
    results = []
    
    try:
        for i in range(30):
            result = run_single_iteration(i)
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