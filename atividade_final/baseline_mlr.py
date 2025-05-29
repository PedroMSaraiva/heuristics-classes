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

# --- 0. Configuração do Logger ---
logger = get_logger('baseline_mlr')

# --- 1. Caminhos dos arquivos ---
BASE_DIR = Path(__file__).resolve().parent
CALIB_CSV = BASE_DIR / 'data' / 'csv' / 'all_data_matlab.csv'
IDRC_CSV  = BASE_DIR / 'data' / 'csv' / 'all_data_IDRC.csv'
OUTPUT_CSV = BASE_DIR / 'results' / 'baseline_results.csv'
OUTPUT_PLOTS = BASE_DIR / 'results' / 'baseline_plots'
OUTPUT_PLOTS.mkdir(exist_ok=True)

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
    mask_calib = df['targetCalibration'].notna()
    X_train = pd.DataFrame({
        'wl':    df.loc[mask_calib, 'wl'],
        'input': df.loc[mask_calib, 'inputCalibration']
    })
    y_train = df.loc[mask_calib, 'targetCalibration']
    mask_test = df['targetTest'].notna()
    X_test = pd.DataFrame({
        'wl':    df.loc[mask_test, 'wl'],
        'input': df.loc[mask_test, 'inputTest']
    })
    y_test = df.loc[mask_test, 'targetTest']
    mask_val = df['inputValidation'].notna()
    X_val = pd.DataFrame({
        'wl':    df.loc[mask_val, 'wl'],
        'input': df.loc[mask_val, 'inputValidation']
    })
    y_val = idrc_df['Value (Reference values)']
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
    logger.info("Treinando o modelo LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Modelo treinado com sucesso.")
    logger.debug(f"Coeficientes do modelo: {model.coef_}")
    logger.debug(f"Intercepto do modelo: {model.intercept_}")
    return model

def save_results_to_csv(metrics_val: dict, metrics_test: dict, output_path: Path):
    """Salva as métricas em CSV.

    Args:
        metrics_val: Métricas de validação.
        metrics_test: Métricas de teste.
        output_path: Caminho do CSV de saída.
    """
    results = pd.DataFrame([
        {'Conjunto': 'Validação', **metrics_val},
        {'Conjunto': 'Teste', **metrics_test},
    ])
    logger.info(f"Salvando resultados em: {output_path}")
    results.to_csv(output_path, index=False)
    logger.info("Resultados salvos com sucesso.")
    logger.info('\n=== Baseline MLR Results ===')
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
    plot_path = output_dir / f"real_vs_pred_{conjunto.lower()}.png"
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
    plot_path = output_dir / f"residuos_vs_pred_{conjunto.lower()}.png"
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
    plot_path = output_dir / f"hist_residuos_{conjunto.lower()}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Histograma dos resíduos salvo em {plot_path}")

def main():
    """Executa o pipeline de Regressão Linear Múltipla Baseline."""
    logger.info("Iniciando pipeline de Regressão Linear Múltipla Baseline.")
    df, idrc_df = load_data(CALIB_CSV, IDRC_CSV)
    X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val = prepare_datasets(df, idrc_df)
    model = train_mlr_model(X_train_scaled, y_train)
    logger.info("Avaliando modelo no conjunto de validação...")
    y_pred_val = model.predict(X_val_scaled)
    metrics_val = compute_metrics(y_val, y_pred_val)
    logger.info(f"Métricas de Validação: {metrics_val}")
    logger.info("Avaliando modelo no conjunto de teste...")
    y_pred_test = model.predict(X_test_scaled)
    metrics_test = compute_metrics(y_test, y_pred_test)
    logger.info(f"Métricas de Teste: {metrics_test}")
    save_results_to_csv(metrics_val, metrics_test, OUTPUT_CSV)
    plot_real_vs_pred(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals_hist(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_real_vs_pred(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals_hist(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    logger.info("Pipeline de Regressão Linear Múltipla Baseline concluído.")

if __name__ == "__main__":
    main()