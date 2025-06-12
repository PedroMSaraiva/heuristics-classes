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

# --- 0. Configuração do Logger ---
logger = get_logger('baseline_mlr')

# --- 1. Caminhos dos arquivos ---
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'calibration.csv'
TEST_CSV = BASE_DIR / 'data' / 'csv_new' / 'test.csv'
VALIDATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'validation.csv'
IDRC_CSV = BASE_DIR / 'data' / 'csv_new' / 'idrc_validation.csv'
WL_CSV = BASE_DIR / 'data' / 'csv_new' / 'wl.csv'
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

def plot_feature_importance(model, feature_names, output_dir):
    """Plota a importância das features baseada nos coeficientes do modelo.

    Args:
        model: Modelo treinado.
        feature_names: Lista de nomes das features.
        output_dir: Pasta de saída.
    """
    # Calcula a importância das features usando os coeficientes absolutos
    importance = np.abs(model.coef_)
    # Normaliza para porcentagem
    importance = 100 * importance / importance.sum()
    
    # Cria DataFrame para ordenação
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    # Plota apenas as top 20 features para melhor visualização
    top_features = importance_df.tail(20)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importância (%)')
    plt.title('Top 20 Features Mais Importantes')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adiciona valores nas barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / "feature_importance_baseline.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de importância das features salvo em {plot_path}")

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
    plt.title(f'Real vs Previsto - {conjunto}\nMSE: {mse:.3f}, R²: {r2:.3f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona texto com métricas
    plt.text(0.05, 0.95, f'MSE: {mse:.3f}\nR²: {r2:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / f"real_vs_pred_{conjunto.lower()}.png"
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
    plt.title(f'Análise de Resíduos - {conjunto}\nDesvio Padrão: {std_residuals:.3f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona estatísticas dos resíduos
    stats_text = f'Média: {np.mean(residuals):.3f}\nDesvio: {std_residuals:.3f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / f"residuos_vs_pred_{conjunto.lower()}.png"
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
    plt.title(f'Distribuição dos Resíduos - {conjunto}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona estatísticas
    stats_text = f'Média: {mean_res:.3f}\nDesvio: {std_res:.3f}\nSkewness: {stats.skew(residuals):.3f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / f"hist_residuos_{conjunto.lower()}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Histograma dos resíduos salvo em {plot_path}")

def main():
    """Executa o pipeline de Regressão Linear Múltipla Baseline."""
    logger.info("Iniciando pipeline de Regressão Linear Múltipla Baseline.")

    df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
    X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val = prepare_datasets(df, idrc_df)
    
    # Obtém os nomes das features
    feature_names = [col for col in df.columns if col != 'target']
    
    model = train_mlr_model(X_train_scaled, y_train)
    
    # Gera predições
    y_pred_val = model.predict(X_val_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calcula métricas
    metrics_val = compute_metrics(y_val, y_pred_val)
    metrics_test = compute_metrics(y_test, y_pred_test)
    
    # Salva resultados
    save_results_to_csv(metrics_val, metrics_test, OUTPUT_CSV)
    
    # Gera visualizações
    plot_feature_importance(model, feature_names, OUTPUT_PLOTS)
    plot_real_vs_pred(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_residuals_hist(y_val, y_pred_val, 'Validação', OUTPUT_PLOTS)
    plot_real_vs_pred(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
    plot_residuals_hist(y_test, y_pred_test, 'Teste', OUTPUT_PLOTS)
        
    logger.info("Pipeline de Regressão Linear Múltipla Baseline concluído.")

if __name__ == "__main__":
    main()