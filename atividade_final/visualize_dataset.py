import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.logging_utils import get_logger

logger = get_logger('visualize_dataset')

def load_dataset(csv_path):
    """Carrega o dataset principal.

    Args:
        csv_path (Path): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame carregado.
    """
    return pd.read_csv(csv_path)

def plot_histograms(df, output_dir):
    """Gera histogramas para cada coluna numérica.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
    """
    df_numeric = df.select_dtypes(include='number')
    for col in df_numeric.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df_numeric[col].dropna(), bins=30, kde=True)
        plt.title(f'Histograma: {col}')
        plt.tight_layout()
        plt.savefig(output_dir / f'hist_{col}.png')
        plt.close()
        logger.info(f'Histograma {col} salvo com sucesso')

def plot_correlation_matrix(df, output_dir):
    """Gera matriz de correlação (heatmap).

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
    """
    df_numeric = df.select_dtypes(include='number')
    plt.figure(figsize=(8,6))
    corr = df_numeric.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png')
    plt.close()
    logger.info('Matriz de correlação salva com sucesso')

def plot_pairplot(df, output_dir):
    """Gera scatterplot matrix (pairplot) das principais features.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
    """
    main_features = [col for col in ['wl','inputCalibration','inputTest','inputValidation','targetCalibration','targetTest'] if col in df.columns]
    if len(main_features) >= 2:
        sns.pairplot(df[main_features].dropna())
        plt.suptitle('Scatterplot Matrix (Pairplot)', y=1.02)
        plt.savefig(output_dir / 'pairplot_main_features.png')
        plt.close()
        logger.info('Pairplot salvo com sucesso')

def plot_na_counts(df, output_dir):
    """Plota contagem de valores nulos por coluna.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
    """
    na_counts = df.isna().sum()
    plt.figure(figsize=(8,4))
    na_counts.plot(kind='bar')
    plt.title('Contagem de Valores Nulos por Coluna')
    plt.ylabel('Nulos')
    plt.tight_layout()
    plt.savefig(output_dir / 'na_counts.png')
    plt.close()
    logger.info('Contagem de valores nulos salva com sucesso')

def main():
    """Executa a análise exploratória do dataset, gerando visualizações e salvando-as."""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_CSV = BASE_DIR / 'data' / 'csv' / 'all_data_matlab.csv'
    OUTPUT_PLOTS = BASE_DIR / 'results' / 'exploratory_plots'
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_CSV)
    plot_histograms(df, OUTPUT_PLOTS)
    plot_correlation_matrix(df, OUTPUT_PLOTS)
    plot_pairplot(df, OUTPUT_PLOTS)
    plot_na_counts(df, OUTPUT_PLOTS)
    logger.info('Visualizações salvas em ' + str(OUTPUT_PLOTS))

if __name__ == "__main__":
    main() 