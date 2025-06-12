import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.logging_utils import get_logger
import numpy as np

logger = get_logger('visualize_dataset')

def load_dataset(calib_path, idrc_path):
    """Carrega os datasets de calibração e IDRC.

    Args:
        calib_path (Path): Caminho para o arquivo CSV de calibração.
        idrc_path (Path): Caminho para o arquivo CSV de validação IDRC.

    Returns:
        tuple: (DataFrame de calibração, DataFrame de IDRC)
    """
    logger.info(f"Carregando dados de calibração de: {calib_path}")
    df_calib = pd.read_csv(calib_path)
    logger.info(f"Dados de calibração carregados. Shape: {df_calib.shape}")
    
    logger.info(f"Carregando dados IDRC de: {idrc_path}")
    df_idrc = pd.read_csv(idrc_path)
    logger.info(f"Dados IDRC carregados. Shape: {df_idrc.shape}")
    
    return df_calib, df_idrc

def plot_histograms(df, output_dir, prefix=''):
    """Gera histogramas para cada coluna numérica.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
        prefix (str): Prefixo para o nome do arquivo.
    """
    df_numeric = df.select_dtypes(include='number')
    
    # Plota histogramas em uma grade para melhor visualização
    n_cols = 4
    n_rows = (len(df_numeric.columns) + n_cols - 1) // n_cols
    
    for i in range(0, len(df_numeric.columns), n_cols):
        cols_batch = df_numeric.columns[i:i + n_cols]
        fig, axes = plt.subplots(1, len(cols_batch), figsize=(20, 4))
        if len(cols_batch) == 1:
            axes = [axes]
            
        for ax, col in zip(axes, cols_batch):
            sns.histplot(df_numeric[col].dropna(), bins=30, kde=True, ax=ax)
            ax.set_title(f'{col}')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}hist_batch_{i//n_cols + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Histogramas batch {i//n_cols + 1} salvos com sucesso')

def plot_correlation_matrix(df, output_dir, prefix=''):
    """Gera matriz de correlação (heatmap) com melhorias visuais.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
        prefix (str): Prefixo para o nome do arquivo.
    """
    df_numeric = df.select_dtypes(include='number')
    
    # Calcula correlação
    corr = df_numeric.corr()
    
    # Cria máscara para mostrar apenas correlações significativas
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Configura o tamanho da figura baseado no número de features
    n_features = len(corr.columns)
    figsize = (max(10, n_features * 0.5), max(8, n_features * 0.4))
    
    plt.figure(figsize=figsize)
    
    # Plota heatmap com melhorias visuais
    sns.heatmap(corr, 
                mask=mask,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'shrink': .5},
                annot_kws={'size': 8})
    
    plt.title('Matriz de Correlação', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('Matriz de correlação salva com sucesso')

def plot_feature_importance(df, output_dir, prefix=''):
    """Plota importância das features baseada na variância.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
        prefix (str): Prefixo para o nome do arquivo.
    """
    df_numeric = df.select_dtypes(include='number')
    
    # Calcula variância para cada feature
    variances = df_numeric.var()
    importance = variances / variances.sum() * 100
    
    # Ordena e pega as top 30 features
    importance = importance.sort_values(ascending=True).tail(30)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance.index, importance.values)
    plt.xlabel('Importância (%)')
    plt.title('Top 30 Features por Variância')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adiciona valores nas barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('Gráfico de importância das features salvo com sucesso')

def plot_na_counts(df, output_dir, prefix=''):
    """Plota contagem de valores nulos por coluna com melhorias visuais.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        output_dir (Path): Pasta de saída dos plots.
        prefix (str): Prefixo para o nome do arquivo.
    """
    na_counts = df.isna().sum()
    na_percent = (na_counts / len(df)) * 100
    
    # Cria DataFrame com contagens e percentuais
    na_df = pd.DataFrame({
        'Nulos': na_counts,
        'Percentual': na_percent
    }).sort_values('Nulos', ascending=False)
    
    # Plota apenas colunas com valores nulos
    na_df = na_df[na_df['Nulos'] > 0]
    
    if len(na_df) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Gráfico de barras
        bars = ax1.bar(na_df.index, na_df['Nulos'])
        ax1.set_title('Contagem de Valores Nulos por Coluna')
        ax1.set_ylabel('Número de Nulos')
        ax1.tick_params(axis='x', rotation=45)
        
        # Adiciona valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Gráfico de percentual
        ax2.bar(na_df.index, na_df['Percentual'], color='orange')
        ax2.set_ylabel('Percentual (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}na_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Contagem de valores nulos salva com sucesso')
    else:
        logger.info('Não há valores nulos no dataset')

def main():
    """Executa a análise exploratória dos datasets, gerando visualizações e salvando-as."""
    BASE_DIR = Path(__file__).resolve().parent
    CALIBRATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'calibration.csv'
    IDRC_CSV = BASE_DIR / 'data' / 'csv_new' / 'idrc_validation.csv'
    OUTPUT_PLOTS = BASE_DIR / 'results' / 'exploratory_plots'
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    
    # Carrega datasets
    df_calib, df_idrc = load_dataset(CALIBRATION_CSV, IDRC_CSV)
    
    # Gera visualizações para dataset de calibração
    plot_histograms(df_calib, OUTPUT_PLOTS, prefix='calib_')
    plot_correlation_matrix(df_calib, OUTPUT_PLOTS, prefix='calib_')
    plot_feature_importance(df_calib, OUTPUT_PLOTS, prefix='calib_')
    plot_na_counts(df_calib, OUTPUT_PLOTS, prefix='calib_')
    
    # Gera visualizações para dataset IDRC
    plot_histograms(df_idrc, OUTPUT_PLOTS, prefix='idrc_')
    plot_correlation_matrix(df_idrc, OUTPUT_PLOTS, prefix='idrc_')
    plot_na_counts(df_idrc, OUTPUT_PLOTS, prefix='idrc_')
    
    logger.info('Visualizações salvas em ' + str(OUTPUT_PLOTS))

if __name__ == "__main__":
    main() 