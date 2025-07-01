import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.logging_utils import get_logger

logger = get_logger('compare_results')

# ATENÇÃO: Os novos arquivos de dados estão em 'data/csv_new'.
# ATENÇÃO: Os resultados devem ser comparados com os arquivos gerados a partir dos novos CSVs em 'data/csv_new'.

def load_results(csv_paths):
    """Carrega múltiplos arquivos CSV de resultados.

    Args:
        csv_paths (list): Lista de caminhos para arquivos CSV.

    Returns:
        pd.DataFrame: DataFrame concatenado com todos os resultados.
    """
    logger.info('Iniciando leitura dos resultados')
    results = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df['Experimento'] = Path(csv_path).stem
        results.append(df)
    logger.info('Leitura dos resultados concluída')
    return pd.concat(results, ignore_index=True)

def plot_comparisons(pivot, metrics, output_dir):
    """Gera gráficos de barras para cada métrica comparada.

    Args:
        pivot (pd.DataFrame): Tabela pivoteada de métricas.
        metrics (list): Lista de métricas.
        output_dir (Path): Pasta de saída dos gráficos.
    """
    logger.info('Iniciando geração dos gráficos de comparação')
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        ax = pivot.loc[:, metric].plot(kind='bar', figsize=(8,5), title=f'Comparação de {metric}')
        ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f'comparacao_{metric}.png')
        plt.close()
        logger.info(f'Gráfico salvo: {output_dir / f"comparacao_{metric}.png"}')

def main():
    """Compara resultados de múltiplos experimentos salvos em CSV e gera gráficos de comparação."""
    logger.info('Iniciando a comparação de resultados')
    parser = argparse.ArgumentParser(description='Comparar resultados de múltiplos experimentos salvos em CSV.')
    parser.add_argument('csvs', nargs='+', help='Caminhos dos arquivos CSV de resultados')
    args = parser.parse_args()
    all_results = load_results(args.csvs)
    metrics = ['MSE','MAE','R2','BIAS','RMSE','SE']
    pivot = all_results.pivot(index='Conjunto', columns='Experimento', values=metrics)
    logger.info('\nResumo das métricas:')
    logger.info(pivot)
    output_dir = Path('results/comparacoes')
    plot_comparisons(pivot, metrics, output_dir)
    logger.info('Comparação de resultados concluída')

if __name__ == "__main__":
    main() 