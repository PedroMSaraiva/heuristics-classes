import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from datetime import datetime
from src.logging_utils import get_logger

logger = get_logger('analyze_multiple_runs')

# Configurações de plot
plt.style.use('default')
sns.set_palette("husl")

BASE_DIR = Path(__file__).resolve().parent
MULTIPLE_RUNS_DIR = BASE_DIR / 'results' / 'multiple_runs'
ANALYSIS_PLOTS_DIR = BASE_DIR / 'results' / 'analysis_plots'
ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_results():
    """Carrega os resultados mais recentes dos algoritmos baseline e genético.
    
    Returns:
        tuple: (baseline_df, genetic_df) ou (None, None) se não encontrar arquivos.
    """
    # Busca pelos arquivos CSV mais recentes
    baseline_files = glob.glob(str(MULTIPLE_RUNS_DIR / 'baseline_mlr_30_runs_*.csv'))
    genetic_files = glob.glob(str(MULTIPLE_RUNS_DIR / 'genetic_mlr_30_runs_*.csv'))
    
    if not baseline_files or not genetic_files:
        logger.error("Arquivos de resultados não encontrados!")
        logger.error("Execute primeiro os scripts baseline_mlr_multiple_runs.py e genetic_mlr_multiple_runs.py")
        return None, None
    
    # Pega os arquivos mais recentes
    latest_baseline = max(baseline_files, key=lambda x: Path(x).stat().st_mtime)
    latest_genetic = max(genetic_files, key=lambda x: Path(x).stat().st_mtime)
    
    logger.info(f"Carregando baseline: {latest_baseline}")
    logger.info(f"Carregando genetic: {latest_genetic}")
    
    baseline_df = pd.read_csv(latest_baseline)
    genetic_df = pd.read_csv(latest_genetic)
    
    # Adiciona coluna de algoritmo
    baseline_df['algorithm'] = 'Baseline MLR'
    genetic_df['algorithm'] = 'Genetic MLR'
    
    return baseline_df, genetic_df

def create_comparison_boxplots(baseline_df, genetic_df):
    """Cria boxplots comparativos das métricas entre os algoritmos.
    
    Args:
        baseline_df (pd.DataFrame): Resultados do baseline MLR.
        genetic_df (pd.DataFrame): Resultados do genetic MLR.
    """
    # Combina os datasets
    combined_df = pd.concat([baseline_df, genetic_df], ignore_index=True)
    
    # Lista de métricas para plotar
    metrics = ['R2', 'MSE', 'RMSE', 'MAE', 'BIAS', 'SE']
    
    # Cria subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparação de Métricas - Baseline MLR vs Genetic MLR (30 Execuções)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Cria boxplot para cada dataset (validation/test) e algoritmo
        sns.boxplot(data=combined_df, x='dataset', y=metric, hue='algorithm', ax=ax)
        
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Adiciona valores médios como anotações
        for j, dataset in enumerate(['test', 'validation']):
            for k, algorithm in enumerate(['Baseline MLR', 'Genetic MLR']):
                subset = combined_df[(combined_df['dataset'] == dataset) & 
                                   (combined_df['algorithm'] == algorithm)]
                if not subset.empty:
                    mean_val = subset[metric].mean()
                    ax.text(j + (k-0.5)*0.3, mean_val, f'{mean_val:.3f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Ajusta a legenda para aparecer apenas no primeiro subplot
        if i == 0:
            ax.legend(title='Algoritmo', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.get_legend().remove()
    
    plt.tight_layout()
    
    # Salva o plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = ANALYSIS_PLOTS_DIR / f'comparison_boxplots_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    logger.info(f"Boxplots comparativos salvos em: {plot_path}")

def create_individual_metric_plots(baseline_df, genetic_df):
    """Cria plots individuais para cada métrica com análise detalhada.
    
    Args:
        baseline_df (pd.DataFrame): Resultados do baseline MLR.
        genetic_df (pd.DataFrame): Resultados do genetic MLR.
    """
    combined_df = pd.concat([baseline_df, genetic_df], ignore_index=True)
    metrics = ['R2', 'MSE', 'RMSE', 'MAE', 'BIAS', 'SE']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Análise Detalhada - {metric}', fontsize=14, fontweight='bold')
        
        # Plot 1: Boxplot por dataset
        ax1 = axes[0]
        sns.boxplot(data=combined_df, x='dataset', y=metric, hue='algorithm', ax=ax1)
        ax1.set_title(f'Distribuição de {metric} por Dataset')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Violin plot para mostrar distribuição completa
        ax2 = axes[1]
        sns.violinplot(data=combined_df, x='dataset', y=metric, hue='algorithm', 
                      split=True, ax=ax2)
        ax2.set_title(f'Distribuição Detalhada de {metric}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva plot individual
        plot_path = ANALYSIS_PLOTS_DIR / f'{metric.lower()}_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot de {metric} salvo em: {plot_path}")

def create_features_analysis(genetic_df):
    """Analisa o número de features selecionadas pelo algoritmo genético.
    
    Args:
        genetic_df (pd.DataFrame): Resultados do genetic MLR.
    """
    if 'num_features' not in genetic_df.columns:
        logger.warning("Coluna 'num_features' não encontrada nos dados genéticos")
        return
    
    # Pega apenas dados de validação (evita duplicação)
    genetic_val = genetic_df[genetic_df['dataset'] == 'validation'].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Análise do Número de Features Selecionadas (Genetic MLR)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Histograma
    ax1 = axes[0]
    ax1.hist(genetic_val['num_features'], bins=15, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Número de Features')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição do Número de Features')
    ax1.grid(True, alpha=0.3)
    
    # Adiciona estatísticas
    mean_features = genetic_val['num_features'].mean()
    std_features = genetic_val['num_features'].std()
    ax1.axvline(mean_features, color='red', linestyle='--', 
               label=f'Média: {mean_features:.1f}±{std_features:.1f}')
    ax1.legend()
    
    # Plot 2: Boxplot
    ax2 = axes[1]
    ax2.boxplot(genetic_val['num_features'])
    ax2.set_ylabel('Número de Features')
    ax2.set_title('Boxplot do Número de Features')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relação entre número de features e R²
    ax3 = axes[2]
    ax3.scatter(genetic_val['num_features'], genetic_val['R2'], alpha=0.7)
    ax3.set_xlabel('Número de Features')
    ax3.set_ylabel('R² (Validação)')
    ax3.set_title('Relação Features vs R²')
    ax3.grid(True, alpha=0.3)
    
    # Adiciona linha de tendência
    z = np.polyfit(genetic_val['num_features'], genetic_val['R2'], 1)
    p = np.poly1d(z)
    ax3.plot(genetic_val['num_features'], p(genetic_val['num_features']), 
            "r--", alpha=0.8, label=f'Tendência (slope={z[0]:.4f})')
    ax3.legend()
    
    plt.tight_layout()
    
    # Salva plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = ANALYSIS_PLOTS_DIR / f'features_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Análise de features salva em: {plot_path}")

def create_statistical_summary_table(baseline_df, genetic_df):
    """Cria uma tabela resumo com estatísticas descritivas.
    
    Args:
        baseline_df (pd.DataFrame): Resultados do baseline MLR.
        genetic_df (pd.DataFrame): Resultados do genetic MLR.
    """
    metrics = ['R2', 'MSE', 'RMSE', 'MAE', 'BIAS', 'SE']
    datasets = ['validation', 'test']
    algorithms = ['Baseline MLR', 'Genetic MLR']
    
    # Combina os datasets
    combined_df = pd.concat([baseline_df, genetic_df], ignore_index=True)
    
    summary_data = []
    
    for algorithm in algorithms:
        for dataset in datasets:
            subset = combined_df[(combined_df['algorithm'] == algorithm) & 
                               (combined_df['dataset'] == dataset)]
            
            if not subset.empty:
                for metric in metrics:
                    values = subset[metric]
                    summary_data.append({
                        'Algorithm': algorithm,
                        'Dataset': dataset,
                        'Metric': metric,
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Median': values.median(),
                        'Q25': values.quantile(0.25),
                        'Q75': values.quantile(0.75)
                    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Salva tabela resumo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = ANALYSIS_PLOTS_DIR / f'statistical_summary_{timestamp}.csv'
    summary_df.to_csv(table_path, index=False)
    
    logger.info(f"Tabela de estatísticas salva em: {table_path}")
    
    # Exibe resumo no log
    logger.info("\n" + "="*80)
    logger.info("RESUMO ESTATÍSTICO COMPARATIVO")
    logger.info("="*80)
    
    for dataset in datasets:
        logger.info(f"\n{dataset.upper()}:")
        
        for metric in ['R2', 'MSE', 'MAE']:  # Métricas principais
            baseline_vals = summary_df[(summary_df['Algorithm'] == 'Baseline MLR') & 
                                     (summary_df['Dataset'] == dataset) & 
                                     (summary_df['Metric'] == metric)]
            genetic_vals = summary_df[(summary_df['Algorithm'] == 'Genetic MLR') & 
                                    (summary_df['Dataset'] == dataset) & 
                                    (summary_df['Metric'] == metric)]
            
            if not baseline_vals.empty and not genetic_vals.empty:
                b_mean = baseline_vals['Mean'].iloc[0]
                b_std = baseline_vals['Std'].iloc[0]
                g_mean = genetic_vals['Mean'].iloc[0]
                g_std = genetic_vals['Std'].iloc[0]
                
                improvement = ((g_mean - b_mean) / b_mean) * 100 if metric == 'R2' else ((b_mean - g_mean) / b_mean) * 100
                
                logger.info(f"  {metric}:")
                logger.info(f"    Baseline: {b_mean:.4f} ± {b_std:.4f}")
                logger.info(f"    Genetic:  {g_mean:.4f} ± {g_std:.4f}")
                logger.info(f"    Melhoria: {improvement:+.2f}%")
    
    logger.info("\n" + "="*80)
    
    return summary_df

def main():
    """Executa a análise completa dos resultados de múltiplas execuções."""
    logger.info("Iniciando análise dos resultados de múltiplas execuções")
    
    # Carrega os resultados
    baseline_df, genetic_df = load_latest_results()
    
    if baseline_df is None or genetic_df is None:
        return
    
    logger.info(f"Dados carregados:")
    logger.info(f"   Baseline: {len(baseline_df)} registros")
    logger.info(f"   Genetic: {len(genetic_df)} registros")
    
    try:
        # Cria todos os plots de análise
        logger.info("Criando boxplots comparativos...")
        create_comparison_boxplots(baseline_df, genetic_df)
        
        logger.info("Criando plots individuais de métricas...")
        create_individual_metric_plots(baseline_df, genetic_df)
        
        logger.info("Analisando seleção de features...")
        create_features_analysis(genetic_df)
        
        logger.info("Criando tabela de estatísticas...")
        summary_df = create_statistical_summary_table(baseline_df, genetic_df)
        
        logger.info("Análise completa concluída!")
        logger.info(f"Plots salvos em: {ANALYSIS_PLOTS_DIR}")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {e}")
        raise

if __name__ == "__main__":
    main() 