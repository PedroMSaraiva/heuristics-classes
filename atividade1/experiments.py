from eight_queens import EightQueens
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Configura o backend antes de importar pyplot
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import seaborn as sns
from datetime import datetime

def run_experiments(num_experiments: int = 100) -> Tuple[Dict, Dict, float]:
    """Executa múltiplos experimentos e coleta estatísticas detalhadas."""
    problem = EightQueens()
    
    # Dicionários para armazenar resultados
    hc_results = {
        'success_rate': 0,
        'avg_iterations': 0,
        'avg_time': 0,
        'min_iterations': float('inf'),
        'max_iterations': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'solutions': [],
        'iterations_history': [],
        'time_history': [],
        'conflicts_history': [],
        'local_minima_count': 0  # Número de vezes que ficou preso em mínimo local
    }
    
    sa_results = {
        'success_rate': 0,
        'avg_iterations': 0,
        'avg_time': 0,
        'min_iterations': float('inf'),
        'max_iterations': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'solutions': [],
        'iterations_history': [],
        'time_history': [],
        'conflicts_history': [],
        'acceptance_rate': 0  # Taxa de aceitação de movimentos piores
    }
    
    # Executa os experimentos
    print(f"\nIniciando {num_experiments} experimentos...")
    start_total = time.time()
    
    for i in range(num_experiments):
        if (i + 1) % 10 == 0:
            print(f"Progresso: {i + 1}/{num_experiments}")
            
        # Hill Climbing
        start_time = time.time()
        solution, conflicts_hist, iterations = problem.hill_climbing()
        hc_time = time.time() - start_time
        conflicts = problem.calculate_conflicts(solution)
        
        # Atualiza estatísticas HC
        hc_results['iterations_history'].append(iterations)
        hc_results['time_history'].append(hc_time)
        hc_results['conflicts_history'].append(conflicts)
        
        hc_results['min_iterations'] = min(hc_results['min_iterations'], iterations)
        hc_results['max_iterations'] = max(hc_results['max_iterations'], iterations)
        hc_results['min_time'] = min(hc_results['min_time'], hc_time)
        hc_results['max_time'] = max(hc_results['max_time'], hc_time)
        
        if conflicts > 0:
            hc_results['local_minima_count'] += 1
        
        if conflicts == 0:
            hc_results['success_rate'] += 1
            hc_results['solutions'].append(solution)
        
        # Simulated Annealing
        start_time = time.time()
        solution, conflicts_hist, iterations = problem.simulated_annealing()
        sa_time = time.time() - start_time
        conflicts = problem.calculate_conflicts(solution)
        
        # Atualiza estatísticas SA
        sa_results['iterations_history'].append(iterations)
        sa_results['time_history'].append(sa_time)
        sa_results['conflicts_history'].append(conflicts)
        
        sa_results['min_iterations'] = min(sa_results['min_iterations'], iterations)
        sa_results['max_iterations'] = max(sa_results['max_iterations'], iterations)
        sa_results['min_time'] = min(sa_results['min_time'], sa_time)
        sa_results['max_time'] = max(sa_results['max_time'], sa_time)
        
        if conflicts == 0:
            sa_results['success_rate'] += 1
            sa_results['solutions'].append(solution)
    
    total_time = time.time() - start_total
    
    # Calcula médias e estatísticas finais
    for results in [hc_results, sa_results]:
        results['success_rate'] = (results['success_rate'] / num_experiments) * 100
        results['avg_iterations'] = np.mean(results['iterations_history'])
        results['std_iterations'] = np.std(results['iterations_history'])
        results['avg_time'] = np.mean(results['time_history'])
        results['std_time'] = np.std(results['time_history'])
        results['avg_conflicts'] = np.mean(results['conflicts_history'])
        results['std_conflicts'] = np.std(results['conflicts_history'])
    
    hc_results['local_minima_rate'] = (hc_results['local_minima_count'] / num_experiments) * 100
    
    return hc_results, sa_results, total_time

def plot_detailed_comparison(hc_results: Dict, sa_results: Dict, total_time: float):
    """Plota uma análise detalhada dos resultados."""
    sns.set_style("whitegrid")
    
    # Configuração do layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Taxa de Sucesso
    ax1 = fig.add_subplot(gs[0, 0])
    algorithms = ['Hill Climbing', 'Simulated Annealing']
    success_rates = [hc_results['success_rate'], sa_results['success_rate']]
    sns.barplot(x=algorithms, y=success_rates, palette='viridis', ax=ax1)
    ax1.set_title('Taxa de Sucesso (%)')
    ax1.set_ylim(0, 100)
    
    # 2. Distribuição de Iterações
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=[hc_results['iterations_history'], sa_results['iterations_history']], 
                ax=ax2, palette='viridis')
    ax2.set_xticklabels(algorithms)
    ax2.set_title('Distribuição do Número de Iterações')
    
    # 3. Distribuição de Tempo
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=[hc_results['time_history'], sa_results['time_history']], 
                ax=ax3, palette='viridis')
    ax3.set_xticklabels(algorithms)
    ax3.set_title('Distribuição do Tempo de Execução (s)')
    
    # 4. Distribuição de Conflitos Finais
    ax4 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=hc_results['conflicts_history'], label='Hill Climbing', 
                 alpha=0.5, ax=ax4, color='blue')
    sns.histplot(data=sa_results['conflicts_history'], label='Simulated Annealing', 
                 alpha=0.5, ax=ax4, color='green')
    ax4.set_title('Distribuição dos Conflitos Finais')
    ax4.legend()
    
    # 5. Métricas Específicas
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = {
        'HC - Taxa Mínimo Local (%)': hc_results['local_minima_rate'],
        'HC - Média Iterações': hc_results['avg_iterations'],
        'SA - Média Iterações': sa_results['avg_iterations'],
        'HC - Média Conflitos': hc_results['avg_conflicts'],
        'SA - Média Conflitos': sa_results['avg_conflicts']
    }
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis', ax=ax5)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.set_title('Métricas Comparativas')
    
    # 6. Resumo Estatístico
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_text = f"""
    Resumo dos Experimentos:
    
    Hill Climbing:
    • Taxa de Sucesso: {hc_results['success_rate']:.1f}%
    • Iterações: {hc_results['avg_iterations']:.1f} ± {hc_results['std_iterations']:.1f}
    • Tempo: {hc_results['avg_time']:.3f}s ± {hc_results['std_time']:.3f}s
    • Taxa Mínimo Local: {hc_results['local_minima_rate']:.1f}%
    
    Simulated Annealing:
    • Taxa de Sucesso: {sa_results['success_rate']:.1f}%
    • Iterações: {sa_results['avg_iterations']:.1f} ± {sa_results['std_iterations']:.1f}
    • Tempo: {sa_results['avg_time']:.3f}s ± {sa_results['std_time']:.3f}s
    
    Tempo Total: {total_time:.2f}s
    """
    ax6.text(0, 1, summary_text, fontsize=10, va='top')
    
    plt.tight_layout()
    
    # Salva o gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'experiment_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_file(hc_results: Dict, sa_results: Dict, total_time: float):
    """Salva os resultados detalhados em um arquivo de texto."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'experiment_results_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("=== Resultados Detalhados dos Experimentos ===\n\n")
        
        f.write("Hill Climbing:\n")
        f.write(f"• Taxa de Sucesso: {hc_results['success_rate']:.2f}%\n")
        f.write(f"• Média de Iterações: {hc_results['avg_iterations']:.2f} ± {hc_results['std_iterations']:.2f}\n")
        f.write(f"• Tempo Médio: {hc_results['avg_time']:.4f}s ± {hc_results['std_time']:.4f}s\n")
        f.write(f"• Iterações (min/max): {hc_results['min_iterations']}/{hc_results['max_iterations']}\n")
        f.write(f"• Tempo (min/max): {hc_results['min_time']:.4f}s/{hc_results['max_time']:.4f}s\n")
        f.write(f"• Taxa de Mínimo Local: {hc_results['local_minima_rate']:.2f}%\n")
        f.write(f"• Média de Conflitos: {hc_results['avg_conflicts']:.2f} ± {hc_results['std_conflicts']:.2f}\n\n")
        
        f.write("Simulated Annealing:\n")
        f.write(f"• Taxa de Sucesso: {sa_results['success_rate']:.2f}%\n")
        f.write(f"• Média de Iterações: {sa_results['avg_iterations']:.2f} ± {sa_results['std_iterations']:.2f}\n")
        f.write(f"• Tempo Médio: {sa_results['avg_time']:.4f}s ± {sa_results['std_time']:.4f}s\n")
        f.write(f"• Iterações (min/max): {sa_results['min_iterations']}/{sa_results['max_iterations']}\n")
        f.write(f"• Tempo (min/max): {sa_results['min_time']:.4f}s/{sa_results['max_time']:.4f}s\n")
        f.write(f"• Média de Conflitos: {sa_results['avg_conflicts']:.2f} ± {sa_results['std_conflicts']:.2f}\n\n")
        
        f.write(f"Tempo Total de Execução: {total_time:.2f}s\n")
        
        # Salva algumas soluções encontradas
        if hc_results['solutions']:
            f.write("\nExemplo de Solução (Hill Climbing):\n")
            f.write(str(hc_results['solutions'][0]) + "\n")
        if sa_results['solutions']:
            f.write("\nExemplo de Solução (Simulated Annealing):\n")
            f.write(str(sa_results['solutions'][0]) + "\n")

def main():
    # Executa os experimentos
    print("Iniciando análise comparativa detalhada...")
    hc_results, sa_results, total_time = run_experiments(num_experiments=100)
    
    # Salva os resultados em arquivo
    save_results_to_file(hc_results, sa_results, total_time)
    
    # Plota os resultados
    plot_detailed_comparison(hc_results, sa_results, total_time)

if __name__ == "__main__":
    main() 