import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mealpy.evolutionary_based.GA import BaseGA
from mealpy import FloatVar  # Import the FloatVar class
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import print as rprint
import os

# Set seaborn style for all plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

console = Console()

# Create directory for saving visualizations
os.makedirs("visualizations", exist_ok=True)

# Define the knapsack problem data from the table in the PDF
weights = np.array([8, 4, 7, 2, 6, 10, 3, 5, 11, 9])  # Weights in kg
values = np.array([3, 6, 16, 9, 7, 8, 5, 11, 13, 14])  # Values in R$
max_weight = 30  # Maximum capacity of the knapsack in kg

# Define item names for better visualization
item_names = [f"Item {i+1}" for i in range(len(weights))]

# Define the fitness function for the knapsack problem
def objective_function(solution):
    # Convert continuous solution to binary (0-1)
    solution_bin = np.round(solution).astype(int)
    
    # Calculate total weight and value
    total_weight = np.sum(weights * solution_bin)
    total_value = np.sum(values * solution_bin)
    
    # Apply penalty if weight constraint is violated
    if total_weight > max_weight:
        return -np.inf  # Invalid solution
    
    # The objective is to maximize value
    return total_value

# Define problem dimensions and bounds
n_dims = len(weights)
problem_dict = {
    "obj_func": objective_function,  # Changed from fit_func to obj_func
    "bounds": FloatVar(lb=[0] * n_dims, ub=[1] * n_dims),  # Using FloatVar
    "minmax": "max",     # We want to maximize the fitness
}

# Define experiment configurations
experiments = [
    {
        "name": "Experimento 1: Exploração Inicial",
        "pop_size": 200,
        "epoch": 100,
        "pm": 0.05,  # Mutation rate
        "pc": 0.7,   # Crossover rate
        "elitism": False
    },
    {
        "name": "Experimento 2: Exploração e Intensificação Equilibrada",
        "pop_size": 150,
        "epoch": 75,
        "pm": 0.02,
        "pc": 0.85,
        "elitism": False
    },
    {
        "name": "Experimento 3",
        "pop_size": 300,
        "epoch": 150,
        "pm": 0.1,
        "pc": 0.9,
        "elitism": False
    }
]

# Function to run a single experiment with a given configuration
def run_experiment(config, trial_num, progress=None):
    console.print(f"[bold blue]Rodando {config['name']} - Tentativa {trial_num+1}[/bold blue]")
    
    # For elitism, keep 5% of the population
    if config["elitism"]:
        keep_elites = int(0.05 * config["pop_size"])
    else:
        keep_elites = 2  # Default value in MEALPY
    
    # Create GA model with the specified parameters
    model = BaseGA(
        epoch=config["epoch"],
        pop_size=config["pop_size"],
        pc=config["pc"],
        pm=config["pm"],
        keep_elites=keep_elites,
        selection="tournament"  # Using tournament selection as specified
    )
    
    
    # Solve the problem
    best_agent = model.solve(problem_dict, mode="thread")
    best_position = best_agent.solution
    best_fitness = best_agent.target.fitness
    
    # Convert solution to binary
    best_solution = np.round(best_position).astype(int)
    
    # Calculate total weight and value
    total_weight = np.sum(weights * best_solution)
    total_value = np.sum(values * best_solution)
    
    result = {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "total_weight": total_weight,
        "total_value": total_value,
        "convergence": model.history.list_global_best_fit,
        "config": config["name"],
        "trial": trial_num + 1,
        "elitism": config["elitism"]
    }
    
    return result

# Function to run all experiments with multiple trials
def run_all_experiments():
    all_results = {}
    all_results_list = []  # For easier DataFrame creation
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn()
    ) as progress:
        
        # Run each experiment 5 times
        for config in experiments:
            results = []
            for trial in range(5):
                result = run_experiment(config, trial, progress)
                results.append(result)
                all_results_list.append(result)
                
                # Display interim result
                console.print(f"[green]Tentativa {trial+1} concluída: Fitness = {result['best_fitness']}, Peso = {result['total_weight']}[/green]")
            
            all_results[config["name"]] = results
        
        # Repeat with elitism
        for config in experiments:
            config_with_elitism = config.copy()
            config_with_elitism["elitism"] = True
            config_with_elitism["name"] = config["name"] + " (com Elitismo)"
            
            results = []
            for trial in range(5):
                result = run_experiment(config_with_elitism, trial, progress)
                results.append(result)
                all_results_list.append(result)
                
                # Display interim result
                console.print(f"[green]Tentativa {trial+1} concluída: Fitness = {result['best_fitness']}, Peso = {result['total_weight']}[/green]")
            
            all_results[config_with_elitism["name"]] = results
    
    # Create a DataFrame with all results for easier visualization
    results_df = pd.DataFrame(all_results_list)
    
    return all_results, results_df

# Function to visualize a knapsack solution
def visualize_knapsack(solution, title, filename):
    """Create a visualization of what items are in the knapsack"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create data for visualization
    selected_items = np.where(solution == 1)[0]
    selected_weights = weights[selected_items]
    selected_values = values[selected_items]
    selected_names = [item_names[i] for i in selected_items]
    
    # Create a colormap
    colors = sns.color_palette("viridis", len(selected_items))
    
    # Create the horizontal bar chart
    bars = ax.barh(selected_names, selected_weights, color=colors, alpha=0.7, label='Peso (kg)')
    
    # Add a second axis for values
    ax2 = ax.twiny()
    bars2 = ax2.barh(selected_names, selected_values, color=colors, alpha=0.3, label='Valor (R$)')
    
    # Add value labels to the bars
    for i, (w, v) in enumerate(zip(selected_weights, selected_values)):
        ax.text(w + 0.1, i, f"Peso: {w}kg", va='center')
        ax2.text(v - 4, i, f"Valor: R${v}", va='center', ha='center', color='black', fontweight='bold')
    
    # Add annotations for total weight and value
    total_weight = np.sum(selected_weights)
    total_value = np.sum(selected_values)
    ax.text(0.5, -0.1, f"Peso Total: {total_weight}/{max_weight}kg | Valor Total: R${total_value}", 
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set labels and title
    ax.set_xlabel('Peso (kg)')
    ax2.set_xlabel('Valor (R$)')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Function to create a heat map of solutions
def visualize_solutions_heatmap(results_df, filename="solutions_heatmap"):
    """Create a heatmap showing which items are selected in different experiments"""
    # Extract solutions and create a matrix
    all_solutions = np.array([list(r["best_solution"]) for _, r in results_df.iterrows()])
    
    # Create a DataFrame with solutions
    solution_df = pd.DataFrame(all_solutions, columns=item_names)
    solution_df['Experimento'] = results_df['config']
    solution_df['Trial'] = results_df['trial']
    solution_df['Fitness'] = results_df['best_fitness']
    
    # Reshape data for the heatmap
    pivot_df = solution_df.pivot_table(
        index=['Experimento', 'Trial'], 
        values=item_names, 
        aggfunc='first'
    )
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_df, cmap="YlGnBu", cbar_kws={'label': 'Selecionado (1) / Não Selecionado (0)'})
    plt.title('Itens Selecionados por Experimento e Tentativa', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Function to visualize fitness distribution
def visualize_fitness_distribution(results_df, filename="fitness_distribution"):
    """Create violin plots showing the distribution of fitness values by experiment"""
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    sns.violinplot(x='config', y='best_fitness', data=results_df, hue='elitism', 
                  palette="Set2", split=True, inner="quartile")
    
    # Add individual data points
    sns.swarmplot(x='config', y='best_fitness', data=results_df, color='white', 
                 alpha=0.5, size=5)
    
    plt.title('Distribuição dos Valores de Fitness por Experimento', fontsize=16, fontweight='bold')
    plt.xlabel('Experimento', fontsize=14)
    plt.ylabel('Fitness (Valor Total)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Elitismo')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Function to visualize weight vs value
def visualize_weight_value_relationship(results_df, filename="weight_value_relationship"):
    """Create a scatter plot showing the relationship between weight and value"""
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(x='total_weight', y='total_value', data=results_df, 
                   hue='config', style='elitism', s=100, alpha=0.7)
    
    plt.axvline(x=max_weight, color='red', linestyle='--', 
               label=f'Limite de Peso ({max_weight}kg)')
    
    plt.title('Relação entre Peso e Valor das Soluções', fontsize=16, fontweight='bold')
    plt.xlabel('Peso Total (kg)', fontsize=14)
    plt.ylabel('Valor Total (R$)', fontsize=14)
    plt.legend(title='Experimento', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Function to visualize convergence with better styling
def visualize_convergence(all_results, filename="convergence_evolution"):
    """Create an enhanced visualization of convergence curves"""
    plt.figure(figsize=(14, 10))
    
    # Set a color palette
    palette = sns.color_palette("husl", len(all_results))
    
    # Plot each convergence curve
    for i, (exp_name, results) in enumerate(all_results.items()):
        best_run_idx = np.argmax([r["best_fitness"] for r in results])
        convergence = results[best_run_idx]["convergence"]
        
        # Add some noise to epochs for visibility if they overlap
        epochs = np.arange(len(convergence))
        
        # Plot with better styling
        sns.lineplot(x=epochs, y=convergence, label=exp_name, color=palette[i], 
                    linewidth=2.5, alpha=0.8)
    
    # Enhance the plot
    plt.title('Evolução da Convergência dos Algoritmos Genéticos', fontsize=18, fontweight='bold')
    plt.xlabel('Gerações', fontsize=14)
    plt.ylabel('Melhor Fitness', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a legend with better positioning
    plt.legend(title='Experimento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add annotations for the best fitness
    for i, (exp_name, results) in enumerate(all_results.items()):
        best_run_idx = np.argmax([r["best_fitness"] for r in results])
        best_fitness = results[best_run_idx]["best_fitness"]
        final_epoch = len(results[best_run_idx]["convergence"]) - 1
        
        plt.annotate(f"Fitness: {best_fitness:.1f}", 
                    xy=(final_epoch, results[best_run_idx]["convergence"][-1]), 
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=10, color=palette[i], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Function to compare best solutions across all experiments
def visualize_all_best_solutions(all_best_solutions, filename="all_best_solutions_comparison"):
    """Create a comparison visualization of all best solutions"""
    # Create a figure with subplots for each solution
    num_solutions = len(all_best_solutions)
    fig, axs = plt.subplots(num_solutions, 1, figsize=(14, 6 * num_solutions))
    
    # Set a color palette
    palette = sns.color_palette("viridis", n_dims)
    
    # Plot each solution
    for i, (exp_name, solution_data) in enumerate(all_best_solutions.items()):
        solution = solution_data["solution"]
        total_weight = solution_data["weight"]
        total_value = solution_data["value"]
        
        # Create the bar chart
        ax = axs[i]
        bars = ax.bar(item_names, solution, color=palette, alpha=0.7)
        
        # Add a title for each subplot
        ax.set_title(f"{exp_name} - Fitness: {total_value} - Peso: {total_weight}/{max_weight}kg", 
                    fontsize=14, fontweight='bold')
        
        # Add item values as annotations
        for j, (val, bar) in enumerate(zip(solution, bars)):
            if val == 1:
                ax.text(j, 0.5, f"V: {values[j]}\nP: {weights[j]}kg", 
                       ha='center', va='center', color='white', fontweight='bold')
        
        # Setting y-axis to be binary (0 or 1)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Não', 'Sim'])
        ax.set_ylabel('Selecionado')
        
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Generate a detailed visual comparison of best solutions
def visualize_solution_comparison_grid(all_best_solutions, filename="solutions_comparison_grid"):
    """Create a grid visualization comparing item selection across best solutions"""
    # Create data for heatmap
    solutions_matrix = []
    exp_names = []
    
    for exp_name, solution_data in all_best_solutions.items():
        solutions_matrix.append(solution_data["solution"])
        exp_names.append(exp_name)
    
    solutions_matrix = np.array(solutions_matrix)
    
    # Create the heatmap
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(solutions_matrix, cmap="YlGnBu", 
                    xticklabels=item_names, yticklabels=exp_names, 
                    linewidths=0.5, cbar_kws={'label': 'Selecionado'})
    
    # Add annotations
    for i in range(len(exp_names)):
        for j in range(len(item_names)):
            if solutions_matrix[i, j] == 1:
                text = f"V:{values[j]}\nP:{weights[j]}"
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                       color='black', fontsize=9, fontweight='bold')
    
    # Add weight and value totals
    for i, (exp_name, solution_data) in enumerate(all_best_solutions.items()):
        ax.text(len(item_names) + 0.5, i + 0.5, 
               f"Total: {solution_data['value']}R$ | {solution_data['weight']}kg", 
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.title('Comparação das Melhores Soluções por Experimento', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_items_text(solution):
    """Helper function to generate formatted text for selected items"""
    text = ""
    for i, include in enumerate(solution):
        if include:
            text += f"- Item {i+1}: peso = {weights[i]}kg, valor = R${values[i]}\n"
    return text

# Function to analyze and display results
def analyze_results(all_results, results_df):
    # Create a summary table using rich
    table = Table(title="Resultados dos Experimentos")
    
    table.add_column("Experimento", style="cyan")
    table.add_column("Fitness Média", style="green")
    table.add_column("Desvio Padrão", style="yellow")
    table.add_column("Peso Médio", style="magenta")
    
    summary = []
    all_best_solutions = {}  # Dictionary to store all best solutions
    
    for exp_name, results in all_results.items():
        best_fitnesses = [r["best_fitness"] for r in results]
        mean_fitness = np.mean(best_fitnesses)
        std_fitness = np.std(best_fitnesses)
        
        best_weights = [r["total_weight"] for r in results]
        mean_weight = np.mean(best_weights)
        
        best_run_idx = np.argmax(best_fitnesses)
        best_solution = results[best_run_idx]["best_solution"]
        best_fitness = results[best_run_idx]["best_fitness"]
        best_weight = results[best_run_idx]["total_weight"]
        
        # Store the best solution for this experiment
        all_best_solutions[exp_name] = {
            "solution": best_solution,
            "value": best_fitness,
            "weight": best_weight
        }
        
        summary.append({
            "Experimento": exp_name,
            "Fitness Média": mean_fitness,
            "Desvio Padrão": std_fitness,
            "Peso Médio": mean_weight,
            "Melhor Solução": best_solution
        })
        
        table.add_row(
            exp_name,
            f"{mean_fitness:.2f}",
            f"{std_fitness:.2f}",
            f"{mean_weight:.2f}"
        )
    
    console.print(table)
    
    # Display the best solution from each experiment
    console.print("\n[bold green]Melhores Soluções de cada Experimento:[/bold green]")
    
    for exp_name, solution_data in all_best_solutions.items():
        solution = solution_data["solution"]
        value = solution_data["value"]
        weight = solution_data["weight"]
        
        solution_panel = Panel.fit(
            f"""
[bold]Experimento: {exp_name}[/bold]

Solução binária: {solution}

Itens selecionados:
{generate_items_text(solution)}

Peso total: {weight}/{max_weight}kg
Valor total: R${value}
            """,
            title=f"Melhor Solução - {exp_name}",
            border_style="cyan"
        )
        
        console.print(solution_panel)
    
    # Find the best solution overall
    best_exp = max(summary, key=lambda x: x["Fitness Média"])
    best_solution = best_exp["Melhor Solução"]
    total_weight = np.sum(weights * best_solution)
    total_value = np.sum(values * best_solution)
    
    # Display detailed results panel for the global best
    result_panel = Panel.fit(
        f"""
[bold]Melhor solução global (do experimento {best_exp['Experimento']})[/bold]

Solução binária: {best_solution}

Itens selecionados:
{generate_items_text(best_solution)}

Peso total: {total_weight}/{max_weight}kg
Valor total: R${total_value}
        """,
        title="MELHOR SOLUÇÃO GLOBAL",
        border_style="green"
    )
    
    console.print("\n", result_panel)
    
    # Create visualizations
    console.print("[bold]Gerando visualizações...[/bold]")
    
    # 1. Visualize the best solution as a knapsack
    visualize_knapsack(
        best_solution, 
        f"Melhor Solução Global: {best_exp['Experimento']}", 
        "best_solution_knapsack"
    )
    
    # 2. Visualize all best solutions
    console.print("[bold]Gerando visualizações das melhores soluções de cada experimento...[/bold]")
    
    # Visualize each best solution individually
    for exp_name, solution_data in all_best_solutions.items():
        safe_exp_name = exp_name.replace(":", "_").replace(" ", "_")
        visualize_knapsack(
            solution_data["solution"],
            f"Melhor Solução: {exp_name}",
            f"best_solution_{safe_exp_name}"
        )
    
    # Create comparison visualizations
    visualize_all_best_solutions(all_best_solutions)
    visualize_solution_comparison_grid(all_best_solutions)
    
    # 3. Visualize all solutions as a heatmap
    visualize_solutions_heatmap(results_df)
    
    # 4. Visualize fitness distribution
    visualize_fitness_distribution(results_df)
    
    # 5. Visualize weight vs value relationship
    visualize_weight_value_relationship(results_df)
    
    # 6. Visualize convergence with better styling
    visualize_convergence(all_results)
    
    console.print("[bold green]Todas as visualizações foram geradas e salvas na pasta 'visualizations'[/bold green]")

# Run all experiments and analyze results
if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]Algoritmo Genético para o Problema da Mochila[/bold]\n\n"
        f"Número de itens: {n_dims}\n"
        f"Capacidade máxima da mochila: {max_weight}kg",
        title="Configuração do Problema",
        border_style="blue"
    ))
    
    all_results, results_df = run_all_experiments()
    analyze_results(all_results, results_df)
