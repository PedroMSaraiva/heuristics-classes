import numpy as np
import logging
import os
from gpu_optimized import optimized_f6_wrapper, FloatVar, CustomACOR, OriginalPSO
from mealpy.utils.agent import Agent
from mealpy.utils.problem import Problem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("teste")

# Verificar se a função objetivo funciona isoladamente
test_vector = np.random.uniform(-100, 100, 10)
fitness_value = optimized_f6_wrapper(test_vector)
print(f"Teste da função objetivo com vetor aleatório: {fitness_value:.6f}")

# Definir um problema simples para testar
problem_size = 10
problem_dict = {
    "obj_func": optimized_f6_wrapper, 
    "bounds": FloatVar(lb=[-100] * problem_size, ub=[100] * problem_size),
    "minmax": "min",
    "name": "F6-Test"
}
problem = Problem(**problem_dict)

# Criar e testar agente individual
solution = np.random.uniform(-100, 100, problem_size)
agent = Agent(solution=solution) 
agent.target = problem.get_target(solution)
print(f"Agente individual - Solução: {solution[:3]}... - Fitness: {agent.target.fitness:.6f}")

# Configurações simples
config = {"epoch": 10, "pop_size": 10}

# Testar ACO
print("\n===== TESTE ACO =====")
aco = CustomACOR(epoch=config["epoch"], pop_size=config["pop_size"])
aco_best = aco.solve(problem, mode="thread")
print(f"ACO melhor fitness: {aco_best.target.fitness if hasattr(aco_best.target, 'fitness') else 'N/A'}")
print(f"ACO histórico fitness: {aco.history.list_global_best_fit}")

# Testar PSO para comparação
print("\n===== TESTE PSO =====")
pso = OriginalPSO(epoch=config["epoch"], pop_size=config["pop_size"])
pso_best = pso.solve(problem, mode="thread")
print(f"PSO melhor fitness: {pso_best.target.fitness if hasattr(pso_best.target, 'fitness') else 'N/A'}")
print(f"PSO histórico fitness: {pso.history.list_global_best_fit}")

print("\n===== COMPARAÇÃO =====")
print(f"ACO histórico tamanho: {len(aco.history.list_global_best_fit)}")
print(f"PSO histórico tamanho: {len(pso.history.list_global_best_fit)}")
print(f"ACO população final: {len(aco.pop)} agentes")
print(f"PSO população final: {len(pso.pop)} agentes")

# Salvar os históricos e comparar
os.makedirs("debug", exist_ok=True)

# Verificar histórico de população
if hasattr(aco.history, 'list_population'):
    aco_pop_history = aco.history.list_population
    print(f"ACO histórico população: {len(aco_pop_history)} épocas")
    
    # Verificar se as populações contêm agentes válidos
    if aco_pop_history:
        first_pop = aco_pop_history[0]
        print(f"ACO primeira população: {len(first_pop)} agentes")
        if first_pop:
            first_agent = first_pop[0]
            print(f"ACO primeiro agente: {first_agent.solution[:3]}... - Fitness: {first_agent.target.fitness if hasattr(first_agent.target, 'fitness') else 'N/A'}")
else:
    print("ACO não tem histórico de população!")

# Verificar histórico de população do PSO
if hasattr(pso.history, 'list_population'):
    pso_pop_history = pso.history.list_population
    print(f"PSO histórico população: {len(pso_pop_history)} épocas")
    
    # Verificar se as populações contêm agentes válidos
    if pso_pop_history:
        first_pop = pso_pop_history[0]
        print(f"PSO primeira população: {len(first_pop)} agentes")
        if first_pop:
            first_agent = first_pop[0]
            print(f"PSO primeiro agente: {first_agent.solution[:3]}... - Fitness: {first_agent.target.fitness if hasattr(first_agent.target, 'fitness') else 'N/A'}")
else:
    print("PSO não tem histórico de população!")