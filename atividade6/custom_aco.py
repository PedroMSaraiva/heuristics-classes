#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020

"""
Versão personalizada do algoritmo ACO (Ant Colony Optimization) compatível com NumPy 2.0+
"""

import numpy as np
from mealpy.optimizer import Optimizer
from numba import njit

# Numba-jitted helper function to select an agent's index based on weights
@njit
def _select_agent_idx_numba(weights, rand_val):
    temp_sum = 0.0
    selected_idx = 0
    for i in range(len(weights)):
        temp_sum += weights[i]
        if temp_sum >= rand_val:
            selected_idx = i
            break
    return selected_idx

# Numba-jitted helper function to calculate sigma for a specific dimension
@njit
def _calculate_sigma_for_dim_j_numba(solutions_col_j, zeta, pop_size_for_logic):
    n_agents_in_col = len(solutions_col_j)

    if n_agents_in_col == 0:
        return 0.0  # Should not happen if population is valid
    
    # Se pop_size_for_logic (que é self.pop_size do algoritmo) for 1, 
    # ou se houver apenas uma solução na coluna (o que implicaria pop_size_for_logic=1 ou erro),
    # o comportamento original era retornar zeta. Isso é para evitar divisão por zero ou comportamento indefinido.
    if pop_size_for_logic <= 1 or n_agents_in_col <=1:
        return zeta

    # Vectorized calculation of circular differences
    # np.roll(solutions_col_j, 1) faz o shift circular para pegar o elemento anterior (o último para o primeiro)
    diffs = np.abs(solutions_col_j - np.roll(solutions_col_j, 1))
    sigma_sum = np.sum(diffs)
    
    # A fórmula original era zeta * sigma_sum / (pop_size_for_logic - 1.0)
    # Garantir que pop_size_for_logic - 1.0 não seja zero já foi tratado acima.
    return zeta * sigma_sum / (pop_size_for_logic - 1.0)

# Main Numba-jitted function to generate a single child solution vector
@njit
def _generate_one_child_solution_numba(n_dims, weights_arr, solutions_arr_2d, zeta_val, pop_size_param):
    child = np.zeros(n_dims)
    num_agents_in_pop = solutions_arr_2d.shape[0]

    if num_agents_in_pop == 0: # Safety check, should not occur with proper initialization
        return child 

    for j in range(n_dims):
        # Select agent for this dimension's mean
        rand_val_selection = np.random.uniform(0.0, 1.0)
        selected_agent_idx = _select_agent_idx_numba(weights_arr, rand_val_selection)
        # Ensure index is valid for solutions_arr_2d (it should be if weights_arr length matches pop_size_param)
        selected_agent_idx = min(selected_agent_idx, num_agents_in_pop - 1) 

        # Extract column for current dimension j from the solutions array
        solutions_col_j = solutions_arr_2d[:, j]
        sigma_j = _calculate_sigma_for_dim_j_numba(solutions_col_j, zeta_val, pop_size_param)
        
        mean_pos_j = solutions_arr_2d[selected_agent_idx, j]
        child[j] = np.random.normal(mean_pos_j, sigma_j)
        
    return child


class CustomACOR(Optimizer):
    """
    Versão personalizada do algoritmo Ant Colony Optimization for Continuous Domains (ACOR)
    adaptada para ser compatível com NumPy 2.0+ e otimizada com Numba.

    References:
        1. Socha, K. and Dorigo, M., 2008. Ant colony optimization for continuous domains.
        European journal of operational research, 185(3), pp.1155-1173.
    """

    def __init__(self, epoch=10000, pop_size=50, sample_count=30, intent_factor=0.5, zeta=1.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 50
            sample_count (int): Number of Newly Generated Sample Points, default = 30
            intent_factor (float): Intensification Factor (Selection Pressure), default = 0.5
            zeta (float): Deviation-Distance Ratio, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sample_count = self.validator.check_int("sample_count", sample_count, [2, 10000])
        self.intent_factor = self.validator.check_float("intent_factor", intent_factor, (0, 1.0))
        self.zeta = self.validator.check_float("zeta", zeta, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "sample_count", "intent_factor", "zeta"])
        self.sort_flag = True
        
        # Definir constantes para acessar posição e target nas soluções
        # Estas constantes são usadas na classe Optimizer da mealpy
        # No entanto, em mealpy, geralmente acessamos agent.solution e agent.target.fitness
        # self.ID_POS = 0  # Índice da posição na solução - Removido
        # self.ID_TAR = 1  # Índice do target na solução - Removido

    def initialize_variables(self):
        self.weights = np.zeros(self.pop_size)
        for i in range(0, self.pop_size):
            self.weights[i] = 1.0 / (self.intent_factor * self.pop_size * np.sqrt(2.0 * np.pi)) * \
                              np.exp(-0.5 * ((i - 1.0) / (self.intent_factor * self.pop_size)) ** 2)
        self.weights = self.weights / np.sum(self.weights)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        Optimized with Numba.
        Args:
            epoch (int): The current iteration
        """
        if not self.pop: # Population should be initialized by Optimizer base class
            return

        # Convert current population to NumPy array for Numba-compatibility
        # solutions_array has shape (pop_size, n_dims)
        try:
            solutions_array = np.array([agent.solution for agent in self.pop], dtype=np.float64)
        except AttributeError: # Fallback if self.pop contains non-agent objects (e.g. lists)
             # This might happen if an intermediate step failed to create proper Agent objects
             # Try to recover if they are list-like [solution_vector, target_obj, generation]
            temp_solutions = []
            for item in self.pop:
                if hasattr(item, 'solution'):
                    temp_solutions.append(item.solution)
                elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], np.ndarray):
                    temp_solutions.append(item[0]) # Assuming solution is the first element
            if not temp_solutions: # If still no valid solutions found
                # self.logger.warning("Population contains no valid solutions in evolve method.")
                return
            solutions_array = np.array(temp_solutions, dtype=np.float64)

        # Ensure solutions_array is usable
        if solutions_array.ndim != 2 or solutions_array.shape[0] == 0:
            # self.logger.warning(f"Solutions array is not valid in evolve. Shape: {solutions_array.shape}")
            return
        
        # Check if solutions_array.shape[0] matches self.pop_size, use the smaller if mismatch
        # to prevent errors in Numba, though they should match.
        # effective_pop_size = min(solutions_array.shape[0], self.pop_size)
        # This should ideally not be needed if self.pop is consistent with self.pop_size.
        # For now, we assume solutions_array.shape[0] == self.pop_size based on Mealpy's structure.

        if len(self.weights) != self.pop_size or solutions_array.shape[0] != self.pop_size :
            # self.logger.warning("Mismatch between weights, solutions_array, and pop_size. Re-initializing weights or skipping evolve.")
            # This indicates an inconsistent state. Might be safer to re-initialize weights
            # or skip this evolution step. For simplicity, let's assume they match or are handled.
            # If weights are not matching solutions_array.shape[0], Numba calls might fail.
            # A robust implementation might re-calculate weights if pop_size changed, or error out.
            # For this optimization, we assume consistency is maintained by the Optimizer.
            pass

        newly_generated_agents = []
        for _ in range(self.sample_count):
            # Generate one new solution vector using the Numba-jitted function
            child_solution_vec = _generate_one_child_solution_numba(
                self.problem.n_dims, 
                self.weights,           # Should be (pop_size,)
                solutions_array,       # Should be (pop_size, n_dims)
                self.zeta, 
                self.pop_size          # The algorithm's parameter for population size
            )
            
            pos_new = self.correct_solution(child_solution_vec)
            # target = self.get_target(pos_new) # generate_agent typically recalculates target
            
            # Create a new Mealpy Agent object.
            # self.generate_agent(solution) creates an agent and calculates its target.
            new_agent = self.generate_agent(pos_new) 
            newly_generated_agents.append(new_agent)

        # Update the main population with the new solutions
        self.pop = self.get_sorted_population(self.pop + newly_generated_agents, self.pop_size)

    # Sobrescrever o método get_index_roulette_wheel_selection para usar np.ptp em vez de list_fitness.ptp()
    def get_index_roulette_wheel_selection(self, list_fitness=None):
        """
        Versão modificada do método get_index_roulette_wheel_selection que usa np.ptp em vez de list_fitness.ptp()
        """
        if list_fitness is None:
            if not self.pop: # Handle empty population
                return 0 # Or raise error, or return a default valid index if pop_size > 0
            list_fitness = np.array([agent.target.fitness for agent in self.pop])
            if len(list_fitness) == 0: # If pop was not empty but somehow no fitness values
                return 0 
            
        fitness_range = np.max(list_fitness) - np.min(list_fitness) if len(list_fitness) > 0 else 0
        if fitness_range == 0:
            return self.generator.integers(0, len(list_fitness)) if len(list_fitness) > 0 else 0
            
        if self.problem.minmax == "min":
            norm_fitness = (np.max(list_fitness) - list_fitness) + self.EPSILON 
        else: # max
            norm_fitness = (list_fitness - np.min(list_fitness)) + self.EPSILON

        sum_fit = np.sum(norm_fitness)
        if sum_fit == 0:
             return self.generator.integers(0, len(list_fitness)) if len(list_fitness) > 0 else 0

        r = self.generator.random() * sum_fit
        current_sum = 0.0
        for idx, f_val in enumerate(norm_fitness):
            current_sum += f_val # Use current_sum to avoid floating point issues with repeated subtraction
            if current_sum >= r:
                return idx
        return len(list_fitness) - 1 if len(list_fitness) > 0 else 0 