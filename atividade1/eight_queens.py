import numpy as np
import random
import math
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from typing import List, Tuple

class EightQueens:
    def __init__(self):
        self.board_size = 8
        self.num_queens = 8

    def generate_random_state(self) -> List[int]:
        """Gera um estado aleatório inicial com 8 rainhas."""
        return [random.randint(0, self.board_size - 1) for _ in range(self.num_queens)]

    def calculate_conflicts(self, state: List[int]) -> int:
        """Calcula o número de conflitos (ataques) entre rainhas no tabuleiro."""
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j]:
                    conflicts += 1
                offset = j - i
                if state[i] == state[j] - offset or state[i] == state[j] + offset: 
                    conflicts += 1
        return conflicts

    def get_neighbors(self, state: List[int]) -> List[List[int]]:
        """Gera todos os estados vizinhos possíveis movendo uma rainha por vez."""
        neighbors = []
        for i in range(len(state)):
            for j in range(self.board_size):
                if state[i] != j:
                    neighbor = state.copy()
                    neighbor[i] = j
                    neighbors.append(neighbor)
        return neighbors

    def hill_climbing(self, max_iterations: int = 2000) -> Tuple[List[int], List[int], int]:
        """Implementa o algoritmo Hill Climbing."""
        current_state = self.generate_random_state()
        current_conflicts = self.calculate_conflicts(current_state)
        iterations = 0
        conflicts_history = [current_conflicts]

        while iterations < max_iterations and current_conflicts > 0:
            neighbors = self.get_neighbors(current_state)
            found_better = False

            # Encontra o vizinho com menor número de conflitos
            for neighbor in neighbors:
                neighbor_conflicts = self.calculate_conflicts(neighbor)
                if neighbor_conflicts < current_conflicts:
                    current_state = neighbor
                    current_conflicts = neighbor_conflicts
                    found_better = True
                    conflicts_history.append(current_conflicts)
                    break

            if not found_better:  # Atingiu um mínimo local
                break

            iterations += 1

        return current_state, conflicts_history, iterations

    def simulated_annealing(self, initial_temp: float = 10.0, cooling_rate: float = 0.99, 
                          max_iterations: int = 2000) -> Tuple[List[int], List[int], int]:
        """Implementa o algoritmo Simulated Annealing."""
        current_state = self.generate_random_state()
        current_conflicts = self.calculate_conflicts(current_state)
        temperature = initial_temp
        iterations = 0
        conflicts_history = [current_conflicts]

        while iterations < max_iterations and current_conflicts > 0:
            i = random.randint(0, self.board_size - 1)
            j = random.randint(0, self.board_size - 1)
            new_state = current_state.copy()
            new_state[i] = j

            new_conflicts = self.calculate_conflicts(new_state)
            delta_e = new_conflicts - current_conflicts

            # Aceita o novo estado se for melhor ou com uma probabilidade baseada na temperatura
            if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
                current_state = new_state
                current_conflicts = new_conflicts
                conflicts_history.append(current_conflicts)

            temperature *= cooling_rate
            iterations += 1

        return current_state, conflicts_history, iterations

    def plot_board(self, state: List[int], title: str = ""):
        """Plota o tabuleiro de xadrez com as rainhas."""
        board = np.zeros((self.board_size, self.board_size))
        for i in range(len(state)):
            board[state[i]][i] = 1

        plt.figure(figsize=(8, 8))
        plt.imshow(board, cmap='binary')
        plt.grid(True)
        plt.title(title)
        for i in range(len(state)):
            plt.text(i, state[i], '♕', fontsize=20, ha='center', va='center')
        plt.xticks(range(self.board_size))
        plt.yticks(range(self.board_size))

    def plot_conflicts_history(self, hill_climbing_history: List[int], 
                             simulated_annealing_history: List[int]):
        """Plota o histórico de conflitos para ambos os algoritmos."""
        plt.figure(figsize=(10, 6))
        plt.plot(hill_climbing_history, label='Hill Climbing', marker='o')
        plt.plot(simulated_annealing_history, label='Simulated Annealing', marker='o')
        plt.xlabel('Iterações')
        plt.ylabel('Número de Conflitos')
        plt.title('Comparação da Convergência dos Algoritmos')
        plt.legend()
        plt.grid(True)

def main():
    problem = EightQueens()
    
    hc_solution, hc_history, hc_iterations = problem.hill_climbing()
    hc_conflicts = problem.calculate_conflicts(hc_solution)
    print(f"\nHill Climbing:")
    print(f"Solução encontrada: {hc_solution}")
    print(f"Número de conflitos: {hc_conflicts}")
    print(f"Número de iterações: {hc_iterations}")
    
    sa_solution, sa_history, sa_iterations = problem.simulated_annealing()
    sa_conflicts = problem.calculate_conflicts(sa_solution)
    print(f"\nSimulated Annealing:")
    print(f"Solução encontrada: {sa_solution}")
    print(f"Número de conflitos: {sa_conflicts}")
    print(f"Número de iterações: {sa_iterations}")
    
    # Plota os resultados
    problem.plot_board(hc_solution, "Solução Hill Climbing")
    problem.plot_board(sa_solution, "Solução Simulated Annealing")
    problem.plot_conflicts_history(hc_history, sa_history)
    plt.show()

if __name__ == "__main__":
    main() 