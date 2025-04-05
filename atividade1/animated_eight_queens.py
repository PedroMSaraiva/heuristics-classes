import numpy as np
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Tuple
import time

class AnimatedEightQueens:
    def __init__(self):
        self.board_size = 8
        self.num_queens = 8
        self.states_history = []
        
    def generate_random_state(self) -> List[int]:
        """Gera um estado aleatório inicial com 8 rainhas."""
        return [random.randint(0, self.board_size - 1) for _ in range(self.num_queens)]

    def calculate_conflicts(self, state: List[int]) -> int:
        """Calcula o número de conflitos (ataques) entre rainhas no tabuleiro."""
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j]:  # Mesma linha
                    conflicts += 1
                offset = j - i
                if state[i] == state[j] - offset or state[i] == state[j] + offset:  # Diagonal
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
        """Implementa o algoritmo Hill Climbing com histórico de estados."""
        current_state = self.generate_random_state()
        current_conflicts = self.calculate_conflicts(current_state)
        iterations = 0
        conflicts_history = [current_conflicts]
        self.states_history = [(current_state.copy(), current_conflicts)]

        while iterations < max_iterations and current_conflicts > 0:
            neighbors = self.get_neighbors(current_state)
            found_better = False

            for neighbor in neighbors:
                neighbor_conflicts = self.calculate_conflicts(neighbor)
                if neighbor_conflicts < current_conflicts:
                    current_state = neighbor
                    current_conflicts = neighbor_conflicts
                    found_better = True
                    conflicts_history.append(current_conflicts)
                    self.states_history.append((current_state.copy(), current_conflicts))
                    break

            if not found_better:
                break

            iterations += 1

        return current_state, conflicts_history, iterations

    def simulated_annealing(self, initial_temp: float = 10.0, cooling_rate: float = 0.99,
                          max_iterations: int = 2000) -> Tuple[List[int], List[int], int]:
        """Implementa o algoritmo Simulated Annealing com histórico de estados."""
        current_state = self.generate_random_state()
        current_conflicts = self.calculate_conflicts(current_state)
        temperature = initial_temp
        iterations = 0
        conflicts_history = [current_conflicts]
        self.states_history = [(current_state.copy(), current_conflicts)]

        while iterations < max_iterations and current_conflicts > 0:
            i = random.randint(0, self.board_size - 1)
            j = random.randint(0, self.board_size - 1)
            new_state = current_state.copy()
            new_state[i] = j

            new_conflicts = self.calculate_conflicts(new_state)
            delta_e = new_conflicts - current_conflicts

            if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
                current_state = new_state
                current_conflicts = new_conflicts
                conflicts_history.append(current_conflicts)
                self.states_history.append((current_state.copy(), current_conflicts))

            temperature *= cooling_rate
            iterations += 1

        return current_state, conflicts_history, iterations

    def create_board_animation(self, save_video: bool = False, filename: str = "animation.gif"):
        """Cria uma animação do processo de busca."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        board = np.zeros((self.board_size, self.board_size))
        img = ax1.imshow(board, cmap='binary')
        ax1.grid(True)
        ax1.set_xticks(range(self.board_size))
        ax1.set_yticks(range(self.board_size))
        
        conflicts = [state[1] for state in self.states_history]
        line, = ax2.plot([], [], 'b-')
        ax2.set_xlim(0, len(conflicts))
        ax2.set_ylim(0, max(conflicts) + 1)
        ax2.set_xlabel('Iterações')
        ax2.set_ylabel('Número de Conflitos')
        ax2.grid(True)
        
        info_text = ax1.text(-0.1, -1.5, '', transform=ax1.transAxes)
        
        def init():
            img.set_array(np.zeros((self.board_size, self.board_size)))
            line.set_data([], [])
            return [img, line, info_text]
        
        def animate(frame):
            board = np.zeros((self.board_size, self.board_size))
            state = self.states_history[frame][0]
            for i in range(len(state)):
                board[state[i]][i] = 1
            img.set_array(board)
            
            conflicts = [state[1] for state in self.states_history[:frame+1]]
            line.set_data(range(len(conflicts)), conflicts)
            
            info_text.set_text(f'Iteração: {frame}\nConflitos: {self.states_history[frame][1]}')
            
            for txt in ax1.texts[1:]:  # Remove rainhas anteriores
                txt.remove()
            for i in range(len(state)):
                ax1.text(i, state[i], '♕', fontsize=20, ha='center', va='center')
            
            return [img, line, info_text]
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.states_history),
                           interval=100, blit=True)
        
        if save_video:
            writer = PillowWriter(fps=10)
            anim.save(filename, writer=writer)
        
        plt.show()

def main():
    # Inicializa o problema
    problem = AnimatedEightQueens()
    
    # Executa Hill Climbing e mostra animação
    print("\nExecutando Hill Climbing...")
    final_state, history, iterations = problem.hill_climbing()
    print(f"Solução encontrada: {final_state}")
    print(f"Número de conflitos: {problem.calculate_conflicts(final_state)}")
    print(f"Número de iterações: {iterations}")
    problem.create_board_animation(save_video=True, filename="hill_climbing.gif")
    
    # Executa Simulated Annealing e mostra animação
    print("\nExecutando Simulated Annealing...")
    final_state, history, iterations = problem.simulated_annealing()
    print(f"Solução encontrada: {final_state}")
    print(f"Número de conflitos: {problem.calculate_conflicts(final_state)}")
    print(f"Número de iterações: {iterations}")
    problem.create_board_animation(save_video=True, filename="simulated_annealing.gif")

if __name__ == "__main__":
    main() 