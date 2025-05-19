import numpy as np
import matplotlib.pyplot as plt
from mealpy.evolutionary_based import GA
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.ACOR import OriginalACOR
from custom_aco import CustomACOR
from mealpy import FloatVar
from numba import njit, prange, float64, int64, set_num_threads, cuda
import time
import os
import multiprocessing
import traceback
import logging
import datetime
import platform
import threading
import concurrent.futures
import pandas as pd
import seaborn as sns
from scipy import stats
from PIL import Image

# Importar nossa versão personalizada do ACO
try:
    from custom_aco import CustomACOR
    CUSTOM_ACO_AVAILABLE = True
    print("CustomACOR importado com sucesso!")
except ImportError as e:
    CUSTOM_ACO_AVAILABLE = False
    print(f"Erro ao importar CustomACOR: {e}")

# Tentar importar o suporte para AMD GPU
try:
    from amd_gpu_support import check_amd_gpu_support, get_amd_accelerator
    AMD_MODULE_AVAILABLE = True
except ImportError:
    AMD_MODULE_AVAILABLE = False

# Configurar logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"gpu_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Compatibilidade com NumPy 2.0 para o método ptp() que foi removido
# Esta função é necessária para o algoritmo ACO
def add_numpy_compatibility():
    logger.info("Adicionando compatibilidade com NumPy 2.0 para o método ptp()")
    try:
        import numpy as np
        np_version = np.__version__
        logger.info(f"Versão do NumPy: {np_version}")
        
        # Verificar se estamos na versão 2.0 ou superior
        if int(np_version.split('.')[0]) >= 2:
            logger.info("Detectado NumPy 2.0+, adicionando compatibilidade para ptp()")
            
            # Adicionar o método ptp() à classe ndarray
            if not hasattr(np.ndarray, 'ptp'):
                def _ptp(self, axis=None, out=None, keepdims=False):
                    return np.ptp(self, axis=axis, out=out, keepdims=keepdims)
                
                np.ndarray.ptp = _ptp
                logger.info("Método ptp() adicionado com sucesso à classe ndarray")
            else:
                logger.info("Método ptp() já existe na classe ndarray")
    except Exception as e:
        logger.error(f"Erro ao adicionar compatibilidade com NumPy: {e}")

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimization")

logger.info("Iniciando processo de otimização com GPU")
logger.info(f"Número de CPUs disponíveis: {multiprocessing.cpu_count()}")

# Adicionar compatibilidade com NumPy 2.0 para ACO
add_numpy_compatibility()

# Detectar disponibilidade de GPU
has_cuda = False
has_rocm = False
amd_accelerator = None

try:
    if cuda.is_available():
        has_cuda = True
        logger.info(f"CUDA disponível: {cuda.get_current_device().name}")
        logger.info(f"Número de GPUs CUDA: {cuda.get_num_devices()}")
except Exception as e:
    logger.warning(f"CUDA não está disponível: {e}")

# Verificar AMD ROCm (para GPUs AMD)
try:
    # Verificação simples para AMD ROCm
    if platform.system() == "Linux" and os.path.exists("/opt/rocm"):
        has_rocm = True
        logger.info("ROCm detectado para GPU AMD")
    elif platform.system() == "Windows" and "AMD" in os.popen("wmic path win32_VideoController get name").read():
        has_rocm = True
        logger.info("GPU AMD detectada no Windows")
        
    # Verificar suporte através do módulo AMD
    if AMD_MODULE_AVAILABLE:
        amd_available, device_name = check_amd_gpu_support()
        if amd_available:
            logger.info(f"Suporte PyOpenCL para GPU AMD disponível: {device_name}")
            has_rocm = True
except Exception as e:
    logger.warning(f"Erro ao verificar ROCm: {e}")

# Configurações globais
problem_size = 10  # Dimensão do problema
c1 = 2.05  # Parâmetro PSO
c2 = 2.05  # Parâmetro PSO
w_min = 0.4  # Parâmetro PSO
w_max = 0.9  # Parâmetro PSO

# Configurações para otimização máxima do Numba
os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())
set_num_threads(multiprocessing.cpu_count())
logger.info(f"NUMBA_NUM_THREADS configurado para: {os.environ.get('NUMBA_NUM_THREADS')}")

# Constantes pré-calculadas para F6 - evita recálculos repetidos
A = 0.5
B = 3
KMAX = 20

# Pré-cálculo das potências de a e b - melhora o desempenho evitando recálculos
@njit(fastmath=True, cache=True)
def precalculate_powers():
    a_powers = np.zeros(KMAX + 1, dtype=np.float64)
    b_powers = np.zeros(KMAX + 1, dtype=np.float64)
    for k in range(KMAX + 1):
        a_powers[k] = A ** k
        b_powers[k] = B ** k
    return a_powers, b_powers

# Pré-calcular os valores das potências
logger.info("Pré-calculando potências para F6")
A_POWERS, B_POWERS = precalculate_powers()
logger.info("Potências pré-calculadas")

# Inicializar acelerador AMD se disponível
if AMD_MODULE_AVAILABLE and has_rocm:
    logger.info("Inicializando acelerador para GPU AMD")
    amd_accelerator = get_amd_accelerator(A_POWERS, B_POWERS)
    if amd_accelerator.available:
        logger.info("Acelerador AMD inicializado com sucesso")
    else:
        logger.warning("Falha ao inicializar acelerador AMD")
        amd_accelerator = None

# Constante para F6 - parte que não depende de x
@njit(fastmath=True, cache=True)
def calculate_f6_constant_term(dim):
    constant_sum = 0.0
    for k in range(KMAX + 1):
        constant_sum += A_POWERS[k] * np.cos(2 * np.pi * B_POWERS[k] * 0.5)
    return dim * constant_sum

# Versão para CPU da F1
@njit("float64(float64[:])", fastmath=True, parallel=True, cache=True)
def evaluate_f1_cpu(x):
    """Versão altamente otimizada da High Conditioned Elliptic Function (F1)
    
    f₁(x) = Σᵢ₌₁ᴰ (10⁶)^((i-1)/(D-1)) * xᵢ²
    """
    D = len(x)
    result = 0.0
    for i in prange(D):
        exponent = i / (D - 1) if D > 1 else 0
        coefficient = (10**6) ** exponent
        result += coefficient * (x[i] * x[i])  # Mais rápido que x[i]**2
    return result

# Versão para CPU da F6
@njit("float64(float64[:])", fastmath=True, parallel=True, cache=True)
def evaluate_f6_cpu(x):
    """Versão altamente otimizada da Weierstrass Function (F6)
    
    f₆(x) = Σᵢ₌₁ᴰ (Σₖ₌₀ᵏᵐᵃˣ [aᵏ cos(2πbᵏ(xᵢ+0.5))]) - D·Σₖ₌₀ᵏᵐᵃˣ[aᵏ cos(2πbᵏ·0.5)]
    """
    D = len(x)
    # Usar o termo pré-calculado
    constant_term = calculate_f6_constant_term(D)
    
    # Calcular o primeiro somatório (que depende de x) em paralelo
    variable_sum = 0.0
    for i in prange(D):
        inner_sum = 0.0
        for k in range(KMAX + 1):
            inner_sum += A_POWERS[k] * np.cos(2 * np.pi * B_POWERS[k] * (x[i] + 0.5))
        variable_sum += inner_sum
    
    # Retornar o resultado final
    return variable_sum - constant_term

# Versão para GPU da F1 (CUDA)
@cuda.jit(device=True)
def evaluate_f1_element_gpu(x, i, D):
    exponent = i / (D - 1) if D > 1 else 0
    coefficient = (10**6) ** exponent
    return coefficient * (x[i] * x[i])

@cuda.jit
def evaluate_f1_kernel(x_array, results, n_vectors, vector_size):
    idx = cuda.grid(1)
    if idx < n_vectors:
        result = 0.0
        for i in range(vector_size):
            result += evaluate_f1_element_gpu(x_array[idx], i, vector_size)
        results[idx] = result

# Versão para GPU da F6 (CUDA)
@cuda.jit(device=True)
def evaluate_f6_element_gpu(x, i, a_powers, b_powers, kmax):
    inner_sum = 0.0
    for k in range(kmax + 1):
        inner_sum += a_powers[k] * cuda.libdevice.cos(2 * np.pi * b_powers[k] * (x[i] + 0.5))
    return inner_sum

@cuda.jit
def evaluate_f6_kernel(x_array, results, n_vectors, vector_size, a_powers, b_powers, kmax, constant_term):
    idx = cuda.grid(1)
    if idx < n_vectors:
        variable_sum = 0.0
        for i in range(vector_size):
            variable_sum += evaluate_f6_element_gpu(x_array[idx], i, a_powers, b_powers, kmax)
        results[idx] = variable_sum - constant_term

# Função para avaliar um lote de vetores usando GPU
def batch_evaluate_gpu(vectors, func_name="f1"):
    # Tentar usar AMD GPU primeiro se disponível
    if amd_accelerator is not None and amd_accelerator.available:
        if func_name == "f1":
            result = amd_accelerator.evaluate_f1_batch(vectors)
            if result is not None:
                return result
        elif func_name == "f6":
            constant_term = calculate_f6_constant_term(vectors.shape[1])
            result = amd_accelerator.evaluate_f6_batch(vectors, constant_term)
            if result is not None:
                return result
    
    # Fallback para CUDA se disponível
    if has_cuda:
        try:
            # Preparar dados para GPU
            n_vectors = len(vectors)
            vector_size = len(vectors[0])
            x_device = cuda.to_device(vectors)
            results_device = cuda.device_array(n_vectors, dtype=np.float64)
            
            # Configurar grade de threads
            threads_per_block = 256
            blocks_per_grid = (n_vectors + threads_per_block - 1) // threads_per_block
            
            # Executar kernel apropriado
            if func_name == "f1":
                evaluate_f1_kernel[blocks_per_grid, threads_per_block](x_device, results_device, n_vectors, vector_size)
            else:  # f6
                a_powers_device = cuda.to_device(A_POWERS)
                b_powers_device = cuda.to_device(B_POWERS)
                constant_term = calculate_f6_constant_term(vector_size)
                evaluate_f6_kernel[blocks_per_grid, threads_per_block](
                    x_device, results_device, n_vectors, vector_size, 
                    a_powers_device, b_powers_device, KMAX, constant_term
                )
            
            # Transferir resultados de volta para host
            results = results_device.copy_to_host()
            return results
        except Exception as e:
            logger.error(f"Erro ao usar CUDA: {e}")
    
    # Fallback para CPU
    if func_name == "f1":
        return np.array([evaluate_f1_cpu(v) for v in vectors])
    else:  # f6
        return np.array([evaluate_f6_cpu(v) for v in vectors])

# Contador de avaliações para monitoramento - usando threading.Lock para evitar race conditions
eval_counter = {'f1': 0, 'f6': 0}
eval_counter_lock = threading.Lock()

# Wrapper para F1 que usa GPU quando disponível
def optimized_f1_wrapper(x):
    global eval_counter
    with eval_counter_lock:
        eval_counter['f1'] += 1
        if eval_counter['f1'] % 1000 == 0:
            logger.debug(f"F1 avaliações: {eval_counter['f1']}")
    
    # Para avaliações individuais, usar CPU é mais eficiente (evita overhead de transferência)
    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        return evaluate_f1_cpu(x)
    elif isinstance(x, list):
        return evaluate_f1_cpu(np.array(x, dtype=np.float64))
    else:
        logger.warning(f"Tipo de entrada não suportado para F1: {type(x)}")
        return evaluate_f1_cpu(np.array(x, dtype=np.float64))

# Wrapper para F6 que usa GPU quando disponível
def optimized_f6_wrapper(x):
    global eval_counter
    with eval_counter_lock:
        eval_counter['f6'] += 1
        if eval_counter['f6'] % 1000 == 0:
            logger.debug(f"F6 avaliações: {eval_counter['f6']}")
    
    # Para avaliações individuais, usar CPU é mais eficiente
    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        return evaluate_f6_cpu(x)
    elif isinstance(x, list):
        return evaluate_f6_cpu(np.array(x, dtype=np.float64))
    else:
        logger.warning(f"Tipo de entrada não suportado para F6: {type(x)}")
        return evaluate_f6_cpu(np.array(x, dtype=np.float64))

# Função para executar um único treinamento
def train_model_optimized(model="PSO", problem="f6", config_idx=0, run_id=0):
    """Versão extremamente otimizada da função train_model"""
    
    process_id = os.getpid()
    thread_id = threading.get_ident()
    logger.info(f"Iniciando treinamento: modelo={model}, problema={problem}, config={config_idx}, run={run_id}, PID={process_id}, Thread={thread_id}")
    
    start_time = time.time()
    
    # Definir apenas uma configuração por execução para maximizar paralelismo
    configs = [
        {"name": "Config 1", "epoch": 500, "pop_size": 20},
        {"name": "Config 2", "epoch": 1000, "pop_size": 50},
        {"name": "Config 3", "epoch": 2000, "pop_size": 100}
    ]
    
    config = configs[config_idx]
    logger.info(f"Executando {config['name']}: Epoch={config['epoch']}, Pop_size={config['pop_size']}, PID={process_id}, Thread={thread_id}")
    
    # Usar as funções de avaliação Numba-otimizadas
    if problem == "f1":
        problem_dict = {"obj_func": optimized_f1_wrapper}
    elif problem == "f6":
        problem_dict = {"obj_func": optimized_f6_wrapper}
    else:
        logger.error(f"Problema desconhecido: {problem}")
        return None
    
    metadata_dict = {
        "bounds": FloatVar(lb=[-100] * problem_size, ub=[100] * problem_size),
        "minmax": "min",
        "name": f"Config-{config['name']}",
        "verbose": False,  # Reduzir saída para melhorar desempenho
        "log_to": "console", 
        "save_population": True,  # Necessário para visualizações
    }
    
    main_problem = {}
    main_problem.update(problem_dict)
    main_problem.update(metadata_dict)
    
    algorithm = None
    try:
        if model == "GA":
            logger.info(f"Criando algoritmo GA, PID={process_id}, Thread={thread_id}")
            algorithm = GA.BaseGA(
                epoch=config['epoch'],
                pop_size=config['pop_size'],
                pc=0.95,
                pm=0.025
            )
        elif model == "PSO":
            logger.info(f"Criando algoritmo PSO, PID={process_id}, Thread={thread_id}")
            algorithm = OriginalPSO(
                epoch=config["epoch"],
                pop_size=config["pop_size"],
                c1=c1,
                c2=c2,
                w_min=w_min,
                w_max=w_max
             )
        elif model == "ACO":
            logger.info(f"Criando algoritmo ACO, PID={process_id}, Thread={thread_id}")
            try:
                # Garantir que a compatibilidade com NumPy 2.0 esteja aplicada
                if not hasattr(np.ndarray, 'ptp'):
                    logger.info(f"Adicionando compatibilidade para ptp, PID={process_id}, Thread={thread_id}")
                    def _ptp(self, axis=None, out=None, keepdims=False):
                        return np.ptp(self, axis=axis, out=out, keepdims=keepdims)
                    np.ndarray.ptp = _ptp
                
                algorithm = CustomACOR(
                    epoch=config["epoch"],
                    pop_size=config["pop_size"],   
                )
            except Exception as e:
                logger.error(f"Erro ao criar ACO: {e}, PID={process_id}, Thread={thread_id}")
                if "ptp" in str(e):
                    # Contornar problema do ptp no NumPy 2.0
                    if not hasattr(np.ndarray, 'ptp'):
                        logger.info(f"Adicionando compatibilidade para ptp, PID={process_id}, Thread={thread_id}")
                        def _ptp(self, axis=None, out=None, keepdims=False):
                            return np.ptp(self, axis=axis, out=out, keepdims=keepdims)
                        np.ndarray.ptp = _ptp
                    
                    algorithm = CustomACOR(
                        epoch=config["epoch"],
                        pop_size=config["pop_size"],   
                    )
                else:
                    return None
        else:
            logger.error(f"Modelo desconhecido: {model}, PID={process_id}, Thread={thread_id}")
            return None
    except Exception as e:
        logger.error(f"Erro ao criar algoritmo: {e}, PID={process_id}, Thread={thread_id}")
        traceback.print_exc()
        return None

    # Medir o tempo da execução
    logger.info(f"Iniciando resolução do problema, PID={process_id}, Thread={thread_id}")
    solve_start = time.time()
    
    try:
        # Configurar o dicionário de terminação para early stopping
        term_dict = {
            "max_early_stop": 100  # Mantendo o valor anterior, ajuste conforme necessário
        }
        
        best_agent = algorithm.solve(main_problem, termination=term_dict)
            
        solve_time = time.time() - solve_start
        
        best_position = best_agent.solution
        best_fitness = best_agent.target.fitness
        
        # Criar diretórios para resultados
        main_path = f"results/{model}/{problem}/Config{config_idx + 1}"
        os.makedirs(main_path, exist_ok=True)
        
        # Salvar gráficos da execução
        base_title = f"{model} - {problem} - {config['name']} - Run {run_id}"
        try:
            algorithm.history.save_global_objectives_chart(filename=f"{main_path}/goc_{config_idx}_{run_id}", title=f"{base_title} - Global Objectives Chart", verbose=False)
            algorithm.history.save_local_objectives_chart(filename=f"{main_path}/loc_{config_idx}_{run_id}", title=f"{base_title} - Local Objectives Chart", verbose=False)
            algorithm.history.save_global_best_fitness_chart(filename=f"{main_path}/gbfc_{config_idx}_{run_id}", title=f"{base_title} - Global Best Fitness Chart", verbose=False)
            algorithm.history.save_local_best_fitness_chart(filename=f"{main_path}/lbfc_{config_idx}_{run_id}", title=f"{base_title} - Local Best Fitness Chart", verbose=False)
            algorithm.history.save_runtime_chart(filename=f"{main_path}/rtc_{config_idx}_{run_id}", title=f"{base_title} - Runtime Chart", verbose=False)
            algorithm.history.save_exploration_exploitation_chart(filename=f"{main_path}/eec_{config_idx}_{run_id}", title=f"{base_title} - Exploration Exploitation Chart", verbose=False)
            algorithm.history.save_diversity_chart(filename=f"{main_path}/dc_{config_idx}_{run_id}", title=f"{base_title} - Diversity Chart", verbose=False)
            
            # Salvar trajetória apenas se houver agentes suficientes
            if algorithm.pop_size > 7:
                try:
                    algorithm.history.save_trajectory_chart(
                        list_agent_idx=[3, 5, 6, 7],
                        selected_dimensions=[0, 1],  # Tenta com base 0
                        filename=f"{main_path}/tc_{config_idx}_{run_id}",
                        title=f"{base_title} - Trajectory Chart",
                        verbose=False
                    )
                except Exception as e:
                    if "the index of selected dimensions should be in range" in str(e):
                        # Tenta novamente com base 1
                        try:
                            algorithm.history.save_trajectory_chart(
                                list_agent_idx=[3, 5, 6, 7],
                                selected_dimensions=[1, 2],  # Corrige para base 1
                                filename=f"{main_path}/tc_{config_idx}_{run_id}",
                                title=f"{base_title} - Trajectory Chart",
                                verbose=False
                            )
                            logger.info("save_trajectory_chart: Corrigido para índices base 1.")
                        except Exception as e2:
                            logger.warning(f"Erro ao salvar trajectory chart mesmo após correção: {e2}")
                    else:
                        logger.warning(f"Erro ao salvar trajectory chart: {e}")
        except Exception as e:
            logger.warning(f"Erro ao salvar alguns gráficos: {e}, PID={process_id}, Thread={thread_id}")
        
        # Extrair históricos para visualização
        fitness_history = []
        exploration_history = []
        exploitation_history = []
        positions_history = []
        
        try:
            # Extrair histórico de fitness global - Adicionando verificações robustas
            if hasattr(algorithm.history, 'list_global_best_fit'):
                fitness_history = algorithm.history.list_global_best_fit
                # Verificar se o histórico de fitness tem valores reais
                if fitness_history and all(val == 0.0 for val in fitness_history):
                    logger.warning(f"ATENÇÃO: Todos os valores de fitness no histórico são 0.0, indicando possível problema!")
            
            # Debug para verificar o fitness 
            logger.info(f"Fitness history: {fitness_history[:5]}... (total: {len(fitness_history)} itens)")
            
            # Extrair históricos de exploration/exploitation
            if hasattr(algorithm.history, 'list_exploration') and hasattr(algorithm.history, 'list_exploitation'):
                exploration_history = algorithm.history.list_exploration
                exploitation_history = algorithm.history.list_exploitation
            
            # Tentar extrair histórico de posições (para animação) - Versão mais robusta
            if hasattr(algorithm.history, 'list_population'):
                positions = algorithm.history.list_population
                # Converter para formato adequado para animação
                for idx, pop in enumerate(positions):
                    pos_array = []
                    for agent in pop:
                        if hasattr(agent, 'solution'):
                            pos_array.append(agent.solution)
            
                    if pos_array:
                        positions_history.append(pos_array)
                    else:
                        logger.warning(f"População na época {idx} não tem soluções válidas")
            
            # Verificar se conseguimos extrair posições
            if not positions_history:
                logger.warning("Não foi possível extrair histórico de posições!")
            else:
                logger.info(f"Histórico de posições extraído: {len(positions_history)} épocas, {len(positions_history[0])} agentes por época")
        except Exception as e:
            logger.warning(f"Erro ao extrair históricos: {e}, PID={process_id}, Thread={thread_id}")
        
        logger.info(f"Problema resolvido em {solve_time:.2f} segundos, PID={process_id}, Thread={thread_id}")
        logger.info(f"Melhor fitness: {best_fitness}, PID={process_id}, Thread={thread_id}")
    except Exception as e:
        logger.error(f"Erro durante resolução: {e}, PID={process_id}, Thread={thread_id}")
        traceback.print_exc()
        return None

    result = {
        "config": config,
        "best_fitness": best_fitness,
        "best_position": best_position,
        "execution_time": solve_time,
        "run_id": run_id,
        "process_id": process_id,
        "thread_id": thread_id,
        "config_idx": config_idx,
        "model": model,
        "problem": problem,
        "fitness_history": fitness_history,
        "exploration_history": exploration_history,
        "exploitation_history": exploitation_history,
        "positions_history": positions_history
    }
    
    total_time = time.time() - start_time
    logger.info(f"Treinamento concluído em {total_time:.2f} segundos, PID={process_id}, Thread={thread_id}")
    
    return result

def warmup_numba():
    """Pré-compila as funções Numba para evitar o overhead de compilação inicial"""
    logger.info("Pré-compilando funções Numba...")
    
    # Forçar a criação de direcionamentos para vários tamanhos de array
    for size in [10, 20, 50]:
        test_array = np.random.random(size)
        start = time.time()
        _ = evaluate_f1_cpu(test_array)
        logger.info(f"Pré-compilação evaluate_f1_cpu para tamanho {size}: {time.time() - start:.4f}s")
        
        start = time.time()
        _ = evaluate_f6_cpu(test_array)
        logger.info(f"Pré-compilação evaluate_f6_cpu para tamanho {size}: {time.time() - start:.4f}s")
    
    # Pré-compilar funções GPU se disponível
    if has_cuda:
        logger.info("Pré-compilando funções CUDA...")
        test_vectors = np.random.random((10, problem_size)).astype(np.float64)
        
        start = time.time()
        _ = batch_evaluate_gpu(test_vectors, "f1")
        logger.info(f"Pré-compilação batch_evaluate_gpu para F1: {time.time() - start:.4f}s")
        
        start = time.time()
        _ = batch_evaluate_gpu(test_vectors, "f6")
        logger.info(f"Pré-compilação batch_evaluate_gpu para F6: {time.time() - start:.4f}s")
    
    # Pré-calcular as constantes para F6
    for dim in [10, 20, 50]:
        start = time.time()
        _ = calculate_f6_constant_term(dim)
        logger.info(f"Pré-compilação constant_term para dimensão {dim}: {time.time() - start:.4f}s")
    
    logger.info("Pré-compilação concluída!")

# Versão paralela usando numba diretamente em vez de multiprocessing
@njit(parallel=True)
def parallel_sum(n):
    """Função simples para testar paralelização"""
    acc = 0
    # Usar prange para paralelização
    for i in prange(n):
        acc += i
    return acc

def worker_initializer():
    """Inicializador para processos worker"""
    # Configurar o processo para usar o máximo de CPU
    process_id = os.getpid()
    logger.info(f"Inicializando worker com PID={process_id}")
    
    # Adicionar compatibilidade com NumPy 2.0 para ACO em cada worker
    add_numpy_compatibility()

# Função para gerar todas as combinações de parâmetros para execução em paralelo
def generate_all_tasks(models, problems, num_runs):
    all_tasks = []
    for model in models:
        for problem in problems:
            for config_idx in range(3):  # 3 configurações
                for run_id in range(num_runs // 3):  # Distribuir execuções entre configurações
                    all_tasks.append((model, problem, config_idx, run_id))
    return all_tasks

def run_analysis_optimized(models=None, problems=None, num_runs=30):
    """Função principal para executar análise completa com máxima paralelização"""
    logger.info("Iniciando análise otimizada com paralelismo total")
    
    if models is None:
        models = ["PSO", "GA"]  # Usando PSO e GA que são compatíveis com NumPy 2.0
    
    if problems is None:
        problems = ["f1", "f6"]
    
    logger.info(f"Modelos: {models}")
    logger.info(f"Problemas: {problems}")
    logger.info(f"Número de execuções: {num_runs}")
    
    # Pré-compilar funções Numba antes de iniciar
    warmup_numba()
    
    # Testar paralelização do Numba
    logger.info("Testando paralelização do Numba...")
    start_time = time.time()
    _ = parallel_sum(10000000)  # Deve ser rápido se a paralelização estiver funcionando
    logger.info(f"Teste de paralelização concluído em {time.time() - start_time:.4f} segundos")
    
    # Gerar todas as tarefas para execução paralela
    all_tasks = generate_all_tasks(models, problems, num_runs)
    logger.info(f"Total de tarefas geradas: {len(all_tasks)}")
    
    # Resultado compartilhado para armazenar todos os resultados
    all_results = []
    results_lock = threading.Lock()
    
    # Contador para monitorar progresso
    tasks_completed = 0
    tasks_total = len(all_tasks)
    tasks_completed_lock = threading.Lock()
    
    def process_result(future):
        nonlocal tasks_completed
        try:
            result = future.result()
            if result:
                with results_lock:
                    all_results.append(result)
            
            with tasks_completed_lock:
                tasks_completed += 1
                if tasks_completed % 5 == 0 or tasks_completed == tasks_total:
                    logger.info(f"Progresso: {tasks_completed}/{tasks_total} tarefas concluídas ({tasks_completed/tasks_total*100:.1f}%)")
        except Exception as e:
            logger.error(f"Erro ao processar resultado: {e}")
            traceback.print_exc()
    
    # Usar ThreadPoolExecutor para gerenciar as submissões ao ProcessPoolExecutor
    # Isso permite melhor controle sobre o progresso e evita sobrecarregar a memória
    start_time = time.time()
    
    try:
        # Usar contexto 'spawn' para maior compatibilidade
        ctx = multiprocessing.get_context('spawn')
        
        # Determinar o número ideal de workers para o pool
        # Usar 90% dos cores disponíveis para maximizar paralelismo
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.9))
        logger.info(f"Usando {num_workers} workers para {tasks_total} tarefas")
        
        # Executar todas as tarefas em paralelo
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=ctx,
            initializer=worker_initializer
        ) as executor:
            # Submeter todas as tarefas de uma vez
            futures = [executor.submit(train_model_optimized, *task) for task in all_tasks]
            
            # Adicionar callback para processar resultados à medida que ficam prontos
            for future in futures:
                future.add_done_callback(process_result)
            
            # Aguardar a conclusão de todas as tarefas
            concurrent.futures.wait(futures)
            
    except Exception as e:
        logger.error(f"Erro durante execução paralela: {e}")
        traceback.print_exc()
    
    total_time = time.time() - start_time
    logger.info(f"Todas as tarefas concluídas em {total_time:.2f}s")
    logger.info(f"Resultados válidos: {len(all_results)}/{tasks_total}")
    
    # Organizar resultados por modelo e problema
    final_results = {}
    for model in models:
        final_results[model] = {}
        for problem in problems:
            # Filtrar resultados para este modelo e problema
            filtered_results = [r for r in all_results if r["model"] == model and r["problem"] == problem]
            
            if filtered_results:
                best_fitness_values = [result["best_fitness"] for result in filtered_results]
                execution_times = [result["execution_time"] for result in filtered_results]
                
                # Guardar estatísticas
                final_results[model][problem] = {
                    "mean_fitness": np.mean(best_fitness_values),
                    "std_fitness": np.std(best_fitness_values),
                    "min_fitness": np.min(best_fitness_values),
                    "max_fitness": np.max(best_fitness_values),
                    "mean_time": np.mean(execution_times),
                    "std_time": np.std(execution_times),
                }
                
                # Imprimir estatísticas
                stats = final_results[model][problem]
                logger.info(f"Estatísticas para {model} no problema {problem}:")
                logger.info(f"Fitness médio: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
                logger.info(f"Tempo médio de execução: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
                logger.info(f"Melhor fitness: {stats['min_fitness']:.4f}")
                
                # Gerar visualizações para este modelo e problema
                try:
                    generate_visualizations(filtered_results, model, problem)
                    create_animation_static(filtered_results, model, problem)
                except Exception as e:
                    logger.error(f"Erro ao gerar visualizações para {model} no problema {problem}: {e}")
            else:
                logger.warning(f"Não há resultados para {model} no problema {problem}")
    
    logger.info("Análise completa!")
    return final_results

def plot_results(results, algo_name="PSO", function_name="f6"):
    """
    Plota gráficos de comparação de resultados entre diferentes configurações
    """
    logger.info(f"Gerando gráficos para {algo_name} no problema {function_name}")
    
    # Criar diretórios necessários
    main_path = f"results/{algo_name}/{function_name}"
    os.makedirs(main_path, exist_ok=True)
    
    # Agrupar resultados por configuração
    config_results = {}
    for result in results:
        config_name = result.get('config', {}).get('name', 'Unknown')
        if config_name not in config_results:
            config_results[config_name] = []
        config_results[config_name].append(result)
    
    # Para cada configuração, calcular médias
    avg_results = []
    for config_name, config_data in config_results.items():
        # Extrair dados relevantes
        fitness_histories = [r.get('fitness_history', []) for r in config_data if 'fitness_history' in r]
        exploration_histories = [r.get('exploration_history', []) for r in config_data if 'exploration_history' in r]
        exploitation_histories = [r.get('exploitation_history', []) for r in config_data if 'exploitation_history' in r]
        
        # Se não temos históricos, continuar para a próxima configuração
        if not fitness_histories:
            logger.warning(f"Sem dados de histórico para {config_name}")
            continue
        
        # Calcular médias
        max_length = max(len(hist) for hist in fitness_histories)
        avg_fitness = np.zeros(max_length)
        count_fitness = np.zeros(max_length)
        
        for hist in fitness_histories:
            for i, value in enumerate(hist):
                avg_fitness[i] += value
                count_fitness[i] += 1
        
        # Evitar divisão por zero
        count_fitness[count_fitness == 0] = 1
        avg_fitness = avg_fitness / count_fitness
        
        # Fazer o mesmo para exploration/exploitation se disponível
        avg_exploration = None
        avg_exploitation = None
        
        if exploration_histories and exploitation_histories:
            max_length_exp = max(len(hist) for hist in exploration_histories)
            avg_exploration = np.zeros(max_length_exp)
            count_exploration = np.zeros(max_length_exp)
            
            for hist in exploration_histories:
                for i, value in enumerate(hist):
                    avg_exploration[i] += value
                    count_exploration[i] += 1
            
            count_exploration[count_exploration == 0] = 1
            avg_exploration = avg_exploration / count_exploration
            
            max_length_expl = max(len(hist) for hist in exploitation_histories)
            avg_exploitation = np.zeros(max_length_expl)
            count_exploitation = np.zeros(max_length_expl)
            
            for hist in exploitation_histories:
                for i, value in enumerate(hist):
                    avg_exploitation[i] += value
                    count_exploitation[i] += 1
            
            count_exploitation[count_exploitation == 0] = 1
            avg_exploitation = avg_exploitation / count_exploitation
        
        # Calcular estatísticas de fitness
        best_fitness_values = [r.get('best_fitness', float('inf')) for r in config_data]
        avg_best_fitness = np.mean(best_fitness_values)
        std_best_fitness = np.std(best_fitness_values)
        
        # Adicionar resultado médio
        avg_result = {
            'config': config_data[0]['config'],
            'fitness_history': avg_fitness,
            'exploration_history': avg_exploration,
            'exploitation_history': avg_exploitation,
            'best_fitness': avg_best_fitness,
            'best_fitness_std': std_best_fitness
        }
        
        avg_results.append(avg_result)
    
    # Gerar gráfico
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for result in avg_results:
        plt.plot(result["fitness_history"],
                label=f"{result['config']['name']} (Best: {result['best_fitness']:.4f} ± {result['best_fitness_std']:.4f})")
    
    plt.title(f'Evolução do Fitness Global - {algo_name} - {function_name}')
    plt.xlabel('Iterações')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    if avg_results and all(r['exploration_history'] is not None and r['exploitation_history'] is not None for r in avg_results):
        plt.subplot(2, 1, 2)
        for result in avg_results:
            plt.plot(result["exploration_history"],
                    label=f"{result['config']['name']} - Exploration")
            plt.plot(result["exploitation_history"],
                    linestyle='--',
                    label=f"{result['config']['name']} - Exploitation")
        
        plt.title(f'Exploração vs Exploitação - {algo_name} - {function_name}')
        plt.xlabel('Iterações')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{main_path}/resultados_comparacao.png', dpi=300)
    plt.close()
    
    # Imprimir estatísticas
    logger.info("\nComparação de resultados:")
    for result in avg_results:
        logger.info(f"{result['config']['name']} - Epoch: {result['config']['epoch']}, Pop_size: {result['config']['pop_size']}")
        logger.info(f"  Melhor fitness: {result['best_fitness']:.4f} ± {result['best_fitness_std']:.4f}")
        
        if result['exploration_history'] is not None and result['exploitation_history'] is not None:
            exp_ratio = np.mean(result['exploration_history']) / max(np.mean(result['exploitation_history']), 1e-10)
            logger.info(f"  Razão média Exploration/Exploitation: {exp_ratio:.4f}")

def analyze_results(results, model, problem):
    """
    Analisa resultados de múltiplas execuções e gera visualizações estatísticas
    """
    logger.info(f"Analisando resultados para {model} no problema {problem}")
    
    # Criar diretórios necessários
    main_path = f"results/{model}/{problem}"
    os.makedirs(main_path, exist_ok=True)
    
    # Preparar dados para análise
    stats_data = []
    for result in results:
        stats_data.append({
            'Run': result.get('run_id', 0),
            'Config': result.get('config', {}).get('name', 'Unknown'),
            'Epoch': result.get('config', {}).get('epoch', 0),
            'Pop_Size': result.get('config', {}).get('pop_size', 0),
            'Best_Fitness': result.get('best_fitness', float('inf')),
            'Execution_Time': result.get('execution_time', 0),
            'Config_Idx': result.get('config_idx', 0)
        })
    
    # Converter para DataFrame
    df = pd.DataFrame(stats_data)
    
    # 1. Box plots para melhor fitness por configuração
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Config', y='Best_Fitness')
    plt.title(f'Distribuição do Melhor Fitness - {model} - {problem}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{main_path}/fitness_boxplot.png', dpi=300)
    plt.close()
    
    # 2. Resumo estatístico
    stats_summary = df.groupby('Config')['Best_Fitness'].agg([
        'mean', 'std', 'median', 'min', 'max', 'count'
    ]).round(6)
    
    # Salvar resumo estatístico em CSV
    stats_summary.to_csv(f'{main_path}/stats_summary.csv')
    
    # 3. Análise estatística detalhada - adicionando tratamento para valores quase zero
    fitness_list = df['Best_Fitness'].values
    all_zero = all(abs(val) < 1e-10 for val in fitness_list)
    
    if all_zero:
        logger.info(f"Todos os valores de fitness são próximos de zero para {model} no problema {problem}")
        # Criar estatísticas sem skew e kurtosis para evitar warnings
        detailed_stats = df.groupby('Config').agg({
            'Best_Fitness': ['mean', 'std', 'median', 'min', 'max'],
            'Execution_Time': ['mean', 'std', 'min', 'max']
        }).round(6)
    else:
        # Calcular estatísticas completas se não forem todos zero
        detailed_stats = df.groupby('Config').agg({
            'Best_Fitness': ['mean', 'std', 'median', 'min', 'max', 
                            lambda x: stats.skew(x),  # Assimetria
                            lambda x: stats.kurtosis(x)],  # Curtose
            'Execution_Time': ['mean', 'std', 'min', 'max']
        }).round(6)
    
    # Renomear colunas para clareza
    detailed_stats.columns = ['_'.join(col).strip() for col in detailed_stats.columns.values]
    detailed_stats.to_csv(f'{main_path}/detailed_stats.csv')
    
    # 4. Box plots para tempo de execução por configuração
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Config', y='Execution_Time')
    plt.title(f'Distribuição do Tempo de Execução - {model} - {problem}')
    plt.xticks(rotation=45)
    plt.ylabel('Tempo (segundos)')
    plt.tight_layout()
    plt.savefig(f'{main_path}/execution_time_boxplot.png', dpi=300)
    plt.close()
    
    # 5. Scatter plot de fitness vs tempo de execução
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Execution_Time', y='Best_Fitness', hue='Config', style='Config')
    plt.title(f'Fitness vs Tempo de Execução - {model} - {problem}')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Melhor Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{main_path}/fitness_vs_time.png', dpi=300)
    plt.close()
    
    # Imprimir estatísticas
    logger.info(f"\nResumo estatístico para {model} no problema {problem}:")
    for config in df['Config'].unique():
        config_data = df[df['Config'] == config]
        logger.info(f"\nConfiguração: {config}")
        logger.info(f"  Número de execuções: {len(config_data)}")
        logger.info(f"  Fitness médio: {config_data['Best_Fitness'].mean():.6f} ± {config_data['Best_Fitness'].std():.6f}")
        logger.info(f"  Melhor fitness: {config_data['Best_Fitness'].min():.6f}")
        logger.info(f"  Tempo médio: {config_data['Execution_Time'].mean():.2f}s ± {config_data['Execution_Time'].std():.2f}s")
    
    return stats_summary, detailed_stats

def create_animation_static(results, model, problem):
    """
    Cria uma visualização avançada do processo de otimização.
    """
    logger.info(f"Criando animação avançada para {model} no problema {problem}")
    
    # Criar diretórios necessários
    main_path = f"results/{model}/{problem}"
    os.makedirs(main_path, exist_ok=True)
    temp_dir = f'{main_path}/frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Selecionar o melhor resultado para animação
    best_result = None
    best_fitness = float('inf')
    
    for result in results:
        if result.get('best_fitness', float('inf')) < best_fitness:
            best_fitness = result.get('best_fitness', float('inf'))
            best_result = result
    
    if not best_result:
        logger.warning(f"Nenhum resultado disponível para {model} no problema {problem}")
        return None
    
    # Obter posição do melhor resultado
    best_position = best_result.get('best_position')
    if best_position is None:
        logger.warning(f"Sem dados de posição para {model} no problema {problem}")
        return None
    
    # Obter histórico de fitness para visualização
    fitness_history = best_result.get('fitness_history', [])
    if not fitness_history:
        logger.warning(f"Sem histórico de fitness para {model} no problema {problem}")
        return None
    
    # Verificar histórico de posições
    position_data = best_result.get('positions_history')
    if position_data is None or len(position_data) == 0:
        logger.warning(f"Sem histórico de posições disponível para {model} no problema {problem}, criando visualização sintética...")
        
        # Código para criar dados sintéticos (mantido do original)
        # ...
    else:
        logger.info(f"Usando histórico de posições real com {len(position_data)} épocas e {len(position_data[0])} partículas")
    
    # Detectar convergência
    is_near_zero = abs(best_fitness) < 1e-6
    convergence_epoch = None
    if is_near_zero:
        for i, fit in enumerate(fitness_history):
            if abs(fit) < 1e-6:
                convergence_epoch = i
                logger.info(f"Convergência para quase zero ocorreu na época {convergence_epoch}")
                break
    
    # Preparar para rastrear trajetórias das melhores partículas
    num_particles = len(position_data[0])
    logger.info(f"Número de partículas disponíveis: {num_particles}")
    
    # Selecionar partículas para rastreamento (no máximo 10 para não poluir)
    max_tracked = min(10, num_particles)
    tracked_indices = list(range(max_tracked))
    
    # Armazenar trajetórias das partículas rastreadas
    trajectories = {idx: [] for idx in tracked_indices}
    
    # Determinar passo para limitar o total de frames
    total_frames = len(position_data)
    step = max(1, total_frames // 40)  # Ajustar para aproximadamente 40 frames
    
    # Preparar arquivos de frame
    frame_files = []
    
    # Determinar escala global para evitar saltos no zoom
    # Coletar todas as posições para encontrar limites globais
    all_x = []
    all_y = []
    for frame in position_data:
        for pos in frame:
            all_x.append(pos[0])
            all_y.append(pos[1])
    
    # Calcular limites com margem extra para zoom global
    global_x_min, global_x_max = min(all_x), max(all_x)
    global_y_min, global_y_max = min(all_y), max(all_y)
    
    # Adicionar margem
    margin_x = (global_x_max - global_x_min) * 0.1
    margin_y = (global_y_max - global_y_min) * 0.1
    
    # Limites globais com margem
    global_xlim = [global_x_min - margin_x, global_x_max + margin_x]
    global_ylim = [global_y_min - margin_y, global_y_max + margin_y]
    
    logger.info(f"Criando {total_frames//step} frames de animação...")
    
    # Criar um mostrador de progresso
    frames_to_generate = len(range(0, total_frames, step))
    frames_completed = 0
    
    # Definir as cores para as trajetórias
    cmap = plt.get_cmap('tab10')
    trajectory_colors = [cmap(i % 10) for i in range(max_tracked)]
    
    # Função auxiliar para gerar uma figura de múltiplos painéis
    def create_multi_panel_figure(i, include_trajectories=True):
        # Criar figura com múltiplos painéis
        fig = plt.figure(figsize=(16, 12))
        # Layout da figura: 2 linhas, 2 colunas com diferentes tamanhos de painel
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
        
        # Painel 1: Visualização principal com posições atuais e trajetórias
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Painel 2: Histórico de fitness
        ax_fitness = fig.add_subplot(gs[1, 0])
        
        # Painel 3: Zoom na região de interesse
        ax_zoom = fig.add_subplot(gs[0, 1])
        
        # Painel 4: Informações extras e estatísticas
        ax_info = fig.add_subplot(gs[1, 1])
        
        # Obter posições e fitness atuais
        positions = position_data[i]
        current_fitness = fitness_history[min(i, len(fitness_history)-1)]
        
        # 1. PAINEL PRINCIPAL - Posições e trajetórias
        # Atualizar trajetórias para este frame
        for idx in tracked_indices:
            if idx < len(positions):
                trajectories[idx].append(positions[idx])
        
        # Plotar todas as partículas
        x_values = [pos[0] for pos in positions]
        y_values = [pos[1] for pos in positions]
        distances = [np.sqrt((x-best_position[0])**2 + (y-best_position[1])**2) for x, y in zip(x_values, y_values)]
        
        # Calcular estatísticas para este frame
        min_dist = min(distances) if distances else 0
        max_dist = max(distances) if distances else 0
        mean_dist = sum(distances) / len(distances) if distances else 0
        std_dist = np.std(distances) if distances else 0
        
        # Plotar partículas com cores baseadas na distância
        scatter = ax_main.scatter(
            x_values, y_values, 
            c=distances, cmap='coolwarm_r', 
            s=30, alpha=0.7, 
            edgecolors='black', linewidths=0.5
        )
        
        # Adicionar trajetórias se solicitado
        if include_trajectories:
            for idx, trajectory in trajectories.items():
                if len(trajectory) > 1:  # Precisa de pelo menos 2 pontos para plotar linha
                    traj_x = [pos[0] for pos in trajectory]
                    traj_y = [pos[1] for pos in trajectory]
                    ax_main.plot(traj_x, traj_y, '-', color=trajectory_colors[idx % len(trajectory_colors)], 
                                alpha=0.5, linewidth=1.5, label=f'Partícula {idx}')
        
        # Adicionar a melhor posição
        ax_main.scatter(
            best_position[0], best_position[1],
            marker='*', s=200, color='gold', edgecolors='black',
            label='Melhor Posição'
        )
        
        # Configurar o painel principal
        ax_main.set_title(f'Posições das Partículas - Época {i+1}/{total_frames}', fontsize=12)
        ax_main.set_xlabel('Dimensão 1', fontsize=10)
        ax_main.set_ylabel('Dimensão 2', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(global_xlim)
        ax_main.set_ylim(global_ylim)
        
        # Legenda compacta para o painel principal
        if include_trajectories:
            handles, labels = ax_main.get_legend_handles_labels()
            # Limitar para não ficar muito grande
            max_items = min(6, len(handles))
            ax_main.legend(handles[:max_items], labels[:max_items], loc='upper right', fontsize=8)
        
        # 2. PAINEL DE FITNESS - Histórico de fitness
        ax_fitness.plot(fitness_history[:i+1], 'b-', linewidth=2)
        ax_fitness.set_title('Histórico de Fitness', fontsize=12)
        ax_fitness.set_xlabel('Época', fontsize=10)
        ax_fitness.set_ylabel('Fitness', fontsize=10)
        ax_fitness.grid(True, alpha=0.3)
        # Marcar época atual
        ax_fitness.axvline(x=i, color='r', linestyle='--', alpha=0.5)
        if convergence_epoch is not None and convergence_epoch <= i:
            ax_fitness.axvline(x=convergence_epoch, color='g', linestyle='-', alpha=0.5)
            ax_fitness.text(convergence_epoch, max(fitness_history[:i+1]), 'Convergência', 
                         color='green', ha='right', va='bottom', fontsize=8)
        
        # 3. PAINEL DE ZOOM - Foco nas melhores partículas
        # Determinar área de zoom (centrada na melhor posição)
        zoom_level = 0.2  # 20% do espaço de busca global
        if is_near_zero:
            # Zoom mais próximo se já convergiu
            zoom_level = 0.05
        
        # Calcular limites do zoom
        range_x = global_xlim[1] - global_xlim[0]
        range_y = global_ylim[1] - global_ylim[0]
        
        zoom_xlim = [
            best_position[0] - range_x * zoom_level,
            best_position[0] + range_x * zoom_level
        ]
        
        zoom_ylim = [
            best_position[1] - range_y * zoom_level,
            best_position[1] + range_y * zoom_level
        ]
        
        # Plotar partículas na área de zoom
        ax_zoom.scatter(
            x_values, y_values, 
            c=distances, cmap='coolwarm_r', 
            s=50, alpha=0.7, 
            edgecolors='black', linewidths=0.5
        )
        
        # Adicionar melhor posição
        ax_zoom.scatter(
            best_position[0], best_position[1],
            marker='*', s=200, color='gold', edgecolors='black',
            label='Melhor Posição'
        )
        
        # Configurar o painel de zoom
        ax_zoom.set_title('Zoom na Região de Interesse', fontsize=12)
        ax_zoom.set_xlim(zoom_xlim)
        ax_zoom.set_ylim(zoom_ylim)
        ax_zoom.grid(True, alpha=0.3)
        
        # 4. PAINEL DE INFORMAÇÕES - Estatísticas e meta-dados
        ax_info.axis('off')  # Remover eixos para texto puro
        
        # Montar texto informativo
        info_text = [
            f"Algoritmo: {model}",
            f"Problema: {problem}",
            f"Época: {i+1}/{total_frames}",
            f"Fitness atual: {current_fitness:.6e}" if abs(current_fitness) < 1e-6 else f"Fitness atual: {current_fitness:.6f}",
            f"Melhor fitness: {best_fitness:.6e}" if abs(best_fitness) < 1e-6 else f"Melhor fitness: {best_fitness:.6f}",
            f"Partículas: {len(positions)}",
            "",
            "Estatísticas de distância:",
            f"  Mínima: {min_dist:.2f}",
            f"  Média: {mean_dist:.2f}",
            f"  Máxima: {max_dist:.2f}",
            f"  Desvio padrão: {std_dist:.2f}"
        ]
        
        # Adicionar informação sobre convergência
        if convergence_epoch is not None:
            info_text.append("")
            info_text.append(f"Convergência na época {convergence_epoch}")
            if i >= convergence_epoch:
                info_text.append("Status: Convergido ✓")
            else:
                info_text.append(f"Status: Explorando... ({i}/{convergence_epoch})")
        
        # Adicionar texto ao painel
        ax_info.text(0.05, 0.95, '\n'.join(info_text), 
                   transform=ax_info.transAxes, 
                   fontsize=10, va='top')
        
        # Ajustar layout
        plt.tight_layout()
        
        return fig
    
    # Gerar frames
    for i in range(0, total_frames, step):
        if i >= len(position_data):
            break
        
        try:
            # Criar figura multi-painel
            fig = create_multi_panel_figure(i)
            
            # Salvar frame
            frame_file = f"{temp_dir}/frame_{i:04d}.png"
            plt.savefig(frame_file, dpi=150)
            frame_files.append(frame_file)
            plt.close(fig)
            
            # Atualizar progresso
            frames_completed += 1
            if frames_completed % 5 == 0 or frames_completed == frames_to_generate:
                logger.info(f"Progresso: {frames_completed}/{frames_to_generate} frames ({frames_completed/frames_to_generate*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Erro ao criar frame {i}: {e}")
            traceback.print_exc()
    
    # Criar GIF a partir dos frames
    if frame_files:
        try:
            frames = [Image.open(f) for f in frame_files]
            
            if frames:
                output_path = f'{main_path}/optimization.gif'
                
                # Salvar como GIF
                frames[0].save(
                    output_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=250,  # ms entre frames
                    loop=0  # loop infinito
                )
                
                logger.info(f"Animação salva em {output_path}")
                
                # Limpar arquivos temporários
                for f in frame_files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.warning(f"Erro ao remover arquivo temporário {f}: {e}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Erro ao criar GIF: {e}")
            traceback.print_exc()
    else:
        logger.warning("Nenhum frame foi criado, a animação não pôde ser gerada")
    
    return None

def generate_visualizations(results, model, problem):
    """
    Gera todas as visualizações para um conjunto de resultados
    """
    logger.info(f"Gerando visualizações para {model} no problema {problem}")
    
    try:
        # Análise estatística detalhada
        stats_summary, detailed_stats = analyze_results(results, model, problem)
        
        # Plotar resultados comparativos entre configurações
        plot_results(results, model, problem)
        
        # Criar diretórios necessários
        main_path = f"results/{model}/{problem}"
        os.makedirs(main_path, exist_ok=True)
        
        # Salvar dados brutos para análise posterior
        try:
            # Salvar apenas dados essenciais para evitar arquivos muito grandes
            essential_data = []
            for result in results:
                essential_result = {
                    'run_id': result.get('run_id', 0),
                    'config_idx': result.get('config_idx', 0),
                    'config_name': result.get('config', {}).get('name', 'Unknown'),
                    'best_fitness': result.get('best_fitness', float('inf')),
                    'execution_time': result.get('execution_time', 0)
                }
                essential_data.append(essential_result)
            
            # Salvar como CSV
            pd.DataFrame(essential_data).to_csv(f'{main_path}/raw_results.csv', index=False)
        except Exception as e:
            logger.error(f"Erro ao salvar dados brutos: {e}")
        
        logger.info(f"Visualizações geradas com sucesso para {model} no problema {problem}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Criar diretório principal para resultados
        os.makedirs("results", exist_ok=True)
        
        # Definir modelos e problemas a serem executados
        MODELS = ["PSO", 
                  "ACO"
                  ]  # Usando nossa versão personalizada do ACO e PSO
        PROBLEMS = ["f1", 
                    "f6"
                    ]
        
        # Para teste rápido, use valores menores
        NUM_RUNS = 30  # 3 execuções por configuração
        logger.info(f"Iniciando execução principal com {NUM_RUNS} execuções")
        
        # Criar diretórios para cada modelo e problema
        for model in MODELS:
            for problem in PROBLEMS:
                os.makedirs(f"results/{model}/{problem}", exist_ok=True)
        
        # Executar análise otimizada com paralelismo total
        results = run_analysis_optimized(models=MODELS, problems=PROBLEMS, num_runs=NUM_RUNS)
        
        logger.info("Análise completa!")
        
        # Resumo final dos resultados
        logger.info("\n===== RESUMO FINAL DOS RESULTADOS =====")
        for model in MODELS:
            for problem in PROBLEMS:
                if model in results and problem in results[model]:
                    stats = results[model][problem]
                    logger.info(f"\n{model} no problema {problem}:")
                    logger.info(f"  Fitness médio: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
                    logger.info(f"  Melhor fitness: {stats['min_fitness']:.6f}")
                    logger.info(f"  Tempo médio: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
                    logger.info(f"  Visualizações disponíveis em: results/{model}/{problem}/")
        
    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        traceback.print_exc() 