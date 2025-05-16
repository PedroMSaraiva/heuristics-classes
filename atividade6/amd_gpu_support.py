"""
Módulo de suporte para GPUs AMD usando PyOpenCL.
Este módulo tenta usar PyOpenCL para acelerar cálculos em GPUs AMD.
"""

import numpy as np
import logging
import os
import platform

# Tentar importar PyOpenCL
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

logger = logging.getLogger("optimization")

class AMDGPUAccelerator:
    """Classe para acelerar cálculos em GPUs AMD usando PyOpenCL"""
    
    def __init__(self):
        self.ctx = None
        self.queue = None
        self.programs = {}
        self.available = False
        self.device_name = "Nenhum"
        
        if not PYOPENCL_AVAILABLE:
            logger.warning("PyOpenCL não está disponível. Instale com: pip install pyopencl")
            return
            
        try:
            # Tentar inicializar o contexto OpenCL
            platforms = cl.get_platforms()
            if not platforms:
                logger.warning("Nenhuma plataforma OpenCL encontrada")
                return
                
            # Procurar por dispositivos AMD
            for platform in platforms:
                if "AMD" in platform.name.upper():
                    devices = platform.get_devices()
                    if devices:
                        self.ctx = cl.Context(devices)
                        self.queue = cl.CommandQueue(self.ctx)
                        self.device_name = devices[0].name
                        self.available = True
                        logger.info(f"GPU AMD encontrada: {self.device_name}")
                        break
            
            # Se não encontrou AMD, tenta usar qualquer GPU disponível
            if not self.available:
                for platform in platforms:
                    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if gpu_devices:
                        self.ctx = cl.Context(gpu_devices)
                        self.queue = cl.CommandQueue(self.ctx)
                        self.device_name = gpu_devices[0].name
                        self.available = True
                        logger.info(f"GPU encontrada (não-AMD): {self.device_name}")
                        break
            
            if not self.available:
                logger.warning("Nenhuma GPU OpenCL encontrada")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar OpenCL: {e}")
            self.available = False
    
    def _compile_program(self, program_name, source_code):
        """Compila um programa OpenCL"""
        if not self.available:
            return False
            
        try:
            self.programs[program_name] = cl.Program(self.ctx, source_code).build()
            return True
        except Exception as e:
            logger.error(f"Erro ao compilar programa {program_name}: {e}")
            return False
    
    def initialize_f1(self):
        """Inicializa o kernel para a função F1"""
        if not self.available:
            return False
            
        # Código OpenCL para a função F1
        f1_kernel_src = """
        __kernel void evaluate_f1(__global const float* x,
                                 __global float* result,
                                 const int vector_size)
        {
            int idx = get_global_id(0);
            float sum = 0.0f;
            
            for (int i = 0; i < vector_size; i++) {
                float exponent = (float)i / (float)(vector_size - 1);
                float coefficient = pow(1000000.0f, exponent);
                sum += coefficient * x[idx * vector_size + i] * x[idx * vector_size + i];
            }
            
            result[idx] = sum;
        }
        """
        
        return self._compile_program("f1", f1_kernel_src)
    
    def initialize_f6(self, a_powers, b_powers, kmax):
        """Inicializa o kernel para a função F6"""
        if not self.available:
            return False
            
        # Código OpenCL para a função F6
        f6_kernel_src = """
        __kernel void evaluate_f6(__global const float* x,
                                 __global const float* a_powers,
                                 __global const float* b_powers,
                                 __global float* result,
                                 const int vector_size,
                                 const int kmax,
                                 const float constant_term)
        {
            int idx = get_global_id(0);
            float variable_sum = 0.0f;
            
            for (int i = 0; i < vector_size; i++) {
                float inner_sum = 0.0f;
                for (int k = 0; k <= kmax; k++) {
                    inner_sum += a_powers[k] * cos(2.0f * M_PI * b_powers[k] * (x[idx * vector_size + i] + 0.5f));
                }
                variable_sum += inner_sum;
            }
            
            result[idx] = variable_sum - constant_term;
        }
        """
        
        success = self._compile_program("f6", f6_kernel_src)
        
        if success:
            # Criar buffers para os arrays constantes
            self.a_powers_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                         hostbuf=np.array(a_powers, dtype=np.float32))
            self.b_powers_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                         hostbuf=np.array(b_powers, dtype=np.float32))
        
        return success
    
    def evaluate_f1_batch(self, vectors):
        """Avalia um lote de vetores usando a função F1 na GPU"""
        if not self.available or "f1" not in self.programs:
            return None
            
        try:
            # Converter para float32 para melhor desempenho em GPUs
            vectors_np = np.array(vectors, dtype=np.float32)
            n_vectors = vectors_np.shape[0]
            vector_size = vectors_np.shape[1]
            
            # Criar buffers
            vectors_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                   hostbuf=vectors_np)
            result_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 
                                  size=n_vectors * np.dtype(np.float32).itemsize)
            
            # Executar kernel
            kernel = self.programs["f1"].evaluate_f1
            kernel.set_args(vectors_buf, result_buf, np.int32(vector_size))
            
            # Definir tamanho global e local de trabalho
            global_size = (n_vectors,)
            local_size = None  # Deixar o OpenCL decidir
            
            # Executar
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            
            # Ler resultados
            result = np.empty(n_vectors, dtype=np.float32)
            cl.enqueue_copy(self.queue, result, result_buf)
            self.queue.finish()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao avaliar F1 na GPU AMD: {e}")
            return None
    
    def evaluate_f6_batch(self, vectors, constant_term):
        """Avalia um lote de vetores usando a função F6 na GPU"""
        if not self.available or "f6" not in self.programs:
            return None
            
        try:
            # Converter para float32 para melhor desempenho em GPUs
            vectors_np = np.array(vectors, dtype=np.float32)
            n_vectors = vectors_np.shape[0]
            vector_size = vectors_np.shape[1]
            
            # Criar buffers
            vectors_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                   hostbuf=vectors_np)
            result_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 
                                  size=n_vectors * np.dtype(np.float32).itemsize)
            
            # Executar kernel
            kernel = self.programs["f6"].evaluate_f6
            kernel.set_args(vectors_buf, self.a_powers_buf, self.b_powers_buf, 
                           result_buf, np.int32(vector_size), np.int32(20), 
                           np.float32(constant_term))
            
            # Definir tamanho global e local de trabalho
            global_size = (n_vectors,)
            local_size = None  # Deixar o OpenCL decidir
            
            # Executar
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            
            # Ler resultados
            result = np.empty(n_vectors, dtype=np.float32)
            cl.enqueue_copy(self.queue, result, result_buf)
            self.queue.finish()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao avaliar F6 na GPU AMD: {e}")
            return None

# Função para verificar se temos suporte a GPU AMD
def check_amd_gpu_support():
    """Verifica se há suporte para GPU AMD"""
    accelerator = AMDGPUAccelerator()
    return accelerator.available, accelerator.device_name

# Função para obter uma instância do acelerador AMD
def get_amd_accelerator(a_powers=None, b_powers=None):
    """Retorna uma instância configurada do acelerador AMD"""
    accelerator = AMDGPUAccelerator()
    
    if accelerator.available:
        # Inicializar kernels
        f1_ok = accelerator.initialize_f1()
        f6_ok = False
        
        if a_powers is not None and b_powers is not None:
            f6_ok = accelerator.initialize_f6(a_powers, b_powers, len(a_powers)-1)
        
        if f1_ok:
            logger.info("Kernel F1 inicializado com sucesso para GPU AMD")
        else:
            logger.warning("Falha ao inicializar kernel F1 para GPU AMD")
            
        if f6_ok:
            logger.info("Kernel F6 inicializado com sucesso para GPU AMD")
        else:
            logger.warning("Falha ao inicializar kernel F6 para GPU AMD")
    
    return accelerator 