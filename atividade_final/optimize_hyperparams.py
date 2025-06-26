#!/usr/bin/env python3
"""
Script para otimização de hiperparâmetros do algoritmo genético.
Permite execução rápida com diferentes configurações.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from genetic_mlr import run_with_optimization
from src.logging_utils import get_logger
import argparse
import time

logger = get_logger('hyperopt')

def quick_optimization(n_trials: int = 30):
    """Executa otimização rápida para teste."""
    logger.info(f"🚀 Iniciando otimização rápida com {n_trials} trials")
    start_time = time.time()
    
    results = run_with_optimization(optimize=True, n_trials=n_trials)
    
    end_time = time.time()
    logger.info(f"⏱️ Otimização concluída em {end_time - start_time:.1f} segundos")
    
    return results

def full_optimization(n_trials: int = 100):
    """Executa otimização completa."""
    logger.info(f"🔥 Iniciando otimização completa com {n_trials} trials")
    start_time = time.time()
    
    results = run_with_optimization(optimize=True, n_trials=n_trials)
    
    end_time = time.time()
    logger.info(f"⏱️ Otimização concluída em {end_time - start_time:.1f} segundos")
    
    return results

def run_with_best_params():
    """Executa com os melhores parâmetros salvos."""
    logger.info("⚡ Executando com melhores parâmetros salvos")
    start_time = time.time()
    
    results = run_with_optimization(optimize=False)
    
    end_time = time.time()
    logger.info(f"⏱️ Execução concluída em {end_time - start_time:.1f} segundos")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Otimização de Hiperparâmetros do Algoritmo Genético')
    parser.add_argument('mode', choices=['quick', 'full', 'best'], 
                       help='Modo de execução: quick (30 trials), full (100 trials), best (usar salvos)')
    parser.add_argument('--trials', type=int, 
                       help='Número customizado de trials (sobrescreve o padrão do modo)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        trials = args.trials or 30
        results = quick_optimization(trials)
    elif args.mode == 'full':
        trials = args.trials or 100
        results = full_optimization(trials)
    elif args.mode == 'best':
        results = run_with_best_params()
    
    # Relatório final
    print("\n" + "="*60)
    print("📊 RELATÓRIO FINAL DA OTIMIZAÇÃO")
    print("="*60)
    print(f"R² Validação: {results['metrics_val']['R2']:.4f}")
    print(f"R² Teste: {results['metrics_test']['R2']:.4f}")
    print(f"MSE Validação: {results['metrics_val']['MSE']:.4f}")
    print(f"MAE Validação: {results['metrics_val']['MAE']:.4f}")
    print(f"Features selecionadas: {results['num_features']}")
    print(f"Melhores parâmetros: {results['best_params']}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main() 