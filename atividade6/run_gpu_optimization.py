#!/usr/bin/env python
import sys
import argparse
from gpu_optimized import run_analysis_optimized
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Executar otimização com GPU')
    parser.add_argument('--models', nargs='+', default=["PSO", "GA"], 
                        help='Modelos para executar (PSO, GA, ACO)')
    parser.add_argument('--problems', nargs='+', default=["f1", "f6"], 
                        help='Problemas para resolver (f1, f6)')
    parser.add_argument('--runs', type=int, default=9, 
                        help='Número total de execuções (será dividido entre as configurações)')
    
    args = parser.parse_args()
    
    logging.info(f"Iniciando otimização com GPU: modelos={args.models}, problemas={args.problems}, execuções={args.runs}")
    
    try:
        results = run_analysis_optimized(models=args.models, problems=args.problems, num_runs=args.runs)
        logging.info("Otimização concluída com sucesso!")
    except Exception as e:
        logging.error(f"Erro durante a otimização: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 