#!/usr/bin/env python3
"""
Script principal para executar análise estatística completa dos algoritmos MLR.

Este script orquestra:
1. Execução de 30 iterações do Baseline MLR
2. Execução de 30 iterações do Genetic MLR (com melhores parâmetros)
3. Análise comparativa com geração de plots e estatísticas

Uso:
    python run_statistical_analysis.py [--baseline-only] [--genetic-only] [--analysis-only]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from src.logging_utils import get_logger
import time

logger = get_logger('statistical_analysis')

BASE_DIR = Path(__file__).resolve().parent

def run_script(script_name: str, description: str) -> bool:
    """Executa um script Python e retorna True se bem-sucedido.
    
    Args:
        script_name (str): Nome do script a ser executado.
        description (str): Descrição do que o script faz.
        
    Returns:
        bool: True se execução foi bem-sucedida, False caso contrário.
    """
    script_path = BASE_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"ERRO: Script não encontrado: {script_path}")
        return False
    
    logger.info(f"INICIANDO: {description}")
    logger.info(f"Executando: {script_name}")
    
    start_time = time.time()
    
    try:
        # Executa o script usando o mesmo interpretador Python
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Substitui caracteres problemáticos
            timeout=7200  # Timeout de 2 horas
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"SUCESSO: {description} concluído!")
            logger.info(f"Tempo de execução: {elapsed_time/60:.1f} minutos")
            
            # Log da saída se houver informações importantes
            if result.stdout and len(result.stdout.strip()) > 0:
                logger.debug("Saída do script:")
                for line in result.stdout.strip().split('\n')[-10:]:  # Últimas 10 linhas
                    logger.debug(f"  {line}")
                    
            return True
        else:
            logger.error(f"ERRO: {description} falhou!")
            logger.error(f"Código de saída: {result.returncode}")
            
            if result.stderr:
                logger.error("Erro:")
                for line in result.stderr.strip().split('\n'):
                    logger.error(f"  {line}")
                    
            if result.stdout:
                logger.error("Saída:")
                for line in result.stdout.strip().split('\n')[-5:]:  # Últimas 5 linhas
                    logger.error(f"  {line}")
                    
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description} excedeu o tempo limite (2 horas)")
        return False
    except Exception as e:
        logger.error(f"ERRO: Erro ao executar {description}: {e}")
        return False

def check_prerequisites():
    """Verifica se os pré-requisitos estão atendidos.
    
    Returns:
        bool: True se todos os pré-requisitos estão atendidos.
    """
    logger.info("Verificando pré-requisitos...")
    
    # Verifica se os arquivos de dados existem
    required_files = [
        'data/csv_new/calibration.csv',
        'data/csv_new/test.csv',
        'data/csv_new/validation.csv',
        'data/csv_new/idrc_validation.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("ERRO: Arquivos de dados obrigatórios não encontrados:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    # Verifica se os melhores parâmetros existem para o genetic MLR
    best_params_file = BASE_DIR / 'results' / 'best_hyperparameters.json'
    if not best_params_file.exists():
        logger.warning("AVISO: Arquivo de melhores parâmetros não encontrado:")
        logger.warning(f"  - {best_params_file}")
        logger.warning("  O algoritmo genético usará parâmetros padrão.")
    
    # Verifica se os scripts existem
    required_scripts = [
        'baseline_mlr_multiple_runs.py',
        'genetic_mlr_multiple_runs.py',
        'analyze_multiple_runs.py'
    ]
    
    missing_scripts = []
    for script_name in required_scripts:
        script_path = BASE_DIR / script_name
        if not script_path.exists():
            missing_scripts.append(script_name)
    
    if missing_scripts:
        logger.error("ERRO: Scripts obrigatórios não encontrados:")
        for script_name in missing_scripts:
            logger.error(f"  - {script_name}")
        return False
    
    logger.info("SUCESSO: Todos os pré-requisitos verificados!")
    return True

def estimate_runtime():
    """Estima o tempo total de execução."""
    logger.info("ESTIMATIVA de tempo de execução:")
    logger.info("  - Baseline MLR (30 execuções): ~5-10 minutos")
    logger.info("  - Genetic MLR (30 execuções): ~60-120 minutos")
    logger.info("  - Análise e plots: ~2-5 minutos")
    logger.info("  Tempo total estimado: 70-140 minutos")

def main():
    """Função principal que orquestra toda a análise estatística."""
    parser = argparse.ArgumentParser(
        description='Executa análise estatística completa dos algoritmos MLR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_statistical_analysis.py                 # Executa tudo
  python run_statistical_analysis.py --baseline-only # Apenas baseline
  python run_statistical_analysis.py --genetic-only  # Apenas genetic
  python run_statistical_analysis.py --analysis-only # Apenas análise
        """
    )
    
    parser.add_argument('--baseline-only', action='store_true',
                       help='Executa apenas as 30 iterações do Baseline MLR')
    parser.add_argument('--genetic-only', action='store_true',
                       help='Executa apenas as 30 iterações do Genetic MLR')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Executa apenas a análise dos resultados existentes')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Pula a verificação de pré-requisitos')
    
    args = parser.parse_args()
    
    # Banner inicial
    logger.info("=" * 80)
    logger.info("ANÁLISE ESTATÍSTICA MLR - BASELINE vs GENETIC")
    logger.info("Execução de 30 iterações + Análise Comparativa")
    logger.info("=" * 80)
    
    # Verifica pré-requisitos
    if not args.skip_checks and not check_prerequisites():
        logger.error("ERRO: Pré-requisitos não atendidos. Abortando execução.")
        return 1
    
    # Estima tempo de execução
    if not (args.baseline_only or args.genetic_only or args.analysis_only):
        estimate_runtime()
        
        # Pergunta confirmação
        try:
            response = input("\nDeseja continuar com a execução completa? (s/N): ").strip().lower()
            if response not in ['s', 'sim', 'y', 'yes']:
                logger.info("Execução cancelada pelo usuário.")
                return 0
        except KeyboardInterrupt:
            logger.info("\nExecução cancelada pelo usuário.")
            return 0
    
    total_start_time = time.time()
    success_count = 0
    total_steps = 0
    
    # Executa Baseline MLR
    if not args.genetic_only and not args.analysis_only:
        total_steps += 1
        if run_script('baseline_mlr_multiple_runs.py', 'Baseline MLR (30 execuções)'):
            success_count += 1
        else:
            logger.error("ERRO: Falha no Baseline MLR. Continuando...")
    
    # Executa Genetic MLR
    if not args.baseline_only and not args.analysis_only:
        total_steps += 1
        if run_script('genetic_mlr_multiple_runs.py', 'Genetic MLR (30 execuções)'):
            success_count += 1
        else:
            logger.error("ERRO: Falha no Genetic MLR. Continuando...")
    
    # Executa análise
    if not args.baseline_only and not args.genetic_only:
        total_steps += 1
        if run_script('analyze_multiple_runs.py', 'Análise comparativa e plots'):
            success_count += 1
        else:
            logger.error("ERRO: Falha na análise. Verifique se os resultados existem.")
    
    # Relatório final
    total_elapsed = time.time() - total_start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("RELATÓRIO FINAL")
    logger.info("=" * 80)
    logger.info(f"Etapas concluídas: {success_count}/{total_steps}")
    logger.info(f"Tempo total: {total_elapsed/60:.1f} minutos")
    
    if success_count == total_steps:
        logger.info("SUCESSO: Análise estatística concluída!")
        logger.info("Verifique os resultados em:")
        logger.info(f"   - {BASE_DIR}/results/multiple_runs/ (dados)")
        logger.info(f"   - {BASE_DIR}/results/analysis_plots/ (gráficos)")
    else:
        logger.warning(f"AVISO: Algumas etapas falharam ({total_steps - success_count} falhas)")
        logger.info("Verifique os logs acima para detalhes dos erros")
        
    logger.info("=" * 80)
    
    return 0 if success_count == total_steps else 1

if __name__ == "__main__":
    sys.exit(main()) 