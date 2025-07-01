#!/usr/bin/env python3
"""
Script de teste para verificar se os problemas de encoding foram resolvidos.
Executa apenas uma iteração do baseline MLR para teste rápido.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from metrics import compute_metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.logging_utils import get_logger

logger = get_logger('test_encoding')

# Caminhos dos arquivos
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'calibration.csv'
TEST_CSV = BASE_DIR / 'data' / 'csv_new' / 'test.csv'
VALIDATION_CSV = BASE_DIR / 'data' / 'csv_new' / 'validation.csv'
IDRC_CSV = BASE_DIR / 'data' / 'csv_new' / 'idrc_validation.csv'

def load_data(calib_path: Path, idrc_path: Path):
    """Carrega os dados de calibração e IDRC."""
    logger.info(f"Carregando dados de calibração de: {calib_path}")
    df = pd.read_csv(calib_path)
    logger.info(f"Dados de calibração carregados. Shape: {df.shape}")
    logger.info(f"Carregando dados IDRC de: {idrc_path}")
    idrc_df = pd.read_csv(idrc_path)
    logger.info(f"Dados IDRC carregados. Shape: {idrc_df.shape}")
    return df, idrc_df

def prepare_datasets(df: pd.DataFrame, idrc_df: pd.DataFrame):
    """Prepara os conjuntos de treino, teste e validação."""
    # Carrega os dados de teste e validação
    test_df = pd.read_csv(TEST_CSV)
    validation_df = pd.read_csv(VALIDATION_CSV)
    
    # Seleciona apenas as colunas numéricas (excluindo a última coluna que é o target)
    feature_columns = [col for col in df.columns if col != 'target']
    
    # Prepara os dados de treino (usando todos os dados de calibração)
    X_train = df[feature_columns]
    y_train = df['target']
    
    # Prepara os dados de teste (do arquivo test.csv)
    X_test = test_df[feature_columns]
    y_test = test_df['target']
    
    # Prepara os dados de validação (do arquivo validation.csv)
    X_val = validation_df[feature_columns]
    y_val = idrc_df['reference']
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val

def test_baseline_mlr():
    """Executa uma única iteração do baseline MLR para teste."""
    logger.info("TESTE: Executando uma iteração do Baseline MLR")
    
    try:
        # Carrega dados
        df, idrc_df = load_data(CALIBRATION_CSV, IDRC_CSV)
        X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val = prepare_datasets(df, idrc_df)
        
        # Treina modelo
        logger.info("Treinando modelo de regressão linear...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Gera predições
        y_pred_val = model.predict(X_val_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calcula métricas
        metrics_val = compute_metrics(y_val, y_pred_val)
        metrics_test = compute_metrics(y_test, y_pred_test)
        
        # Exibe resultados
        logger.info("RESULTADOS DO TESTE:")
        logger.info(f"Validação - R²: {metrics_val['R2']:.4f}, MSE: {metrics_val['MSE']:.4f}")
        logger.info(f"Teste - R²: {metrics_test['R2']:.4f}, MSE: {metrics_test['MSE']:.4f}")
        
        logger.info("SUCESSO: Teste concluído sem erros de encoding!")
        return True
        
    except Exception as e:
        logger.error(f"ERRO: Falha no teste: {e}")
        return False

def main():
    """Função principal de teste."""
    logger.info("=" * 60)
    logger.info("TESTE DE ENCODING - BASELINE MLR")
    logger.info("=" * 60)
    
    success = test_baseline_mlr()
    
    logger.info("=" * 60)
    if success:
        logger.info("RESULTADO: TESTE PASSOU - Sistema funcionando!")
    else:
        logger.info("RESULTADO: TESTE FALHOU - Verifique os erros acima")
    logger.info("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 