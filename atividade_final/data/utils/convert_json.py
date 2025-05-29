import pandas as pd
import json
import os
import sys
from pathlib import Path

def convert_idrc_json_to_csv(json_path):
    """
    Converte um arquivo JSON no formato IDRC para CSV.
    
    Args:
        json_path (str): Caminho para o arquivo JSON de entrada
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"O arquivo {json_path} não foi encontrado.")
        
        # Ler o arquivo JSON
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Verificar se o arquivo tem a estrutura esperada
        required_keys = ['VarLabels0', 'ObjLabels', 'IDRC_Validation_set_references_Sheet1']
        if not all(key in data for key in required_keys):
            raise ValueError("O arquivo JSON não contém todas as chaves necessárias.")
        
        # Extrair os valores e converter para uma lista simples
        values = [item[0] for item in data['IDRC_Validation_set_references_Sheet1']]
        
        # Criar um DataFrame com os labels e valores
        df = pd.DataFrame({
            'Object': data['ObjLabels'],
            'Value': values
        })
        
        # Adicionar a informação do VarLabels0 como metadado no nome da coluna
        df = df.rename(columns={'Value': f"Value ({data['VarLabels0'][0]})"})
        
        # Gerar o caminho de saída baseado no caminho de entrada
        output_path = str(Path(json_path).with_suffix('.csv'))
        
        # Salvar como CSV
        df.to_csv(output_path, index=False)
        print(f"Dados salvos em {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Erro ao processar o arquivo: {str(e)}")
        return None

def convert_matlab_json_to_csv(json_path):
    """
    Converte um arquivo JSON exportado do MATLAB para CSV.
    
    Args:
        json_path (str): Caminho para o arquivo JSON de entrada
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"O arquivo {json_path} não foi encontrado.")
        
        # Ler o arquivo JSON
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Criar um diretório para os arquivos CSV de saída
        output_dir = str(Path(json_path).parent / 'csv_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Processar cada chave do JSON
        for key, value in data.items():
            try:
                # Ignorar chaves internas do MATLAB
                if key.startswith('__'):
                    continue
                
                # Converter o valor para DataFrame
                if isinstance(value, list):
                    # Se for uma lista de listas, converter diretamente
                    if value and isinstance(value[0], list):
                        df = pd.DataFrame(value)
                    # Se for uma lista simples, converter como uma única coluna
                    else:
                        df = pd.DataFrame(value, columns=[key])
                elif isinstance(value, dict):
                    # Se for um dicionário, converter suas chaves e valores
                    df = pd.DataFrame.from_dict(value, orient='index')
                else:
                    # Para outros tipos, criar um DataFrame com um único valor
                    df = pd.DataFrame([value], columns=[key])
                
                # Salvar como CSV
                output_path = os.path.join(output_dir, f"{key}.csv")
                df.to_csv(output_path, index=False)
                print(f"Dados da chave '{key}' salvos em {output_path}")
                
            except Exception as e:
                print(f"Erro ao processar a chave '{key}': {str(e)}")
                continue
        
        return True
        
    except Exception as e:
        print(f"Erro ao processar o arquivo: {str(e)}")
        return False

def convert_json_to_csv(json_path):
    """
    Converte um arquivo JSON para CSV, tratando estruturas de dados específicas.
    """
    print(json_path)
    # Ler o arquivo JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Criar DataFrame vazio para armazenar os dados
    df = pd.DataFrame()
    print(data)
    
    # Para cada chave no JSON
    for key, value in data.items():
        # Se o valor for uma lista de listas (matriz)
        if isinstance(value, list) and value and isinstance(value[0], list):
            # Extrair o primeiro elemento de cada sublista
            df[key] = [item[0] if isinstance(item, list) else item for item in value]
        else:
            # Caso contrário, usar o valor diretamente
            df[key] = value
    
    # Salvar como CSV
    output_path = json_path.replace('.json', '.csv')
    df.to_csv("all_data_matlab.csv", index=False)
    print(f"Arquivo salvo em: {output_path}")
    return df

def main():
    # Verificar se foi fornecido um arquivo como argumento
    if len(sys.argv) < 2:
        print("Uso: python convert_json.py <caminho_do_arquivo.json>")
        print("Exemplo: python convert_json.py data/all_data_matlab.json")
        return
    
    print("avant")
    
    # Obter o caminho do arquivo do argumento da linha de comando
    json_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/data/all_data_matlab.json"
    
    # Converter o arquivo
    convert_json_to_csv(json_path)

if __name__ == "__main__":
    main()

