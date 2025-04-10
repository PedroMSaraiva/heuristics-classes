import pandas as pd
import json
import numpy as np

def convert_json_to_csv():
    # Caminho do arquivo
    json_path = "C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/data/all_data_matlab.json"
    
    # Ler o arquivo JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Criar um dicionário para armazenar os dados processados
    processed_data = {}
    
    # Encontrar o comprimento máximo das listas
    max_length = max(len(value) if isinstance(value, list) else 1 for value in data.values())
    
    # Caso especial: se 'wl' estiver no formato [[valor1, valor2, ...]], ajuste o max_length
    if 'wl' in data and isinstance(data['wl'], list) and isinstance(data['wl'][0], list):
        max_length = max(max_length, len(data['wl'][0]))
    
    # Processar cada chave do JSON
    for key, value in data.items():
        if key == 'wl':  # Tratar especialmente a coluna 'wl'
            if isinstance(value, list):
                # Para 'wl' no formato [[valor1, valor2, ...]]
                if isinstance(value[0], list):
                    # Pegar TODOS os valores da primeira sublista (não apenas o primeiro)
                    numbers = value[0]
                else:
                    numbers = value
                # Preencher com None até atingir max_length
                processed_data[key] = numbers + [None] * (max_length - len(numbers))
            else:
                processed_data[key] = [value] + [None] * (max_length - 1)
        elif isinstance(value, list) and value and isinstance(value[0], list):
            # Para outras listas de listas, extrair o primeiro elemento
            processed_list = [sublist[0] for sublist in value]
            processed_data[key] = processed_list + [None] * (max_length - len(processed_list))
        elif isinstance(value, list):
            # Para listas simples, preencher até max_length
            processed_data[key] = value + [None] * (max_length - len(value))
        else:
            # Para valores não-lista, criar uma lista repetida
            processed_data[key] = [value] * max_length
    
    # Converter para DataFrame
    df = pd.DataFrame(processed_data)
    
    # Salvar como CSV
    df.to_csv("all_data_matlab.csv", index=False)
    print("Arquivo CSV criado com sucesso!")
    
    # Mostrar as primeiras linhas do DataFrame
    print("\nPrimeiras linhas do DataFrame:")
    print(df.head())
    
    # Mostrar informações sobre o DataFrame
    print("\nInformações do DataFrame:")
    print(df.info())

if __name__ == "__main__":
    convert_json_to_csv() 