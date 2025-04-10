import scipy.io
import pandas as pd
import os
import json

if __name__ == "__main__":
    # Carregar o arquivo .mat
    mat = scipy.io.loadmat('C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/ShootOut2012MATLAB/ShootOut2012MATLAB.mat')

    # Imprimir as chaves do arquivo .mat
    print("Chaves disponíveis no arquivo .mat:")
    for key in mat.keys():
        print(f"- {key}")

    # Exibir uma breve descrição dos dados para cada chave
    for key in mat.keys():
        if not key.startswith("__"):  # Ignorar metadados internos
            print(f"\nDescrição dos dados para a chave '{key}':")
            print(mat[key])

    # Criar um dicionário para armazenar todos os dados
    all_data = {}

    # Iterar sobre todas as chaves e adicionar os dados ao dicionário
    for key in mat.keys():
        if not key.startswith("__"):  # Ignorar metadados internos
            data = mat[key]
            # Verificar se os dados podem ser convertidos em uma lista
            if isinstance(data, (list, tuple, pd.DataFrame)) or (hasattr(data, 'shape') and len(data.shape) <= 2):
                all_data[key] = data.tolist() if hasattr(data, 'tolist') else data
            else:
                print(f"Os dados da chave '{key}' não são tabulares e não foram salvos.")

    # Salvar o dicionário como um arquivo JSON
    json_path = 'C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/ShootOut2012MATLAB/all_data_matlab.json'
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_data, json_file, ensure_ascii=False, indent=4)

    print(f"Todos os dados foram salvos em {json_path}")