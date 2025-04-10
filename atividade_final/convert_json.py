import pandas as pd
import json
import os

def convert_json_to_csv(json_path, output_dir):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Criar o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre cada chave no JSON e salvar como CSV
    for key, value in data.items():
        try:
            # Tentar converter os dados em um DataFrame
            df = pd.DataFrame(value)
            csv_path = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Dados da chave '{key}' salvos em {csv_path}")
        except ValueError as e:
            print(f"Erro ao converter dados da chave '{key}': {e}")

if __name__ == "__main__":
    json_path = 'C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/ShootOut2012MATLAB/all_data.json'
    output_dir = 'C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/ShootOut2012MATLAB'
    convert_json_to_csv(json_path, output_dir)

