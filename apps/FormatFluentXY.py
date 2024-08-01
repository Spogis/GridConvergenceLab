import os
import pandas as pd

# Diretórios de entrada e saída
input_dir = '../data/data_Petro/Dados_Veloc_Eixos_XYZ_RefiinoCone/XY_file_directory'
output_dir = '../data/data_Petro/Dados_Veloc_Eixos_XYZ_RefiinoCone/xlsx_output_directory'

# Certifique-se de que o diretório de saída exista
os.makedirs(output_dir, exist_ok=True)

# Processa cada arquivo no diretório de entrada
for filename in os.listdir(input_dir):
    input_file_path = os.path.join(input_dir, filename)

    # Ignora diretórios e arquivos que não sejam os esperados
    if not os.path.isfile(input_file_path) or not filename.startswith('veloc_eixo'):
        continue

    # Lê o arquivo de entrada
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Inicializa listas para armazenar os dados de Position e Velocity Magnitude
    position_data = []
    velocity_data = []

    # Varre as linhas e extrai os dados de Position e Velocity Magnitude
    data_section = False
    for line in lines:
        if (line.strip() == '((xy/key/label "rake-7-eixo-x")' or
                line.strip() == '((xy/key/label "rake-8-eixo-y")' or
                line.strip() == '((xy/key/label "rake-6-eixo-z")'):  # Início dos dados
            data_section = True
            continue
        if data_section:
            if line.strip().startswith('('):  # Fim dos dados
                break
            parts = line.strip().split('\t')
            if len(parts) == 2:  # Garante que há exatamente dois valores
                position, velocity = parts
                position_data.append(float(position))
                velocity_data.append(float(velocity))

    # Cria um DataFrame do pandas com duas colunas: x e y
    df = pd.DataFrame({
        'x': position_data,
        'y': velocity_data
    })

    # Ordena o DataFrame pela coluna x em ordem crescente
    df = df.sort_values(by='x')

    # Caminho do arquivo de saída
    output_file_path = os.path.join(output_dir, filename + '.xlsx')

    # Salva o DataFrame em um arquivo Excel
    df.to_excel(output_file_path, index=False)

    print(f'Dados extraídos, ordenados e salvos em {output_file_path}')
