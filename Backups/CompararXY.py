import pandas as pd

# Carregar os dados dos arquivos CSV
df1 = pd.read_excel('Curva1.xlsx')
df2 = pd.read_excel('Curva2.xlsx')

import numpy as np
from scipy.interpolate import interp1d

# Definir o intervalo comum de x para interpolação (100 pontos)
x_common = np.linspace(min(df1['x'].min(), df2['x'].min()), max(df1['x'].max(), df2['x'].max()), 100)

# Funções de interpolação spline
spline1 = interp1d(df1['x'], df1['y'], kind='cubic', fill_value="extrapolate")
spline2 = interp1d(df2['x'], df2['y'], kind='cubic', fill_value="extrapolate")

# Calcular os valores interpolados
y1_interp = spline1(x_common)
y2_interp = spline2(x_common)

# Criar novos DataFrames com os valores interpolados
df1_interp = pd.DataFrame({'x': x_common, 'y': y1_interp})
df2_interp = pd.DataFrame({'x': x_common, 'y': y2_interp})

from sklearn.metrics.pairwise import cosine_similarity

# Calcular a similaridade cosseno
similarity = cosine_similarity([df1_interp['y']], [df2_interp['y']])
print(f'Similaridade cosseno: {similarity[0, 0]}')

from scipy.spatial.distance import euclidean

# Calcular a distância euclidiana média
distance = euclidean(df1_interp['y'], df2_interp['y'])
print(f'Distância Euclidiana: {distance}')

import matplotlib.pyplot as plt

plt.plot(df1['x'], df1['y'], 'o', label='Dados Originais Arquivo 1')
plt.plot(df2['x'], df2['y'], 'o', label='Dados Originais Arquivo 2')
plt.plot(df1_interp['x'], df1_interp['y'], '-', label='Interpolado Arquivo 1')
plt.plot(df2_interp['x'], df2_interp['y'], '-', label='Interpolado Arquivo 2')
plt.legend()
plt.show()
