import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from scipy.ndimage import binary_erosion
from skimage.morphology import disk
from sklearn.metrics.pairwise import euclidean_distances

# Definir a semente de aleatoriedade
np.random.seed(42)

# Carregar a imagem
img = mpimg.imread('../data/Dados_Veloc_Eixos_XYZ_RefiinoCone/EditedPics/TM-1imp-Cone-Malha_07_1_edited.png')
num_points = 20
margin_percentage = 5  # Ajuste conforme necessário

# Dimensões da imagem
height, width, _ = img.shape

# Função para verificar se o ponto está em uma área válida (Alpha != 0)
def is_valid_point(x, y, image):
    return image[y, x, 3] != 0  # Verifica o canal Alpha

# Gerar a máscara de áreas válidas
alpha_channel = img[:, :, 3]
valid_mask = alpha_channel != 0

# Definir a margem de segurança (em porcentagem)
margin_pixels = int(min(height, width) * margin_percentage / 100)

# Erodir a máscara para criar a margem de segurança interna
eroded_mask = binary_erosion(valid_mask, disk(margin_pixels))

# Gerar uma grade regular de pontos
grid_x, grid_y = np.meshgrid(np.linspace(0, width-1, num=100), np.linspace(0, height-1, num=100))
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T.astype(int)

# Filtrar os pontos válidos com a margem de segurança interna
valid_points = [point for point in grid_points if eroded_mask[point[1], point[0]]]

# Função para selecionar pontos equidistantes
def select_equidistant_points(points, num_points):
    selected_points = [points[0]]
    for _ in range(1, num_points):
        dist = euclidean_distances(points, selected_points).min(axis=1)
        next_point = points[np.argmax(dist)]
        selected_points.append(next_point)
    return selected_points


# Selecionar pontos equidistantes
selected_points = select_equidistant_points(np.array(valid_points), num_points)

# Extrair valores de RGB nos pontos válidos
rgb_values = [img[y, x] for x, y in selected_points]

# Mostrar os pontos na imagem
plt.imshow(img)
plt.scatter(*zip(*selected_points), color='red')
plt.title('Pontos Amostrados na Imagem')
plt.show()

# Exibir os valores RGB
for i, (x, y, rgb) in enumerate(zip(*zip(*selected_points), rgb_values)):
    print(f"Ponto {i+1}: Coordenadas (x, y) = ({x}, {y}), Valor RGB = {rgb}")

# Extrair valores de RGB nos pontos válidos
rgb_values = [img[y, x] for x, y in selected_points]


# Função para calcular estatísticas
def calculate_statistics(rgb_values):
    rgb_array = np.array(rgb_values)
    stats = {
        'Mean_R': np.mean(rgb_array[:, 0]),
        'Median_R': np.median(rgb_array[:, 0]),
        'Standard Deviation_R': np.std(rgb_array[:, 0]),
        'Variance_R': np.var(rgb_array[:, 0]),
        'Covariance_R': np.cov(rgb_array[:, 0]),
        'Mean_G': np.mean(rgb_array[:, 1]),
        'Median_G': np.median(rgb_array[:, 1]),
        'Standard Deviation_G': np.std(rgb_array[:, 1]),
        'Variance_G': np.var(rgb_array[:, 1]),
        'Covariance_G': np.cov(rgb_array[:, 1]),
        'Mean_B': np.mean(rgb_array[:, 2]),
        'Median_B': np.median(rgb_array[:, 2]),
        'Standard Deviation_B': np.std(rgb_array[:, 2]),
        'Variance_B': np.var(rgb_array[:, 2]),
        'Covariance_B': np.cov(rgb_array[:, 2])
    }

    return stats


# Calcular estatísticas para os valores de RGB
stats = calculate_statistics(rgb_values)

# Criar o DataFrame
data = {
    'Statistic': list(stats.keys()),
    'Coarse': list(stats.values())
}

df = pd.DataFrame(data)

# Exibir o DataFrame
df.to_excel('../data/pic_for_gci.xlsx', index=False)