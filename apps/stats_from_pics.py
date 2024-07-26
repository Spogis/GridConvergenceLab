import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import pandas as pd


def gci_from_picture_data(image):
    # Definir a semente de aleatoriedade
    np.random.seed(42)

    # Carregar a imagem usando cv2
    img = image

    # Verificar se a imagem tem um canal Alpha
    has_alpha = img.shape[2] == 4

    # Gerar a máscara de áreas válidas
    if has_alpha:
        alpha_channel = img[:, :, 3]
        valid_mask = alpha_channel != 0
        img_rgb = img[:, :, :3]  # Extrair apenas os canais RGB
    else:
        valid_mask = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        img_rgb = img

    # Transformar a imagem em escala de cinza
    gray_img = rgb2gray(img_rgb)

    # Aplicar a máscara de áreas válidas na imagem em escala de cinza
    valid_gray_values = gray_img[valid_mask]

    # Calcular estatísticas
    mean_gray = np.mean(valid_gray_values)
    std_gray = np.std(valid_gray_values)
    median_gray = np.median(valid_gray_values)
    variance_gray = np.var(valid_gray_values)
    min_gray = np.min(valid_gray_values)
    max_gray = np.max(valid_gray_values)
    range_gray = max_gray - min_gray
    cv_gray = (std_gray / mean_gray) * 100
    quantiles_gray = np.percentile(valid_gray_values, [25, 50, 75])
    skewness_gray = stats.skew(valid_gray_values)
    kurtosis_gray = stats.kurtosis(valid_gray_values)

    # Criar o DataFrame com as estatísticas
    data = {
        'variable': ['Mean', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Coefficient of Variation'],
        'value': [mean_gray, std_gray, variance_gray, skewness_gray, kurtosis_gray, cv_gray]
    }

    df = pd.DataFrame(data)
    return df
