from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Carregar as imagens
img1 = Image.open('../data/Vel1.png')
img2 = Image.open('../data/Vel2.png')

# Converter para escala de cinza
gray1 = img1.convert('L')
gray2 = img2.convert('L')

# Redimensionar a segunda imagem
gray2_resized = gray2.resize(gray1.size)

# Calcular MSE
def mse(imageA, imageB):
    # Converter as imagens para arrays numpy
    arrA = np.array(imageA)
    arrB = np.array(imageB)
    # Calcular o erro quadrático médio
    err = np.sum((arrA - arrB) ** 2)
    err /= float(arrA.shape[0] * arrA.shape[1])
    return err

mse_value = mse(gray1, gray2_resized)
print(f'Mean Squared Error: {mse_value}')

# Calcular SSIM
gray1_array = np.array(gray1)
gray2_resized_array = np.array(gray2_resized)

ssim_value, _ = ssim(gray1_array, gray2_resized_array, full=True)
print(f'Structural Similarity Index: {ssim_value}')

# Exibir as imagens
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Imagem 1')
plt.imshow(gray1, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Imagem 2')
plt.imshow(gray2_resized, cmap='gray')

plt.show()
