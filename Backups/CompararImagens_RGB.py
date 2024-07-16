from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Carregar as imagens
img1 = Image.open('../data/Vel1.png')
img2 = Image.open('../data/Vel2.png')

# Redimensionar a segunda imagem para ter as mesmas dimensões da primeira
img2_resized = img2.resize(img1.size)

# Calcular MSE
def mse(imageA, imageB):
    # Converter as imagens para arrays numpy
    arrA = np.array(imageA)
    arrB = np.array(imageB)
    # Calcular o erro quadrático médio
    err = np.sum((arrA - arrB) ** 2)
    err /= float(arrA.shape[0] * arrA.shape[1] * arrA.shape[2])
    return err

mse_value = mse(img1, img2_resized)
print(f'Mean Squared Error: {mse_value}')

# Calcular SSIM
img1_array = np.array(img1)
img2_resized_array = np.array(img2_resized)

# Calcular SSIM para cada canal de cor
ssim_value_r, _ = ssim(img1_array[:, :, 0], img2_resized_array[:, :, 0], full=True)
ssim_value_g, _ = ssim(img1_array[:, :, 1], img2_resized_array[:, :, 1], full=True)
ssim_value_b, _ = ssim(img1_array[:, :, 2], img2_resized_array[:, :, 2], full=True)

# Média dos SSIMs dos três canais
ssim_value = (ssim_value_r + ssim_value_g + ssim_value_b) / 3
print(f'Structural Similarity Index (SSIM): {ssim_value}')

# Exibir as imagens
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Imagem 1')
plt.imshow(img1)

plt.subplot(1, 2, 2)
plt.title('Imagem 2')
plt.imshow(img2_resized)

plt.show()
