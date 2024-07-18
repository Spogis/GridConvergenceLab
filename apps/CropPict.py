import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

# Definir as pastas de entrada e saída
input_folder  = "../data/Dados_Veloc_Eixos_XYZ_RefiinoCone/Pics"
output_folder = "../data/Dados_Veloc_Eixos_XYZ_RefiinoCone/EditedPics"

# Função para processar uma única imagem
def process_image(input_image_path, output_image_path):
    # Carregar a imagem
    image = cv2.imread(input_image_path)

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar um filtro gaussiano para suavizar a imagem
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detectar círculos na imagem usando a Transformada de Hough
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0] / 2,
                                        param1=100, param2=30, minRadius=0, maxRadius=0)

    # Verificar se algum círculo foi detectado
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        # Encontrar o círculo mais próximo do centro
        image_center = np.array([gray.shape[1] // 2, gray.shape[0] // 2])
        min_dist = float('inf')
        best_circle = None
        for circle in detected_circles[0, :]:
            center = np.array([circle[0], circle[1]])
            dist = np.linalg.norm(center - image_center)
            if dist < min_dist:
                min_dist = dist
                best_circle = circle

        # Extrair informações do melhor círculo encontrado
        if best_circle is not None:
            center_x, center_y, radius = best_circle

            # Diminuir o raio em 1%
            adjusted_radius = int(radius * 0.99)

            # Criar uma máscara circular
            mask = Image.new('L', (adjusted_radius * 2, adjusted_radius * 2), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, adjusted_radius * 2, adjusted_radius * 2), fill=255)

            # Recortar a imagem ao redor do círculo
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            bbox = (center_x - adjusted_radius, center_y - adjusted_radius, center_x + adjusted_radius,
                    center_y + adjusted_radius)
            result_cropped = image_pil.crop(bbox)

            # Aplicar a máscara circular à imagem recortada
            result_cropped.putalpha(mask)

            # Criar uma nova imagem com fundo transparente
            circular_image = Image.new("RGBA", (adjusted_radius * 2, adjusted_radius * 2), (0, 0, 0, 0))
            circular_image.paste(result_cropped, (0, 0), result_cropped)

            # Salvar a imagem final
            circular_image.save(output_image_path)

            print(f"Imagem editada salva em: {output_image_path}")
    else:
        print(f"Nenhum círculo detectado na imagem: {input_image_path}")


# Função para processar todas as imagens em uma pasta
def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_edited.png")
            process_image(input_image_path, output_image_path)


# Processar todas as imagens na pasta especificada
process_images_in_folder(input_folder, output_folder)

