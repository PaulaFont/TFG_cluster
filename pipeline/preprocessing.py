import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from skimage import morphology
import utils

def detect_document(image):
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gris = image.copy()
    
    # Aplicar desenfoque para reducir ruido
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Detección de bordes con Canny
    bordes = cv2.Canny(blur, 50, 150)
    
    # Dilatación para cerrar posibles huecos en los bordes
    kernel = np.ones((5,5), np.uint8)
    bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
    
    # Buscar contornos en los bordes
    contornos, _ = cv2.findContours(bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si no hay contornos, probar con una técnica alternativa
    if not contornos:
        # Técnica alternativa: umbralización adaptativa
        adaptivo = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        contornos, _ = cv2.findContours(adaptivo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si todavía no hay contornos, devolver la imagen original
    if not contornos:
        return image
    
    # Filtrar contornos pequeños
    area_min = gris.shape[0] * gris.shape[1] * 0.05  # 5% del área total
    contornos_filtrados = [cnt for cnt in contornos if cv2.contourArea(cnt) > area_min]
    
    if not contornos_filtrados:
        return image
    
    # Encontrar el contorno con el área más grande
    contorno_max = max(contornos_filtrados, key=cv2.contourArea)
    
    # Aproximar el contorno a un polígono
    epsilon = 0.02 * cv2.arcLength(contorno_max, True)
    aprox = cv2.approxPolyDP(contorno_max, epsilon, True)
    
    # Si tenemos 4 puntos, asumimos que es un rectángulo (documento)
    if len(aprox) == 4:
        # Ordenar los puntos para aplicar transformación perspectiva
        pts = aprox.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # El punto superior izquierdo tendrá la suma más pequeña
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        # El punto inferior derecho tendrá la suma más grande
        rect[2] = pts[np.argmax(s)]
        
        # El punto superior derecho tendrá la diferencia más pequeña
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        # El punto inferior izquierdo tendrá la diferencia más grande
        rect[3] = pts[np.argmax(diff)]
        
        # Calcular dimensiones del nuevo rectángulo
        widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Construir conjunto de puntos de destino
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Calcular matriz de transformación
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Aplicar transformación perspectiva
        recortada = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return recortada
    else:
        # Si no es un cuadrilátero, usar el método simple
        x, y, w, h = cv2.boundingRect(contorno_max)
        # Añadir margen
        margen = 10
        x = max(0, x - margen)
        y = max(0, y - margen)
        w = min(image.shape[1] - x, w + 2*margen)
        h = min(image.shape[0] - y, h + 2*margen)
        
        return image[y:y+h, x:x+w]

def binarize_image(file_path):
    # Open file
    img = cv2.imread(file_path)
    img_document = detect_document(img)     # Cut borders from image
    img = cv2.medianBlur(img_document,3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binarization
    denoised = cv2.fastNlMeansDenoising(gray, None, h=5)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_MEAN_C, 21, 20)
    return binary

def basic_binarize_image(file_path, threshold = 180, h_denoising = 3):
    img = cv2.imread(file_path)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=h_denoising)
    img_document = detect_document(denoised)
    gray = cv2.cvtColor(img_document, cv2.COLOR_BGR2GRAY)

    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary

def save_image(image, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, image)

def binarize(image_path):
    binary = binarize_image(image_path)
    binary_2 = basic_binarize_image(image_path)
    return binary, binary_2

def preprocess_all(folder_path, out_path_1, out_path_2):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
    filenames = [filename for filename in os.listdir(folder_path)]

    for image_path, file_name in zip(image_paths, filenames):
        if os.path.exists(image_path):
            print(f"Processing: {image_path}")
            binary_1, binary_2 = binarize(image_path)
            save_image(binary_1, out_path_1, file_name)
            save_image(binary_2, out_path_2, file_name)
        else:
            print(f"File not found: {image_path}")