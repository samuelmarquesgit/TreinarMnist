import cv2
import numpy as np

def processar_imagem_usuario(imagem_array):
    if len(imagem_array.shape) == 3:
        gray = cv2.cvtColor(imagem_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagem_array
        
    gray = cv2.resize(gray, (28, 28))
    # Inverter (fundo preto, número branco) e normalizar para [0, 1]
    gray = 255 - gray
    gray = gray / 255.0
    return gray.flatten().reshape(1, -1)
