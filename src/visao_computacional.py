import cv2
import numpy as np


def processar_imagem_usuario(imagem_array: np.ndarray) -> np.ndarray:
    """
    Processa uma imagem enviada pelo usuário para o padrão ideal do MNIST.
    O pipeline inclui:
    1. Conversão para escala de cinza e inversão (fundo preto, dígito branco).
    2. Encontrar o Bounding Box (BBox) do dígito.
    3. Recortar o dígito.
    4. Redimensionar preservando o aspect ratio para caber num quadro de 20x20.
    5. Centralizar (por centro de massa) em uma tela preta de 28x28.
    6. Normalizar os pixels para o intervalo [0.0, 1.0].

    Args:
        imagem_array (np.ndarray): Imagem original recebida da UI (pode ser RGB ou Gray).

    Returns:
        np.ndarray: Vetor achatado (1, 784) pronto para inferência pelo modelo.
    """
    # 1. Escala de cinza
    if len(imagem_array.shape) == 3:
        gray = cv2.cvtColor(imagem_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagem_array.copy()

    # Inverter cores (assumindo que o usuário desenhou preto no branco, que é o comum em UIs)
    # Valores próximos a 255 viram 0 (fundo), valores próximos a 0 viram 255 (traço)
    # Dica: Se o fundo já for preto, essa inversão não deve ocorrer, mas o padrão do Canvas
    # normalmente é fundo branco. Vamos usar a média para garantir.
    if gray.mean() > 127:
        gray = 255 - gray

    # 2. Encontrar BBox
    # Aplica um threshold para isolar o dígito
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        # Se estiver em branco, retorna array zerado
        return np.zeros((1, 784), dtype=np.float32)

    # Pegar a maior área ou unir todos os contornos
    c = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 3. Recortar
    recorte = gray[y:y + h, x:x + w]

    # 4. Redimensionar para 20x20 mantendo aspect ratio
    max_dim = max(w, h)
    if max_dim == 0:
        return np.zeros((1, 784), dtype=np.float32)

    # Escala
    fator = 20.0 / max_dim
    new_w = int(w * fator)
    new_h = int(h * fator)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    redimensionado = cv2.resize(recorte, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Centralizar em 28x28 (Padding)
    # Por simplicidade da implementação geométrica (Center of Mass no MNIST original):
    tela = np.zeros((28, 28), dtype=np.uint8)

    # Calcula centro de massa do redimensionado
    M = cv2.moments(redimensionado)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = new_w // 2, new_h // 2

    # Posicionamento compensando o centro de massa para coincidir com o centro (14, 14)
    shift_x = 14 - cX
    shift_y = 14 - cY

    # Prevenir bounds exception ao colar
    start_y = max(0, shift_y)
    start_x = max(0, shift_x)

    end_y = start_y + new_h
    end_x = start_x + new_w

    # Ajuste fino em casos extremos onde o crop passaria do tamanho 28x28
    if end_y > 28:
        end_y = 28
        redimensionado = redimensionado[:(28 - start_y), :]
    if end_x > 28:
        end_x = 28
        redimensionado = redimensionado[:, :(28 - start_x)]

    tela[start_y:end_y, start_x:end_x] = redimensionado

    # 6. Normalizar [0, 1] e formatar
    tela_norm = tela.astype(np.float32) / 255.0
    return tela_norm.flatten().reshape(1, -1)
