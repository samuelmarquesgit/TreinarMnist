"""Cliente HTTP do painel Streamlit para a API de inferência do Módulo 2.

Ativado pela variável ``MNIST_API_URL`` (ex.: ``http://localhost:8000``). Sem ela, ou com a API
fora do ar, o painel continua usando a inferência local do Módulo 1: a integração nunca quebra
a experiência existente.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import httpx
import numpy as np

VARIAVEL_URL = "MNIST_API_URL"
TIMEOUT_SEGUNDOS = 5.0


def obter_url_api() -> str | None:
    """URL base da API configurada no ambiente, sem a barra final (``None`` se ausente)."""
    url = os.getenv(VARIAVEL_URL, "").strip().rstrip("/")
    return url or None


def _cliente(url: str, transporte: httpx.BaseTransport | None) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=TIMEOUT_SEGUNDOS, transport=transporte)


def api_online(url: str, transporte: httpx.BaseTransport | None = None) -> bool:
    """Consulta ``GET /health`` e diz se a API está pronta para classificar.

    Args:
        url: URL base da API.
        transporte: Transporte HTTP alternativo (usado nos testes).

    Returns:
        ``True`` só se a API responder com ``status == "ok"``.
    """
    try:
        with _cliente(url, transporte) as cliente:
            resposta = cliente.get("/health")
        return resposta.status_code == 200 and resposta.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False


def prever_via_api(
    img_array: np.ndarray, url: str, transporte: httpx.BaseTransport | None = None
) -> dict[str, Any] | None:
    """Envia a imagem para ``POST /v1/predict/imagem`` e devolve o JSON da predição.

    Args:
        img_array: Imagem do canvas ou do upload (cinza ou BGR/BGRA, ``uint8``).
        url: URL base da API.
        transporte: Transporte HTTP alternativo (usado nos testes).

    Returns:
        O corpo ``SaidaPredicao`` da API, ou ``None`` se a API falhar ou recusar a imagem.
    """
    ok, png = cv2.imencode(".png", np.asarray(img_array, dtype=np.uint8))
    if not ok:
        return None
    try:
        with _cliente(url, transporte) as cliente:
            resposta = cliente.post(
                "/v1/predict/imagem",
                files={"arquivo": ("canvas.png", png.tobytes(), "image/png")},
            )
    except httpx.HTTPError:
        return None
    if resposta.status_code != 200:
        return None
    corpo: dict[str, Any] = resposta.json()
    return corpo
