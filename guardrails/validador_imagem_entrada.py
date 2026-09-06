"""Guardrail para validação de imagens de entrada no pipeline de visão computacional."""

import os
from PIL import Image


class ValidadorImagemEntrada:
    """Valida integridade e formatos de arquivos de imagem."""

    EXTENSOES_PERMITIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    TAMANHO_MAXIMO_BYTES = 10 * 1024 * 1024  # 10 MB

    @classmethod
    def validar_arquivo(cls, caminho_arquivo: str) -> bool:
        """Verifica se o caminho existe, é uma extensão válida e tamanho razoável.

        Args:
            caminho_arquivo: Caminho relativo ou absoluto da imagem.

        Returns:
            True se válido, lança exceções descritivas caso contrário.
        """
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(
                f"Arquivo de imagem não encontrado: {caminho_arquivo}")

        _, extensao = os.path.splitext(caminho_arquivo)
        if extensao.lower() not in cls.EXTENSOES_PERMITIDAS:
            raise ValueError(
                f"Formato inválido ({extensao}). Extensões permitidas: {cls.EXTENSOES_PERMITIDAS}")

        tamanho = os.path.getsize(caminho_arquivo)
        if tamanho > cls.TAMANHO_MAXIMO_BYTES:
            raise ValueError(
                f"Arquivo muito grande ({tamanho} bytes). Limite máximo: {cls.TAMANHO_MAXIMO_BYTES} bytes.")

        try:
            with Image.open(caminho_arquivo) as img:
                img.verify()
        except Exception as e:
            raise ValueError(
                f"Arquivo corrompido ou formato de imagem inválido: {e}")

        return True
