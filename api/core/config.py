"""Configuração da API via variáveis de ambiente (prefixo ``MNIST_API_``) com pydantic-settings.

Todos os valores têm padrão funcional: a API sobe sem nenhum ``.env``. Caminhos relativos são
resolvidos a partir da **raiz do projeto**, nunca do diretório de trabalho, e nenhum caminho
absoluto da máquina do desenvolvedor aparece no código (edital, item 5.1).

Exemplo de sobrescrita::

    MNIST_API_LIMIAR_CONFIANCA=0.9 uvicorn api.main:app
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ_PROJETO = Path(__file__).resolve().parents[2]


class Configuracoes(BaseSettings):
    """Parâmetros de execução da API, validados na inicialização.

    Attributes:
        caminho_modelo: Artefato ``.joblib`` com o Pipeline e os metadados.
        caminho_checksum: Arquivo ``.sha256`` do artefato.
        caminho_metricas: JSON de métricas gerado pelo pipeline offline.
        versao_api: Versão semântica da API, exibida no Swagger e no ``/health``.
        limiar_confianca: Probabilidade mínima para marcar uma predição como confiável.
        limiar_entropia: Entropia normalizada máxima para marcar uma predição como confiável.
        tamanho_max_upload_bytes: Tamanho máximo da imagem enviada em ``/v1/predict/imagem``.
        max_pixels_imagem: Resolução máxima (largura × altura) da imagem enviada, conferida no
            cabeçalho antes de decodificar, para barrar "bombas de descompressão".
        tamanho_max_lote: Número máximo de amostras em ``/v1/predict/lote``.
        habilitar_diagnostico: Registra a rota que força um erro 500 (só para testes e Postman).
        nivel_log: Nível do logger da API.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNIST_API_",
        env_file=RAIZ_PROJETO / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    caminho_modelo: Path = Field(
        default=Path("models/modelo_mnist_v1.joblib"),
        description="Artefato do modelo (relativo à raiz do projeto ou absoluto)",
    )
    caminho_checksum: Path = Field(
        default=Path("models/modelo_mnist_v1.sha256"),
        description="Checksum SHA-256 do artefato",
    )
    caminho_metricas: Path = Field(
        default=Path("models/metricas_modelo_v1.json"),
        description="Métricas de validação e teste do modelo",
    )
    versao_api: str = Field(
        default="2.0.0", pattern=r"^\d+\.\d+\.\d+$", description="Versão semântica da API"
    )
    limiar_confianca: float = Field(
        default=0.85, gt=0.0, le=1.0, description="Confiança mínima de uma predição confiável"
    )
    limiar_entropia: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Entropia normalizada máxima (0 a 1)"
    )
    tamanho_max_upload_bytes: int = Field(
        default=2 * 1024 * 1024, gt=0, le=10 * 1024 * 1024, description="Limite do upload"
    )
    max_pixels_imagem: int = Field(
        default=25_000_000,
        ge=784,
        le=100_000_000,
        description="Resolução máxima da imagem enviada (largura × altura)",
    )
    tamanho_max_lote: int = Field(default=100, ge=1, le=1000, description="Limite do lote")
    habilitar_diagnostico: bool = Field(
        default=False, description="Expõe a rota de erro 500 proposital (nunca em produção)"
    )
    nivel_log: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Nível do logger da API",
    )

    @field_validator("caminho_modelo", "caminho_checksum", "caminho_metricas")
    @classmethod
    def resolver_a_partir_da_raiz(cls, caminho: Path) -> Path:
        """Converte caminhos relativos em absolutos a partir da raiz do projeto.

        Args:
            caminho: Caminho informado (padrão ou variável de ambiente).

        Returns:
            Caminho absoluto, independente do diretório em que o Uvicorn foi iniciado.
        """
        return caminho if caminho.is_absolute() else (RAIZ_PROJETO / caminho).resolve()


@lru_cache
def obter_configuracoes() -> Configuracoes:
    """Lê e valida a configuração uma única vez por processo.

    Returns:
        Instância compartilhada de :class:`Configuracoes`.
    """
    return Configuracoes()
