"""Carregamento e verificação do artefato do modelo (executado uma única vez, no ``lifespan``).

Antes de aceitar o arquivo, verifica:

1. que o ``.joblib`` e o ``.sha256`` existem;
2. que o SHA-256 confere (artefato não corrompido nem trocado);
3. que o conteúdo é o ``dict {"pipeline", "metadados"}`` gerado pelo pipeline offline;
4. que os metadados têm as chaves obrigatórias e descrevem 10 classes e 784 features;
5. que o objeto tem ``predict_proba``.

A versão do scikit-learn que serializou o modelo é comparada com a instalada: se forem
diferentes, só registra um alerta (o modelo pode funcionar, mas não há garantia).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn

from api.core.excecoes import ErroArtefatoInvalido
from api.core.registrador_log import obter_logger

logger = obter_logger("services.carregador_artefato")

CHAVES_METADADOS_OBRIGATORIAS = (
    "nome_modelo",
    "versao_modelo",
    "classes",
    "n_features",
    "versao_sklearn",
    "data_treino",
    "hiperparametros",
    "amostras",
)
CLASSES_ESPERADAS = list(range(10))
N_FEATURES_ESPERADO = 784


@dataclass(frozen=True)
class ArtefatoModelo:
    """Modelo carregado em memória com seus metadados e métricas.

    Attributes:
        pipeline: Pipeline do scikit-learn com ``predict_proba``.
        metadados: Metadados gravados pelo pipeline offline.
        metricas: Relatório de métricas (``None`` se o JSON não existir).
        sha256: Hash do arquivo carregado.
    """

    pipeline: Any
    metadados: dict[str, Any]
    metricas: dict[str, Any] | None
    sha256: str


def calcular_sha256(caminho: Path) -> str:
    """Calcula o SHA-256 de um arquivo em blocos de 1 MB.

    Args:
        caminho: Arquivo a verificar.

    Returns:
        Hash hexadecimal (64 caracteres).
    """
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _verificar_checksum(caminho_modelo: Path, caminho_checksum: Path) -> str:
    if not caminho_modelo.is_file():
        raise ErroArtefatoInvalido(f"Artefato do modelo não encontrado: {caminho_modelo.name}")
    if not caminho_checksum.is_file():
        raise ErroArtefatoInvalido(f"Checksum do modelo não encontrado: {caminho_checksum.name}")
    partes = caminho_checksum.read_text(encoding="utf-8").split()
    if not partes or len(partes[0]) != 64:
        raise ErroArtefatoInvalido("Arquivo de checksum em formato inválido.")
    obtido = calcular_sha256(caminho_modelo)
    if obtido != partes[0].lower():
        raise ErroArtefatoInvalido(
            "SHA-256 do artefato não confere: arquivo corrompido ou trocado."
        )
    return obtido


def _validar_conteudo(conteudo: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(conteudo, dict) or not {"pipeline", "metadados"} <= set(conteudo):
        raise ErroArtefatoInvalido("Artefato fora do formato {'pipeline', 'metadados'}.")
    pipeline, metadados = conteudo["pipeline"], conteudo["metadados"]
    faltando = [c for c in CHAVES_METADADOS_OBRIGATORIAS if c not in metadados]
    if faltando:
        raise ErroArtefatoInvalido(f"Metadados incompletos, faltam: {faltando}")
    if list(metadados["classes"]) != CLASSES_ESPERADAS:
        raise ErroArtefatoInvalido("O modelo não classifica os dígitos 0 a 9.")
    if metadados["n_features"] != N_FEATURES_ESPERADO:
        raise ErroArtefatoInvalido(f"O modelo espera {metadados['n_features']} features, não 784.")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ErroArtefatoInvalido("O objeto carregado não tem predict_proba.")
    return pipeline, metadados


def _carregar_metricas(caminho_metricas: Path | None) -> dict[str, Any] | None:
    if caminho_metricas is None or not caminho_metricas.is_file():
        logger.warning("Métricas do modelo não encontradas; GET /v1/modelo ficará indisponível.")
        return None
    try:
        metricas: dict[str, Any] = json.loads(caminho_metricas.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Métricas do modelo ilegíveis (%s); seguindo sem elas.", exc)
        return None
    return metricas


def carregar_artefato(
    caminho_modelo: Path,
    caminho_checksum: Path,
    caminho_metricas: Path | None = None,
) -> ArtefatoModelo:
    """Carrega o artefato do disco depois de verificar integridade e formato.

    Args:
        caminho_modelo: Arquivo ``.joblib``.
        caminho_checksum: Arquivo ``.sha256`` do artefato.
        caminho_metricas: JSON de métricas (opcional).

    Returns:
        :class:`ArtefatoModelo` pronto para criar o serviço de inferência.

    Raises:
        ErroArtefatoInvalido: Se qualquer verificação falhar.
    """
    sha256 = _verificar_checksum(caminho_modelo, caminho_checksum)
    try:
        conteudo = joblib.load(caminho_modelo)
    except Exception as exc:  # pickle corrompido, versão incompatível etc.
        raise ErroArtefatoInvalido(f"Não foi possível desserializar o artefato: {exc}") from exc
    pipeline, metadados = _validar_conteudo(conteudo)

    if metadados["versao_sklearn"] != sklearn.__version__:
        logger.warning(
            "Modelo serializado com scikit-learn %s, mas a versão instalada é %s.",
            metadados["versao_sklearn"],
            sklearn.__version__,
        )
    return ArtefatoModelo(
        pipeline=pipeline,
        metadados=metadados,
        metricas=_carregar_metricas(caminho_metricas),
        sha256=sha256,
    )
