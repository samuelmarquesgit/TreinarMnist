"""Contratos de saída da API: o formato garantido de cada resposta de sucesso.

Os schemas de saída também validam: se o serviço produzir algo incoerente (probabilidades que
não somam 1, classe diferente do argmax, status "ok" sem modelo), o FastAPI falha ao serializar
em vez de entregar uma resposta errada ao cliente. O edital pune Pydantic usado só na entrada.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

N_CLASSES = 10
TOLERANCIA_SOMA = 0.01
PADRAO_SEMVER = r"^\d+\.\d+\.\d+$"

Probabilidade = Annotated[float, Field(ge=0.0, le=1.0)]
Metrica = Annotated[float, Field(ge=0.0, le=1.0)]


class SaidaPredicao(BaseModel):
    """Resultado da classificação de uma imagem."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "classe_prevista": 7,
                    "confianca": 0.9993,
                    "probabilidades": [
                        0.0,
                        0.0001,
                        0.0003,
                        0.0001,
                        0.0,
                        0.0,
                        0.0,
                        0.9993,
                        0.0,
                        0.0002,
                    ],
                    "entropia": 0.0031,
                    "alerta_falsa_certeza": False,
                    "confiavel": True,
                    "nome_modelo": "PerceptronMulticamadas",
                    "versao_modelo": "1.0.0",
                    "tempo_inferencia_ms": 0.42,
                }
            ]
        }
    )

    classe_prevista: int = Field(..., ge=0, le=9, description="Dígito previsto (0 a 9)")
    confianca: Probabilidade = Field(..., description="Maior probabilidade da distribuição")
    probabilidades: list[Probabilidade] = Field(
        ...,
        min_length=N_CLASSES,
        max_length=N_CLASSES,
        description="Probabilidade de cada dígito; a posição na lista é o dígito (índice 0 = '0')",
    )
    entropia: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Entropia de Shannon normalizada (0 = certeza, 1 = uniforme)",
    )
    alerta_falsa_certeza: bool = Field(
        ..., description="Guardrail: confiança muito alta numa entrada atípica"
    )
    confiavel: bool = Field(
        ..., description="Confiança acima do limiar, entropia abaixo do limiar e sem alerta"
    )
    nome_modelo: str = Field(..., min_length=1, max_length=64, description="Modelo usado")
    versao_modelo: str = Field(..., pattern=PADRAO_SEMVER, description="Versão do artefato")
    tempo_inferencia_ms: float = Field(..., ge=0.0, description="Tempo de inferência no servidor")

    @model_validator(mode="after")
    def verificar_coerencia(self) -> SaidaPredicao:
        """Garante que a distribuição soma 1 e que classe e confiança batem com o argmax.

        Returns:
            A própria instância, se coerente.

        Raises:
            ValueError: Se a soma, a classe ou a confiança forem incoerentes.
        """
        soma = sum(self.probabilidades)
        if abs(soma - 1.0) > TOLERANCIA_SOMA:
            raise ValueError(f"As probabilidades somam {soma:.4f}; esperado 1 ± {TOLERANCIA_SOMA}.")
        maior = max(self.probabilidades)
        if self.probabilidades[self.classe_prevista] != maior:
            raise ValueError("classe_prevista deve ser o dígito de maior probabilidade.")
        if abs(self.confianca - maior) > 1e-6:
            raise ValueError("confianca deve ser igual à maior probabilidade.")
        return self


class SaidaLote(BaseModel):
    """Resultado da classificação de várias imagens, na ordem recebida."""

    predicoes: list[SaidaPredicao] = Field(
        ..., min_length=1, max_length=100, description="Uma predição por amostra enviada"
    )
    quantidade: int = Field(..., ge=1, le=100, description="Número de amostras processadas")
    tempo_total_ms: float = Field(..., ge=0.0, description="Tempo da inferência vetorizada")

    @model_validator(mode="after")
    def verificar_quantidade(self) -> SaidaLote:
        """Garante que ``quantidade`` corresponde ao número de predições.

        Returns:
            A própria instância, se coerente.

        Raises:
            ValueError: Se os números divergirem.
        """
        if self.quantidade != len(self.predicoes):
            raise ValueError("quantidade deve ser igual ao número de predições.")
        return self


class SaidaSaude(BaseModel):
    """Estado operacional do serviço (``GET /health``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "modelo_carregado": True,
                    "nome_modelo": "PerceptronMulticamadas",
                    "versao_modelo": "1.0.0",
                    "versao_api": "2.0.0",
                    "tempo_atividade_segundos": 12.4,
                }
            ]
        }
    )

    status: Literal["ok", "degradado"] = Field(
        ..., description="'ok' com o modelo em memória; 'degradado' se o modelo não carregou"
    )
    modelo_carregado: bool = Field(..., description="Se o modelo reside na memória RAM")
    nome_modelo: str | None = Field(None, max_length=64, description="Modelo em memória")
    versao_modelo: str | None = Field(None, pattern=PADRAO_SEMVER, description="Versão do artefato")
    versao_api: str = Field(..., pattern=PADRAO_SEMVER, description="Versão da API")
    tempo_atividade_segundos: float = Field(..., ge=0.0, description="Tempo desde a inicialização")

    @model_validator(mode="after")
    def verificar_status(self) -> SaidaSaude:
        """Garante que ``status`` e ``modelo_carregado`` contam a mesma história.

        Returns:
            A própria instância, se coerente.

        Raises:
            ValueError: Se o status contradisser o estado do modelo.
        """
        if (self.status == "ok") != self.modelo_carregado:
            raise ValueError("status 'ok' exige modelo_carregado=True (e 'degradado', False).")
        return self


class MetricasResumo(BaseModel):
    """Métricas de uma partição, copiadas do relatório offline (a API não recalcula)."""

    acuracia: Metrica = Field(..., description="Acurácia")
    precisao_macro: Metrica = Field(..., description="Precisão média entre as classes")
    recall_macro: Metrica = Field(..., description="Recall médio entre as classes")
    f1_macro: Metrica = Field(..., description="F1-Score médio entre as classes")
    roc_auc_ovr: Metrica = Field(..., description="ROC-AUC um-contra-todos, média macro")
    log_loss: float = Field(..., ge=0.0, description="Perda logarítmica (menor é melhor)")
    n_amostras: int = Field(..., ge=1, description="Tamanho da partição")


class SaidaModelo(BaseModel):
    """Metadados e métricas do modelo em produção (``GET /v1/modelo``)."""

    nome_modelo: str = Field(..., min_length=1, max_length=64, description="Nome do modelo")
    versao_modelo: str = Field(..., pattern=PADRAO_SEMVER, description="Versão do artefato")
    versao_sklearn: str = Field(..., min_length=1, description="scikit-learn usado no treino")
    data_treino: str = Field(..., min_length=1, description="Data do treino (ISO-8601, UTC)")
    hiperparametros: dict[str, Any] = Field(..., description="Hiperparâmetros do classificador")
    amostras: dict[str, int] = Field(..., description="Tamanho das partições de treino")
    metricas_validacao: MetricasResumo = Field(..., description="Métricas na validação")
    metricas_teste: MetricasResumo = Field(..., description="Métricas no teste")
