"""Contrato único de erro: 400, 422, 500 e 503 respondem sempre no mesmo formato.

Um cliente (ou a coleção Postman) trata qualquer falha lendo os mesmos campos. A resposta nunca
contém stack trace; o ``id_requisicao`` permite achar o erro completo no log do servidor.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DetalheErro(BaseModel):
    """Um problema específico dentro do erro (ex.: um campo inválido)."""

    campo: str = Field(..., description="Caminho do campo, ex.: 'pixels.3' ou 'amostras.1.pixels'")
    mensagem: str = Field(..., min_length=1, description="O que está errado, em pt-BR")
    tipo: str = Field(..., min_length=1, description="Código do problema, ex.: 'less_than_equal'")
    valor_recebido: Any = Field(None, description="Valor enviado (omitido se for muito grande)")


class SaidaErro(BaseModel):
    """Corpo padronizado de todas as respostas de erro da API."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "codigo": 422,
                    "erro": "VALIDACAO_FALHOU",
                    "mensagem": "Os dados enviados não respeitam o contrato da API.",
                    "detalhes": [
                        {
                            "campo": "pixels.0",
                            "mensagem": "O valor deve ser menor ou igual a 1",
                            "tipo": "less_than_equal",
                            "valor_recebido": 2.0,
                        }
                    ],
                    "id_requisicao": "3f2b9c1e-6a4d-4e8f-9b7a-1c2d3e4f5a6b",
                }
            ]
        }
    )

    codigo: int = Field(..., ge=400, le=599, description="Status HTTP repetido no corpo")
    erro: str = Field(
        ..., pattern=r"^[A-Z][A-Z0-9_]*$", max_length=64, description="Código estável do erro"
    )
    mensagem: str = Field(..., min_length=1, max_length=500, description="Explicação em pt-BR")
    detalhes: list[DetalheErro] | None = Field(None, description="Problemas específicos, se houver")
    id_requisicao: str = Field(
        ..., min_length=1, max_length=64, description="Identificador para rastrear no log"
    )


def _resposta(descricao: str) -> dict[str, Any]:
    return {"model": SaidaErro, "description": descricao}


RESPOSTAS_ERRO_PREDICAO: dict[int | str, dict[str, Any]] = {
    400: _resposta("Entrada válida no formato, mas impossível de classificar (ex.: imagem vazia)"),
    422: _resposta("Dados fora do contrato: tipo errado, fora da faixa ou tamanho incorreto"),
    500: _resposta("Falha inesperada no servidor (sem detalhes internos na resposta)"),
    503: _resposta("Modelo não está carregado em memória"),
}
"""Respostas de erro documentadas no Swagger para as rotas de inferência."""
