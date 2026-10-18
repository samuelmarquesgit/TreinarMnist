"""Rota de diagnóstico que força um erro 500 (só existe com ``MNIST_API_HABILITAR_DIAGNOSTICO=true``).

Não há forma legítima de provocar um 500 pela API: todo erro previsto vira 400, 422 ou 503. Para
demonstrar no vídeo e testar na coleção Postman que uma falha inesperada **não derruba a API** e
devolve um corpo padronizado sem stack trace, esta rota lança uma exceção proposital.

Desligada por padrão, fora do Swagger (``include_in_schema=False``) e nunca registrada em produção.
"""

from __future__ import annotations

from fastapi import APIRouter

roteador = APIRouter(prefix="/v1/_diagnostico", include_in_schema=False)


class FalhaProposital(RuntimeError):
    """Exceção lançada de propósito para exercitar o handler de erro 500."""


@roteador.get("/erro-interno")
def forcar_erro_interno() -> None:
    """Lança uma exceção não tratada para provar o handler de 500."""
    raise FalhaProposital("Falha proposital da rota de diagnóstico (detalhe interno, não exposto).")
