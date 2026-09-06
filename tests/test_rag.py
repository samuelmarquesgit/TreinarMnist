"""Testes do SuporteRAG — embedding mockado para dispensar acesso à rede."""

import pytest
import numpy as np


# ── Embedding fake compatível com a API interna do ChromaDB ────────────────

class _EmbeddingFake:
    """Embedding determinístico baseado em hash — sem rede, sem modelo externo."""

    _DIM = 64

    def name(self) -> str:  # ChromaDB >= 0.6 exige este método
        return "fake"

    def _vetorizar(self, textos):
        resultado = []
        for texto in textos:
            seed = sum(ord(c) for c in str(texto)) % (2 ** 31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._DIM)
            v = v / (np.linalg.norm(v) + 1e-9)
            resultado.append(v.tolist())
        return resultado

    def __call__(self, input):  # noqa: A002  — add / index
        return self._vetorizar(input)

    def embed_query(self, input):  # noqa: A002  — query
        return self._vetorizar(input)


_FAKE_EF = _EmbeddingFake()


@pytest.fixture(autouse=True)
def _patch_ef(monkeypatch):
    """Substitui _ef_padrao pelo embedding fake em todos os testes."""
    monkeypatch.setattr("src.modelos.suporte_rag._ef_padrao", _FAKE_EF)


# ── Importação tardia (após o patch ser aplicado) ──────────────────────────

@pytest.fixture()
def rag_em_memoria():
    from src.modelos.suporte_rag import SuporteRAG
    return SuporteRAG(em_memoria=True)


# ── Testes ─────────────────────────────────────────────────────────────────

def test_inicializacao_rag(rag_em_memoria):
    """Base de conhecimento deve ser populada na criação."""
    assert rag_em_memoria.colecao.count() > 0


def test_consulta_rag_retorna_resultado(rag_em_memoria):
    """Consulta deve retornar exatamente 1 documento."""
    resposta = rag_em_memoria.consultar("tamanho imagem", n_resultados=1)
    assert len(resposta) == 1
    assert isinstance(resposta[0], str)
    assert len(resposta[0]) > 0


def test_consulta_rag_n_resultados(rag_em_memoria):
    """n_resultados controla a quantidade de documentos retornados."""
    resposta = rag_em_memoria.consultar("qualquer coisa", n_resultados=3)
    assert len(resposta) == 3


def test_consulta_pergunta_vazia_levanta_erro(rag_em_memoria):
    """Pergunta em branco deve levantar ValueError."""
    with pytest.raises(ValueError, match="A pergunta não pode ser vazia."):
        rag_em_memoria.consultar("")


def test_consulta_rag_persistente(tmp_path):
    """Modo persistente deve criar a coleção e populá-la normalmente."""
    from src.modelos.suporte_rag import SuporteRAG
    rag = SuporteRAG(em_memoria=False, diretorio_banco=str(tmp_path / "rag_db"))
    assert rag.colecao.count() > 0
    resposta = rag.consultar("mnist dataset", n_resultados=2)
    assert len(resposta) == 2
