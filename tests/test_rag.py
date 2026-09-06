import pytest
from src.modelos.suporte_rag import SuporteRAG


def test_inicializacao_rag():
    """Base de conhecimento deve ser populada na criacao."""
    rag = SuporteRAG(em_memoria=True)
    assert rag.colecao.count() > 0


def test_consulta_rag_retorna_resultado():
    """Consulta deve retornar pelo menos 1 documento."""
    rag = SuporteRAG(em_memoria=True)
    resposta = rag.consultar("tamanho imagem", n_resultados=1)
    assert len(resposta) == 1
    assert isinstance(resposta[0], str)
    assert len(resposta[0]) > 0


def test_consulta_rag_n_resultados():
    """n_resultados controla a quantidade de documentos retornados."""
    rag = SuporteRAG(em_memoria=True)
    resposta = rag.consultar("qualquer coisa", n_resultados=3)
    assert len(resposta) == 3


def test_consulta_pergunta_vazia_levanta_erro():
    """Pergunta em branco deve levantar ValueError."""
    rag = SuporteRAG(em_memoria=True)
    with pytest.raises(ValueError, match="A pergunta não pode ser vazia."):
        rag.consultar("")


def test_consulta_rag_persistente(tmp_path):
    """Modo persistente deve criar a colecao e popula-la normalmente."""
    caminho = str(tmp_path / "rag_db")
    rag = SuporteRAG(em_memoria=False, diretorio_banco=caminho)
    assert rag.colecao.count() > 0
    resposta = rag.consultar("mnist dataset", n_resultados=2)
    assert len(resposta) == 2
