"""Testes da criação preguiçosa da função de embedding do RAG (TM-056).

O SentenceTransformer baixa o modelo do HuggingFace ao ser instanciado. Ele deve ser criado só
quando o RAG é usado, nunca no import do módulo.
"""

import importlib
from unittest.mock import MagicMock

import chromadb.utils.embedding_functions as funcoes_embedding
import pytest

import src.modelos.suporte_rag as suporte_rag


@pytest.fixture
def modulo_limpo(monkeypatch):
    """Recarrega o módulo com as funções de embedding do ChromaDB substituídas por mocks."""
    sentence = MagicMock(name="SentenceTransformerEmbeddingFunction")
    padrao = MagicMock(name="DefaultEmbeddingFunction")
    monkeypatch.setattr(funcoes_embedding, "SentenceTransformerEmbeddingFunction", sentence)
    monkeypatch.setattr(funcoes_embedding, "DefaultEmbeddingFunction", padrao)
    modulo = importlib.reload(suporte_rag)
    yield modulo, sentence, padrao
    importlib.reload(suporte_rag)


def test_import_nao_cria_o_embedding(modulo_limpo):
    modulo, sentence, padrao = modulo_limpo
    assert modulo._ef_padrao is None
    sentence.assert_not_called()
    padrao.assert_not_called()


def test_embedding_e_criado_uma_vez_e_reutilizado(modulo_limpo):
    modulo, sentence, _ = modulo_limpo
    primeira = modulo.obter_funcao_embedding()
    segunda = modulo.obter_funcao_embedding()
    assert primeira is segunda
    sentence.assert_called_once_with(model_name="all-MiniLM-L6-v2")
    assert modulo._SENTENCE_TRANSFORMERS_OK is True


@pytest.mark.parametrize("erro", [ImportError("sem sentence_transformers"), ValueError("x")])
def test_fallback_para_o_embedding_padrao_do_chromadb(modulo_limpo, erro):
    modulo, sentence, padrao = modulo_limpo
    sentence.side_effect = erro
    assert modulo.obter_funcao_embedding() is padrao.return_value
    assert modulo._SENTENCE_TRANSFORMERS_OK is False
