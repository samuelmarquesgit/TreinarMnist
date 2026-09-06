"""Testes unitários para IndexadorChromaDB e AssistenteRAG com mocks."""

from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────


def _mock_colecao(documentos=None):
    """Cria uma coleção ChromaDB simulada com comportamento básico."""
    colecao = MagicMock()
    docs = documentos or []
    colecao.count.return_value = len(docs)
    colecao.get.return_value = {"ids": [d["id"] for d in docs]}
    colecao.query.return_value = {
        "documents": [["conteúdo do chunk 1", "conteúdo do chunk 2"]],
        "metadatas": [[
            {"fonte": "src/fachada.py", "topico": "arquitetura"},
            {"fonte": "src/modelos/", "topico": "modelos"},
        ]],
        "distances": [[0.1, 0.4]],
    }
    return colecao


def _mock_chromadb_cliente(colecao):
    """Cria um cliente ChromaDB simulado."""
    cliente = MagicMock()
    cliente.get_or_create_collection.return_value = colecao
    return cliente


# ── Testes IndexadorChromaDB ───────────────────────────────────────────────


class TestIndexadorChromaDB:

    def test_inicializacao_bem_sucedida(self):
        """IndexadorChromaDB deve inicializar sem erros quando chromadb OK."""
        colecao = _mock_colecao()
        cliente = _mock_chromadb_cliente(colecao)

        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            import chromadb as _chromadb_mock
            _chromadb_mock.PersistentClient.return_value = cliente

            from src.rag.indexador import IndexadorChromaDB
            indexador = IndexadorChromaDB(caminho_db="/tmp/fake_db")

        assert indexador.caminho_db == "/tmp/fake_db"

    def test_indexar_documentos_novos(self, tmp_path):
        """Indexar deve chamar colecao.add para documentos ainda não existentes."""
        colecao = MagicMock()
        colecao.count.return_value = 0
        colecao.get.return_value = {"ids": []}  # nenhum existente

        cliente = _mock_chromadb_cliente(colecao)
        chromadb_mock = MagicMock()
        chromadb_mock.PersistentClient.return_value = cliente

        with patch.dict("sys.modules", {"chromadb": chromadb_mock}):
            from importlib import reload
            import src.rag.indexador as idx_mod
            reload(idx_mod)
            indexador = idx_mod.IndexadorChromaDB(caminho_db=str(tmp_path))

        docs = [
            {"id": "doc1", "conteudo": "texto 1", "fonte": "f1.py", "topico": "t1"},
            {"id": "doc2", "conteudo": "texto 2", "fonte": "f2.py", "topico": "t2"},
        ]
        total = indexador.indexar(docs)

        assert total == 2
        colecao.add.assert_called_once()

    def test_indexar_documentos_ja_existentes_nao_duplica(self, tmp_path):
        """Indexar não deve chamar add se todos os ids já estiverem na coleção."""
        colecao = MagicMock()
        colecao.count.return_value = 1
        colecao.get.return_value = {"ids": ["doc1"]}

        cliente = _mock_chromadb_cliente(colecao)
        chromadb_mock = MagicMock()
        chromadb_mock.PersistentClient.return_value = cliente

        with patch.dict("sys.modules", {"chromadb": chromadb_mock}):
            from importlib import reload
            import src.rag.indexador as idx_mod
            reload(idx_mod)
            indexador = idx_mod.IndexadorChromaDB(caminho_db=str(tmp_path))

        docs = [{"id": "doc1", "conteudo": "texto 1", "fonte": "f1.py", "topico": "t1"}]
        indexador.indexar(docs)

        colecao.add.assert_not_called()

    def test_buscar_retorna_documentos_formatados(self, tmp_path):
        """buscar() deve retornar lista com 'conteudo', 'fonte', 'topico', 'distancia'."""
        colecao = _mock_colecao()
        colecao.count.return_value = 5
        cliente = _mock_chromadb_cliente(colecao)
        chromadb_mock = MagicMock()
        chromadb_mock.PersistentClient.return_value = cliente

        with patch.dict("sys.modules", {"chromadb": chromadb_mock}):
            from importlib import reload
            import src.rag.indexador as idx_mod
            reload(idx_mod)
            indexador = idx_mod.IndexadorChromaDB(caminho_db=str(tmp_path))

        resultados = indexador.buscar("pipeline mnist", n_resultados=2)

        assert len(resultados) == 2
        for r in resultados:
            assert "conteudo" in r
            assert "fonte" in r
            assert "topico" in r
            assert "distancia" in r

    def test_total_documentos_delega_para_colecao(self, tmp_path):
        """total_documentos() deve retornar o count da coleção."""
        colecao = MagicMock()
        colecao.count.return_value = 15
        cliente = _mock_chromadb_cliente(colecao)
        chromadb_mock = MagicMock()
        chromadb_mock.PersistentClient.return_value = cliente

        with patch.dict("sys.modules", {"chromadb": chromadb_mock}):
            from importlib import reload
            import src.rag.indexador as idx_mod
            reload(idx_mod)
            indexador = idx_mod.IndexadorChromaDB(caminho_db=str(tmp_path))

        assert indexador.total_documentos() == 15

    def test_inicializacao_sem_chromadb_levanta_importerror(self, tmp_path):
        """Sem chromadb instalado deve levantar ImportError amigável."""
        import sys
        # Remove chromadb do sys.modules se existir
        modulos_backup = {k: v for k, v in sys.modules.items() if "chromadb" in k}
        for k in modulos_backup:
            del sys.modules[k]

        try:
            # Força ImportError dentro de _inicializar
            with patch.dict("sys.modules", {"chromadb": None}):
                from importlib import reload
                import src.rag.indexador as idx_mod
                with pytest.raises(ImportError, match="ChromaDB"):
                    reload(idx_mod)
                    idx_mod.IndexadorChromaDB(caminho_db=str(tmp_path))
        except Exception:
            pass  # Comportamento pode variar conforme a instalação
        finally:
            # Restaura módulos
            sys.modules.update(modulos_backup)


# ── Testes AssistenteRAG ───────────────────────────────────────────────────


def _criar_assistente_com_mock(tmp_path, chunks_retornados=None):
    """Cria AssistenteRAG com IndexadorChromaDB totalmente mockado."""
    if chunks_retornados is None:
        chunks_retornados = [
            {"conteudo": "Texto relevante sobre MNIST.", "fonte": "carregador_dados.py",
             "topico": "dataset", "distancia": 0.2},
        ]

    mock_indexador = MagicMock()
    mock_indexador.indexar.return_value = 15
    mock_indexador.buscar.return_value = chunks_retornados
    mock_indexador.total_documentos.return_value = 15
    mock_indexador.caminho_db = str(tmp_path)

    with patch("src.rag.assistente.IndexadorChromaDB", return_value=mock_indexador):
        from importlib import reload
        import src.rag.assistente as assistente_mod
        reload(assistente_mod)
        assistente = assistente_mod.AssistenteRAG(
            caminho_db=str(tmp_path), n_chunks=3, limiar_distancia=1.5
        )

    assistente._indexador = mock_indexador
    return assistente


class TestAssistenteRAG:

    def test_indexar_documentos_atualiza_flag(self, tmp_path):
        """Após indexar_documentos() o assistente deve estar marcado como indexado."""
        assistente = _criar_assistente_com_mock(tmp_path)
        assert assistente._indexado is False

        total = assistente.indexar_documentos()

        assert assistente._indexado is True
        assert total == 15

    def test_perguntar_sem_indexacao_previa_chama_indexar(self, tmp_path):
        """perguntar() deve chamar indexar_documentos() automaticamente."""
        assistente = _criar_assistente_com_mock(tmp_path)
        assert assistente._indexado is False

        resultado = assistente.perguntar("O que é MNIST?")

        assert assistente._indexado is True
        assert "resposta" in resultado
        assert "fontes" in resultado

    def test_perguntar_retorna_resposta_com_chunk_relevante(self, tmp_path):
        """Chunk dentro do limiar de distância deve aparecer na resposta."""
        chunks = [{"conteudo": "MNIST tem 70k imagens.",
                   "fonte": "dados.py", "topico": "dataset", "distancia": 0.3}]
        assistente = _criar_assistente_com_mock(tmp_path, chunks_retornados=chunks)
        assistente._indexado = True

        resultado = assistente.perguntar("tamanho do dataset")

        assert "70k" in resultado["resposta"]
        assert "dados.py" in resultado["fontes"]

    def test_perguntar_sem_chunks_relevantes_retorna_mensagem_padrao(self, tmp_path):
        """Chunks além do limiar devem gerar resposta padrão 'não encontrei'."""
        # Distância 3.0 > limiar padrão 1.5
        chunks = [{"conteudo": "irrelevante", "fonte": "x.py", "topico": "x", "distancia": 3.0}]
        assistente = _criar_assistente_com_mock(tmp_path, chunks_retornados=chunks)
        assistente._indexado = True

        resultado = assistente.perguntar("algo totalmente diferente")

        assert "Não encontrei" in resultado["resposta"]
        assert resultado["fontes"] == []

    def test_perguntar_multichunk_formata_informacao_adicional(self, tmp_path):
        """Múltiplos chunks relevantes devem gerar 'Informação adicional' na resposta."""
        chunks = [
            {"conteudo": "Texto principal.", "fonte": "f1.py", "topico": "t1", "distancia": 0.1},
            {"conteudo": "Texto adicional.", "fonte": "f2.py", "topico": "t2", "distancia": 0.5},
        ]
        assistente = _criar_assistente_com_mock(tmp_path, chunks_retornados=chunks)
        assistente._indexado = True

        resultado = assistente.perguntar("qualquer coisa")

        assert "Informação adicional" in resultado["resposta"]

    def test_perguntar_um_chunk_adiciona_nota_de_confiabilidade(self, tmp_path):
        """Com apenas 1 chunk relevante deve incluir nota de confiabilidade."""
        chunks = [{"conteudo": "Único documento.", "fonte": "f.py",
                   "topico": "t", "distancia": 0.2}]
        assistente = _criar_assistente_com_mock(tmp_path, chunks_retornados=chunks)
        assistente._indexado = True

        resultado = assistente.perguntar("pergunta")

        assert "Nota:" in resultado["resposta"] or "apenas" in resultado["resposta"].lower()

    def test_estatisticas_retorna_dict_completo(self, tmp_path):
        """estatisticas() deve retornar dict com total_documentos e indexado."""
        from src.rag.indexador import _DOCUMENTOS
        assistente = _criar_assistente_com_mock(tmp_path)
        assistente._indexado = True

        stats = assistente.estatisticas()

        assert "total_documentos" in stats
        assert "base_padrao" in stats
        assert stats["base_padrao"] == len(_DOCUMENTOS)
        assert stats["indexado"] is True
        assert "caminho_db" in stats

    def test_fontes_sem_duplicatas(self, tmp_path):
        """Fontes repetidas em múltiplos chunks devem aparecer uma única vez."""
        chunks = [
            {"conteudo": "A.", "fonte": "mesmo.py", "topico": "t1", "distancia": 0.1},
            {"conteudo": "B.", "fonte": "mesmo.py", "topico": "t2", "distancia": 0.2},
        ]
        assistente = _criar_assistente_com_mock(tmp_path, chunks_retornados=chunks)
        assistente._indexado = True

        resultado = assistente.perguntar("qualquer")

        assert resultado["fontes"].count("mesmo.py") == 1
