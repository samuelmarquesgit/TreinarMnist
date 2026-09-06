from unittest.mock import patch
import pytest
import json
from src.banco_dados.conexao_postgres import ConexaoPostgres, Experimento
try:
    from src.banco_dados.conexao_mongodb import ConexaoMongoDB
    _MONGO_DISPONIVEL = True
except Exception:
    ConexaoMongoDB = None  # type: ignore
    _MONGO_DISPONIVEL = False


def test_conexao_postgres_context_manager():
    # Usamos banco em memoria para teste
    db = ConexaoPostgres(url='sqlite:///:memory:')
    with db.obter_sessao() as sessao:
        novo_exp = Experimento(
            modelo='Regressao Teste',
            acuracia=0.99,
            tempo_treino=1.5)
        sessao.add(novo_exp)

    with db.obter_sessao() as sessao2:
        exp_salvo = sessao2.query(Experimento).first()
        assert exp_salvo is not None
        assert exp_salvo.modelo == 'Regressao Teste'
        assert exp_salvo.acuracia == 0.99


@pytest.mark.skipif(not _MONGO_DISPONIVEL, reason="pymongo indisponivel neste ambiente")
def test_conexao_mongodb_fallback_local(tmp_path, monkeypatch):
    # Força salvamento na pasta temporária para não sujar o reports do projeto
    monkeypatch.chdir(tmp_path)

    mongo = ConexaoMongoDB(uri=None)
    assert mongo.usar_local is True

    dados = {"matriz": [[10, 2], [3, 15]]}
    mongo.salvar_artefato("teste_matriz", dados)

    arquivo_salvo = tmp_path / "reports" / "teste_matriz.json"
    assert arquivo_salvo.exists()

    with open(arquivo_salvo, 'r') as f:
        carregado = json.load(f)
        assert carregado["matriz"] == [[10, 2], [3, 15]]


def test_conexao_postgres_multiplos_experimentos_e_tipagem():
    db = ConexaoPostgres(url='sqlite:///:memory:')
    with db.obter_sessao() as sessao:
        # Forcamos casting explicito no teste para ver se o schema converte
        exp1 = Experimento(
            modelo='Mod1',
            acuracia=float("0.85"),
            tempo_treino=1.0)
        exp2 = Experimento(modelo='Mod2', acuracia=0.90, tempo_treino=2.0)
        sessao.add_all([exp1, exp2])

    with db.obter_sessao() as sessao2:
        todos = sessao2.query(Experimento).order_by(
            Experimento.acuracia.asc()).all()
        assert len(todos) == 2

        # O SQLite/SQLAlchemy pode trazer como float, entao validamos se a
        # tipagem ta nativa
        assert isinstance(todos[0].acuracia, float)
        assert isinstance(todos[1].acuracia, float)

        assert todos[1].modelo == 'Mod2'


def test_conexao_postgres_excecao_rollback():
    """Valida se o context manager faz o rollback adequadamente em caso de erro interno."""
    db = ConexaoPostgres(url='sqlite:///:memory:')

    with pytest.raises(Exception, match="Erro Forcado"):
        with db.obter_sessao():
            # O proprio context manager intercepta o erro interno, faz rollback
            # e da raise
            raise Exception("Erro Forcado")


def test_conexao_postgres_cria_diretorio_reports():
    """Valida a criacao do diretorio fallback do sqlite local."""
    import os
    from src.banco_dados.conexao_postgres import ConexaoPostgres

    # Executa sem falhar, deve criar a pasta 'reports' se iniciada com o caminho relativo
    ConexaoPostgres(url='sqlite:///reports/banco_local_test.db')
    assert os.path.exists('reports')


@pytest.mark.skipif(not _MONGO_DISPONIVEL, reason='pymongo indisponivel')
@patch('src.banco_dados.conexao_mongodb.MongoClient')
def test_conexao_mongodb_remoto_sucesso(mock_mongo_client):
    # Simula cliente remoto
    mongo = ConexaoMongoDB(uri="mongodb://fake:27017")

    assert mongo.usar_local is False
    assert mongo.client is not None
    assert mongo.db is not None

    mongo.salvar_artefato("teste_remoto", {"dados": 123})
    # Valida que inseriu no db chamando insert_one na colecao
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.return_value.insert_one.assert_called_once()


@pytest.mark.skipif(not _MONGO_DISPONIVEL, reason='pymongo indisponivel')
@patch('src.banco_dados.conexao_mongodb.MongoClient')
def test_conexao_mongodb_fallback_excecao(
        mock_mongo_client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Força ServerSelectionTimeoutError no client
    try:
        from pymongo.errors import ServerSelectionTimeoutError
    except Exception:
        pytest.skip("pymongo indisponivel")
    mock_mongo_client.side_effect = ServerSelectionTimeoutError(
        "Timeout simulado")

    mongo = ConexaoMongoDB(uri="mongodb://fake:27017")

    # Deve fazer fallback pro local
    assert mongo.usar_local is True

    mongo.salvar_artefato("teste_timeout", {"dados": 123})
    assert (tmp_path / "reports" / "teste_timeout.json").exists()
