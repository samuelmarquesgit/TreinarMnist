import pytest
import os
import json
from src.banco_dados.conexao_postgres import ConexaoPostgres, Experimento
from src.banco_dados.conexao_mongodb import ConexaoMongoDB

def test_conexao_postgres_context_manager():
    # Usamos banco em memoria para teste
    db = ConexaoPostgres(url='sqlite:///:memory:')
    with db.obter_sessao() as sessao:
        novo_exp = Experimento(modelo='Regressao Teste', acuracia=0.99, tempo_treino=1.5)
        sessao.add(novo_exp)
        
    with db.obter_sessao() as sessao2:
        exp_salvo = sessao2.query(Experimento).first()
        assert exp_salvo is not None
        assert exp_salvo.modelo == 'Regressao Teste'
        assert exp_salvo.acuracia == 0.99

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
