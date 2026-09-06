"""Testes expandidos para a FachadaPipelineIA — cobertura de métodos ainda não cobertos."""

import json
import os
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.fachada import FachadaPipelineIA, ResultadoBenchmark
from src.utilitarios.excecoes import ModeloNaoTreinadoError


# ── ResultadoBenchmark ─────────────────────────────────────────────────────


def test_resultado_benchmark_para_dict_ok():
    """para_dict() deve serializar todos os campos corretamente."""
    rb = ResultadoBenchmark(
        modelo_id="SVM",
        status="ok",
        metricas={"acuracia": 0.97},
        latencia_ms=0.123,
        throughput=1234.56,
    )
    d = rb.para_dict()
    assert d["modelo_id"] == "SVM"
    assert d["status"] == "ok"
    assert d["metricas"]["acuracia"] == pytest.approx(0.97)
    assert d["latencia_ms"] == pytest.approx(0.123, abs=1e-3)
    assert d["throughput_amostras_por_segundo"] == pytest.approx(1234.56, abs=0.01)


def test_resultado_benchmark_para_dict_erro():
    """para_dict() com status de erro deve incluir campo erro."""
    rb = ResultadoBenchmark(modelo_id="Falho", status="erro", erro="Mensagem de falha")
    d = rb.para_dict()
    assert d["status"] == "erro"
    assert d["erro"] == "Mensagem de falha"
    assert d["metricas"] == {}


# ── Métodos de dados ───────────────────────────────────────────────────────


def test_dados_inicializados_falso_antes_de_carregar():
    """dados_inicializados() deve ser False antes de inicializar_dados()."""
    fachada = FachadaPipelineIA()
    assert fachada.dados_inicializados() is False


@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_dados_inicializados_verdadeiro_apos_inicializar(mock_carregar, mock_pre):
    """dados_inicializados() deve ser True após inicializar_dados()."""
    mock_carregar.return_value = (np.ones((100, 784)), np.zeros(100))
    mock_pre.return_value = (
        np.ones((80, 784)), np.ones((20, 784)),
        np.zeros(80), np.zeros(20), "FakeScaler"
    )
    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()
    assert fachada.dados_inicializados() is True


# ── listar_modelos_treinados ───────────────────────────────────────────────


def test_listar_modelos_treinados_vazio():
    """Sem modelos treinados deve retornar lista vazia."""
    fachada = FachadaPipelineIA()
    assert fachada.listar_modelos_treinados() == []


def test_listar_modelos_treinados_retorna_nomes():
    """Deve retornar os nomes de todos os modelos registrados."""
    fachada = FachadaPipelineIA()
    fachada.modelos["ModeloA"] = Mock()
    fachada.modelos["ModeloB"] = Mock()
    lista = fachada.listar_modelos_treinados()
    assert "ModeloA" in lista
    assert "ModeloB" in lista
    assert len(lista) == 2


# ── prever_probabilidades ──────────────────────────────────────────────────


def test_prever_probabilidades_sem_modelo_levanta_erro():
    """prever_probabilidades com modelo não treinado deve levantar ModeloNaoTreinadoError."""
    fachada = FachadaPipelineIA()
    with pytest.raises(ModeloNaoTreinadoError):
        fachada.prever_probabilidades("ModeloInexistente", np.array([[1.0]]))


def test_prever_probabilidades_delega_para_modelo():
    """prever_probabilidades deve delegar ao modelo registrado."""
    fachada = FachadaPipelineIA()
    mock_modelo = Mock()
    esperado = np.array([[0.1] * 10])
    mock_modelo.prever_probabilidades.return_value = esperado
    fachada.modelos["SVM"] = mock_modelo

    X = np.array([[0.5] * 784])
    resultado = fachada.prever_probabilidades("SVM", X)

    np.testing.assert_array_equal(resultado, esperado)
    mock_modelo.prever_probabilidades.assert_called_once_with(X)


# ── obter_estatisticas_dados ───────────────────────────────────────────────


@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_obter_estatisticas_dados_teste(mock_carregar, mock_pre):
    """obter_estatisticas_dados('teste') deve operar sobre X_teste."""
    X_treino = np.ones((80, 784))
    X_teste = np.zeros((20, 784))
    mock_carregar.return_value = (np.ones((100, 784)), np.zeros(100))
    mock_pre.return_value = (X_treino, X_teste, np.zeros(80), np.zeros(20), "FakeScaler")

    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()

    with patch('src.fachada.CalculadorEstatistico') as mock_calc_cls:
        mock_inst = mock_calc_cls.return_value
        mock_inst.estatisticas_descritivas.return_value = {"media": 0.0}
        stats = fachada.obter_estatisticas_dados("teste")

    mock_inst.estatisticas_descritivas.assert_called_once_with(X_teste)
    assert stats == {"media": 0.0}


# ── executar_experimento ───────────────────────────────────────────────────


@patch('src.fachada.calcular_metricas')
@patch('src.fachada.FabricaModelos.criar_modelo')
@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_executar_experimento_retorna_tempo_treino(
    mock_carregar, mock_pre, mock_criar, mock_calc
):
    """executar_experimento() deve incluir 'tempo_treino_segundos' nas métricas."""
    mock_carregar.return_value = (np.ones((100, 784)), np.zeros(100, dtype=int))
    mock_pre.return_value = (
        np.ones((80, 784)), np.ones((20, 784)),
        np.zeros(80, dtype=int), np.zeros(20, dtype=int), "FakeScaler"
    )
    mock_criar.return_value = Mock()
    mock_calc.return_value = {
        "acuracia": 0.92, "precisao": 0.91, "recall": 0.91, "f1": 0.91, "matriz_confusao": []
    }

    fachada = FachadaPipelineIA()
    resultado = fachada.executar_experimento("RegressaoLogistica")

    assert "tempo_treino_segundos" in resultado
    assert resultado["tempo_treino_segundos"] >= 0.0


# ── executar_benchmark ─────────────────────────────────────────────────────


@patch('src.fachada.calcular_metricas')
@patch('src.fachada.FabricaModelos.criar_modelo')
@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_executar_benchmark_gera_arquivo_json(
    mock_carregar, mock_pre, mock_criar, mock_calc, tmp_path
):
    """executar_benchmark() deve criar arquivo JSON em dir_saida."""
    mock_carregar.return_value = (np.ones((100, 784)), np.zeros(100, dtype=int))
    mock_pre.return_value = (
        np.ones((80, 784)), np.ones((20, 784)),
        np.zeros(80, dtype=int), np.zeros(20, dtype=int), "FakeScaler"
    )
    mock_modelo = Mock()
    mock_modelo.prever.return_value = np.zeros(20, dtype=int)
    mock_criar.return_value = mock_modelo
    mock_calc.return_value = {
        "acuracia": 0.90, "precisao": 0.89, "recall": 0.89, "f1": 0.89,
        "matriz_confusao": []
    }

    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()
    dir_saida = str(tmp_path / "benchmarks")

    resultados = fachada.executar_benchmark(["RegressaoLogistica"], dir_saida=dir_saida)

    assert "RegressaoLogistica" in resultados
    assert resultados["RegressaoLogistica"].status == "ok"

    arquivos = os.listdir(dir_saida)
    assert len(arquivos) == 1
    assert arquivos[0].startswith("benchmark_")

    with open(os.path.join(dir_saida, arquivos[0]), encoding="utf-8") as f:
        dados = json.load(f)
    assert "resultados" in dados
    assert "RegressaoLogistica" in dados["resultados"]


@patch('src.fachada.calcular_metricas')
@patch('src.fachada.FabricaModelos.criar_modelo')
@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_executar_benchmark_modelo_que_falha_registra_erro(
    mock_carregar, mock_pre, mock_criar, mock_calc, tmp_path
):
    """Modelo que levanta exceção no treino deve ter status 'erro' no resultado."""
    mock_carregar.return_value = (np.ones((100, 784)), np.zeros(100, dtype=int))
    mock_pre.return_value = (
        np.ones((80, 784)), np.ones((20, 784)),
        np.zeros(80, dtype=int), np.zeros(20, dtype=int), "FakeScaler"
    )
    mock_criar.side_effect = RuntimeError("Modelo indisponível")

    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()
    dir_saida = str(tmp_path / "benchmarks_err")

    resultados = fachada.executar_benchmark(["ModeloBroken"], dir_saida=dir_saida)

    assert resultados["ModeloBroken"].status == "erro"
    assert resultados["ModeloBroken"].erro is not None


@patch('src.fachada.pre_processar_dados')
@patch('src.fachada.carregar_dados_mnist')
def test_executar_benchmark_usa_modelo_ja_treinado(mock_carregar, mock_pre, tmp_path):
    """Modelo já treinado não deve ser retreinado durante o benchmark."""
    mock_carregar.return_value = (np.ones((100, 784)), np.zeros(100, dtype=int))
    mock_pre.return_value = (
        np.ones((80, 784)), np.ones((20, 784)),
        np.zeros(80, dtype=int), np.zeros(20, dtype=int), "FakeScaler"
    )

    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()

    mock_modelo = Mock()
    mock_modelo.prever.return_value = np.zeros(20, dtype=int)
    fachada.modelos["SVM"] = mock_modelo

    with patch('src.fachada.calcular_metricas', return_value={
        "acuracia": 0.95, "precisao": 0.94, "recall": 0.94, "f1": 0.94,
        "matriz_confusao": []
    }):
        dir_saida = str(tmp_path / "bench")
        resultados = fachada.executar_benchmark(["SVM"], dir_saida=dir_saida)

    # treinar_modelo não foi chamado porque SVM já estava em self.modelos
    assert resultados["SVM"].status == "ok"


def test_persistir_benchmark_falha_de_io_nao_levanta_excecao(tmp_path):
    """_persistir_benchmark com diretório sem permissão não deve propagar exceção."""
    fachada = FachadaPipelineIA()
    rb = ResultadoBenchmark(modelo_id="X", status="ok", metricas={"acuracia": 0.9})

    # Passa caminho inválido — OSError deve ser capturado internamente
    fachada._persistir_benchmark({"X": rb}, "2024-01-01T00:00:00Z", "/raiz_invalida_xyz/abc")
    # Se chegou aqui, o erro foi tratado corretamente
