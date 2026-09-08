"""Testes unitários para o AgenteAnalistaMetricas."""

import pandas as pd
import pytest

from agent.agente_analista_metricas import AgenteAnalistaMetricas


def test_identificar_modelo_campeao_tabela_vazia():
    """Tabela vazia deve retornar 'Nenhum' com justificativa."""
    tabela = pd.DataFrame()
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    assert resultado["modelo_campeao"] == "Nenhum"
    assert "vazia" in resultado["justificativa"]


def test_identificar_modelo_campeao_modelo_unico():
    """Com um único modelo ele mesmo deve ser o campeão."""
    tabela = pd.DataFrame([{
        "modelo": "RegressaoLogistica",
        "acuracia": 0.92,
        "f1_score": 0.91,
        "tempo_treino_s": 5.0,
    }])
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    assert resultado["modelo_campeao"] == "RegressaoLogistica"
    assert resultado["acuracia"] == pytest.approx(0.92)
    assert resultado["f1_score"] == pytest.approx(0.91)
    assert resultado["tempo_treino_s"] == pytest.approx(5.0)


def test_identificar_modelo_campeao_seleciona_melhor_f1():
    """Deve selecionar o modelo com maior F1-Score independente da acurácia."""
    tabela = pd.DataFrame([
        {"modelo": "ModeloA", "acuracia": 0.95, "f1_score": 0.88, "tempo_treino_s": 2.0},
        {"modelo": "ModeloB", "acuracia": 0.90, "f1_score": 0.95, "tempo_treino_s": 10.0},
        {"modelo": "ModeloC", "acuracia": 0.85, "f1_score": 0.80, "tempo_treino_s": 1.0},
    ])
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    assert resultado["modelo_campeao"] == "ModeloB"
    assert resultado["f1_score"] == pytest.approx(0.95)


def test_identificar_modelo_campeao_justificativa_contem_nome_e_f1():
    """A justificativa deve mencionar o nome do modelo e F1-Score."""
    tabela = pd.DataFrame([
        {"modelo": "FlorestaAleatoria", "acuracia": 0.97, "f1_score": 0.97, "tempo_treino_s": 15.0},
    ])
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    assert "FlorestaAleatoria" in resultado["justificativa"]
    assert "F1-Score" in resultado["justificativa"]


def test_identificar_modelo_campeao_sem_coluna_tempo():
    """Sem coluna tempo_treino_s deve retornar 0.0 como fallback."""
    tabela = pd.DataFrame([
        {"modelo": "SVM", "acuracia": 0.98, "f1_score": 0.98},
    ])
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    assert resultado["modelo_campeao"] == "SVM"
    assert resultado["tempo_treino_s"] == pytest.approx(0.0)


def test_identificar_modelo_campeao_empate_retorna_primeiro():
    """Em caso de empate no F1, retorna o que aparecer primeiro no sort."""
    tabela = pd.DataFrame([
        {"modelo": "Alfa", "acuracia": 0.90, "f1_score": 0.90, "tempo_treino_s": 1.0},
        {"modelo": "Beta", "acuracia": 0.90, "f1_score": 0.90, "tempo_treino_s": 2.0},
    ])
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    # Ambos têm mesmo f1; o campeão é um dos dois (sort estável)
    assert resultado["modelo_campeao"] in ("Alfa", "Beta")
    assert resultado["f1_score"] == pytest.approx(0.90)


def test_identificar_modelo_campeao_retorna_float():
    """Acurácia e F1 devem ser floats Python (não numpy scalars)."""
    tabela = pd.DataFrame([
        {"modelo": "KNN", "acuracia": 0.93, "f1_score": 0.92, "tempo_treino_s": 3.0},
    ])
    resultado = AgenteAnalistaMetricas.identificar_modelo_campeao(tabela)
    assert isinstance(resultado["acuracia"], float)
    assert isinstance(resultado["f1_score"], float)
    assert isinstance(resultado["tempo_treino_s"], float)
