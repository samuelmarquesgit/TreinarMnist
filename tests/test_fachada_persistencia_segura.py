"""Regressão de segurança: _persistir_modelo não grava fora de artifacts/modelos (SonarCloud S2083)."""

import os

import pytest

import src.fachada as fachada_modulo
from src.fachada import FachadaPipelineIA


@pytest.fixture
def pasta_modelos(tmp_path, monkeypatch):
    """Redireciona a pasta de modelos para um diretório temporário isolado."""
    destino = tmp_path / "artifacts" / "modelos"
    monkeypatch.setattr(fachada_modulo, "_DIR_MODELOS", str(destino))
    return destino


@pytest.mark.parametrize("nome", ["RegressaoLogistica", "KNN", "PerceptronMulticamadas"])
def test_nome_registrado_gera_caminho_dentro_da_pasta(pasta_modelos, nome):
    caminho = FachadaPipelineIA._caminho_seguro_modelo(nome, ".joblib")
    assert os.path.commonpath([os.path.realpath(pasta_modelos), caminho]) == os.path.realpath(
        pasta_modelos
    )
    assert caminho.endswith(f"{nome}.joblib")


@pytest.mark.parametrize(
    "nome",
    [
        "../../escapou",
        "../modelos_falsos",
        "KNN/../../escapou",
        "/tmp/absoluto",
        "C:\\Windows\\escapou",
        "..\\..\\escapou",
        "ModeloInexistente",
        "",
    ],
)
def test_nome_malicioso_ou_desconhecido_e_recusado(pasta_modelos, nome):
    with pytest.raises(ValueError):
        FachadaPipelineIA._caminho_seguro_modelo(nome, ".joblib")


def test_persistir_com_nome_malicioso_nao_grava_nada(pasta_modelos, tmp_path):
    fachada = FachadaPipelineIA()
    with pytest.raises(ValueError, match="não registrado"):
        fachada._persistir_modelo("../../escapou", object())
    assert not any(tmp_path.rglob("*.joblib"))
    assert not pasta_modelos.exists()


def test_persistir_com_nome_valido_grava_na_pasta(pasta_modelos):
    FachadaPipelineIA()._persistir_modelo("KNN", {"modelo": "falso"})
    assert (pasta_modelos / "KNN.joblib").exists()


def test_extensao_pickle_tambem_e_protegida(pasta_modelos):
    assert FachadaPipelineIA._caminho_seguro_modelo("KNN", ".pkl").endswith("KNN.pkl")
    with pytest.raises(ValueError):
        FachadaPipelineIA._caminho_seguro_modelo("../KNN", ".pkl")
