"""Testes do guardrail de falsa certeza (TM-041)."""

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import RAIZ_PROJETO, Configuracoes
from api.main import criar_aplicacao
from api.schemas.exemplos import PIXELS_DIGITO_7
from api.services.guardrail_confianca import (
    FAIXAS_MNIST,
    avaliar,
    caracteristicas_entrada,
    entropia_normalizada,
)
from api.services.servico_inferencia import ServicoInferencia
from tests.api.conftest import imagem_da_classe

DIGITO_7 = np.asarray(PIXELS_DIGITO_7, dtype=np.float32)
CERTA = np.array([0.99] + [0.01 / 9] * 9)
UNIFORME = np.full(10, 0.1)


def test_exemplo_real_do_mnist_e_tipico():
    medidas = caracteristicas_entrada(DIGITO_7)
    for nome, (minimo, maximo) in FAIXAS_MNIST.items():
        assert minimo <= medidas[nome] <= maximo, nome


def test_certeza_alta_em_entrada_tipica_e_confiavel_sem_alerta():
    avaliacao = avaliar(CERTA, DIGITO_7, limiar_confianca=0.85, limiar_entropia=0.3)
    assert avaliacao.confiavel is True
    assert avaliacao.alerta_falsa_certeza is False
    assert avaliacao.motivos == ()


def test_certeza_alta_em_ruido_dispara_alerta_de_falsa_certeza():
    ruido = np.random.default_rng(0).random(784).astype(np.float32)
    avaliacao = avaliar(CERTA, ruido, 0.85, 0.3)
    assert avaliacao.alerta_falsa_certeza is True
    assert avaliacao.confiavel is False
    assert "fracao_pixels_ativos" in avaliacao.motivos


def test_distribuicao_uniforme_nao_e_confiavel_nem_alerta():
    avaliacao = avaliar(UNIFORME, DIGITO_7, 0.85, 0.3)
    assert avaliacao.entropia == pytest.approx(1.0)
    assert (avaliacao.confiavel, avaliacao.alerta_falsa_certeza) == (False, False)


def test_digito_deslocado_para_o_canto_e_atipico():
    imagem = np.zeros((28, 28), dtype=np.float32)
    imagem[2:12, 2:6] = 1.0  # traço no canto superior esquerdo
    medidas = caracteristicas_entrada(imagem.ravel())
    assert medidas["deslocamento_centro_massa"] > FAIXAS_MNIST["deslocamento_centro_massa"][1]
    assert "deslocamento_centro_massa" in avaliar(CERTA, imagem.ravel(), 0.85, 0.3).motivos


def test_imagem_saturada_e_atipica():
    motivos = avaliar(CERTA, np.ones(784, dtype=np.float32), 0.85, 0.3).motivos
    assert {"fracao_pixels_ativos", "intensidade_media"} <= set(motivos)


def test_imagem_vazia_nao_quebra_o_calculo():
    assert caracteristicas_entrada(np.zeros(784))["deslocamento_centro_massa"] == 0.0


@pytest.mark.parametrize(
    ("limiar_confianca", "limiar_entropia", "confiavel"),
    [(0.85, 0.3, True), (0.995, 0.3, False), (0.85, 0.001, False)],
)
def test_limiares_controlam_a_confiabilidade(limiar_confianca, limiar_entropia, confiavel):
    assert avaliar(CERTA, DIGITO_7, limiar_confianca, limiar_entropia).confiavel is confiavel


def test_entropia_normalizada_nos_extremos():
    assert entropia_normalizada(np.array([1.0] + [0.0] * 9)) == pytest.approx(0.0, abs=1e-6)
    assert entropia_normalizada(UNIFORME) == pytest.approx(1.0)


# ── Integração com o serviço ──


def test_servico_preenche_alerta_e_confiabilidade(artefato_temporario):
    servico = ServicoInferencia(artefato_temporario.pipeline, artefato_temporario.metadados)
    resultado, _ = servico.prever(imagem_da_classe(2))
    assert isinstance(resultado.confiavel, bool)
    assert isinstance(resultado.alerta_falsa_certeza, bool)
    assert not (resultado.confiavel and resultado.alerta_falsa_certeza)


def test_modelo_real_confia_no_7_e_alerta_no_ruido():
    real = joblib.load(RAIZ_PROJETO / "models" / "modelo_mnist_v1.joblib")
    servico = ServicoInferencia(real["pipeline"], real["metadados"])
    limpo, _ = servico.prever(DIGITO_7.reshape(1, -1))
    assert (limpo.classe_prevista, limpo.confiavel, limpo.alerta_falsa_certeza) == (7, True, False)

    saturada, _ = servico.prever(np.ones((1, 784), dtype=np.float32))
    assert saturada.confiavel is False
    assert saturada.motivos_atipicidade  # entrada fora da faixa do MNIST


def test_lifespan_repassa_os_limiares_da_configuracao(artefato_temporario):
    config = Configuracoes(
        caminho_modelo=artefato_temporario.modelo,
        caminho_checksum=artefato_temporario.checksum,
        caminho_metricas=artefato_temporario.metricas,
        limiar_confianca=0.95,
        limiar_entropia=0.1,
    )
    app = criar_aplicacao(config)
    with TestClient(app):
        assert (app.state.servico.limiar_confianca, app.state.servico.limiar_entropia) == (
            0.95,
            0.1,
        )
