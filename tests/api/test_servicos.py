"""Testes do carregador de artefato e do serviço de inferência, sem FastAPI (TM-033)."""

import ast
import json

import joblib
import numpy as np
import pytest

from api.core.config import RAIZ_PROJETO
from api.core.excecoes import (
    ErroApiMnist,
    ErroArtefatoInvalido,
    ErroEntradaInvalida,
    ErroModeloIndisponivel,
)
from api.services.carregador_artefato import calcular_sha256, carregar_artefato
from api.services.servico_inferencia import ServicoInferencia, entropia_normalizada
from tests.api.conftest import imagem_da_classe

# ── Exceções de domínio ──


@pytest.mark.parametrize(
    ("classe", "http", "codigo"),
    [
        (ErroEntradaInvalida, 400, "ENTRADA_INVALIDA"),
        (ErroModeloIndisponivel, 503, "MODELO_INDISPONIVEL"),
        (ErroArtefatoInvalido, 500, "ARTEFATO_INVALIDO"),
    ],
)
def test_excecoes_carregam_status_e_codigo(classe, http, codigo):
    erro = classe("algo", detalhes=[{"campo": "x"}])
    assert isinstance(erro, ErroApiMnist)
    assert (erro.codigo_http, erro.codigo_erro, erro.mensagem) == (http, codigo, "algo")
    assert erro.detalhes == [{"campo": "x"}]


# ── Carregador de artefato ──


def test_carrega_artefato_valido_com_metricas(artefato_temporario):
    a = artefato_temporario
    artefato = carregar_artefato(a.modelo, a.checksum, a.metricas)
    assert artefato.metadados["nome_modelo"] == "ModeloTeste"
    assert artefato.metricas["teste"]["acuracia"] == 0.99
    assert artefato.sha256 == calcular_sha256(a.modelo)
    assert artefato.pipeline.predict_proba(imagem_da_classe(3)).shape == (1, 10)


def test_metricas_ausentes_ou_ilegiveis_nao_impedem_o_carregamento(artefato_temporario, tmp_path):
    a = artefato_temporario
    assert carregar_artefato(a.modelo, a.checksum, tmp_path / "nao_existe.json").metricas is None
    assert carregar_artefato(a.modelo, a.checksum, None).metricas is None
    a.metricas.write_text("{quebrado", encoding="utf-8")
    assert carregar_artefato(a.modelo, a.checksum, a.metricas).metricas is None


def test_recusa_artefato_ou_checksum_ausente(artefato_temporario, tmp_path):
    a = artefato_temporario
    with pytest.raises(ErroArtefatoInvalido, match="não encontrado"):
        carregar_artefato(tmp_path / "x.joblib", a.checksum)
    with pytest.raises(ErroArtefatoInvalido, match="Checksum"):
        carregar_artefato(a.modelo, tmp_path / "x.sha256")


def test_recusa_artefato_adulterado(artefato_temporario):
    a = artefato_temporario
    a.checksum.write_text("0" * 64 + "  modelo.joblib\n", encoding="utf-8")
    with pytest.raises(ErroArtefatoInvalido, match="não confere"):
        carregar_artefato(a.modelo, a.checksum)


def test_recusa_checksum_mal_formatado(artefato_temporario):
    artefato_temporario.checksum.write_text("abc\n", encoding="utf-8")
    with pytest.raises(ErroArtefatoInvalido, match="formato inválido"):
        carregar_artefato(artefato_temporario.modelo, artefato_temporario.checksum)


def _regravar(a, conteudo):
    joblib.dump(conteudo, a.modelo)
    a.checksum.write_text(f"{calcular_sha256(a.modelo)}  modelo.joblib\n", encoding="utf-8")


def test_recusa_arquivo_que_nao_e_joblib(artefato_temporario):
    a = artefato_temporario
    a.modelo.write_bytes(b"isto nao e um pickle")
    a.checksum.write_text(f"{calcular_sha256(a.modelo)}  modelo.joblib\n", encoding="utf-8")
    with pytest.raises(ErroArtefatoInvalido, match="desserializar"):
        carregar_artefato(a.modelo, a.checksum)


@pytest.mark.parametrize(
    ("alteracao", "trecho"),
    [
        (lambda a: ["sem", "dict"], "fora do formato"),
        (lambda a: {"pipeline": a.pipeline}, "fora do formato"),
        (lambda a: {"pipeline": a.pipeline, "metadados": {"nome_modelo": "x"}}, "incompletos"),
        (
            lambda a: {"pipeline": a.pipeline, "metadados": {**a.metadados, "classes": [0, 1]}},
            "0 a 9",
        ),
        (
            lambda a: {"pipeline": a.pipeline, "metadados": {**a.metadados, "n_features": 100}},
            "features",
        ),
        (lambda a: {"pipeline": "objeto qualquer", "metadados": a.metadados}, "predict_proba"),
    ],
)
def test_recusa_conteudo_fora_do_contrato(artefato_temporario, alteracao, trecho):
    _regravar(artefato_temporario, alteracao(artefato_temporario))
    with pytest.raises(ErroArtefatoInvalido, match=trecho):
        carregar_artefato(artefato_temporario.modelo, artefato_temporario.checksum)


def test_alerta_quando_versao_do_sklearn_difere(artefato_temporario, caplog):
    a = artefato_temporario
    _regravar(a, {"pipeline": a.pipeline, "metadados": {**a.metadados, "versao_sklearn": "0.0.1"}})
    with caplog.at_level("WARNING", logger="api_mnist"):
        carregar_artefato(a.modelo, a.checksum)
    assert "scikit-learn 0.0.1" in caplog.text


def test_carrega_o_artefato_real_de_producao():
    modelos = RAIZ_PROJETO / "models"
    artefato = carregar_artefato(
        modelos / "modelo_mnist_v1.joblib",
        modelos / "modelo_mnist_v1.sha256",
        modelos / "metricas_modelo_v1.json",
    )
    assert artefato.metadados["nome_modelo"] == "PerceptronMulticamadas"
    assert artefato.metricas["teste"]["acuracia"] > 0.97


# ── Serviço de inferência ──


@pytest.fixture
def servico(artefato_temporario):
    return ServicoInferencia(artefato_temporario.pipeline, artefato_temporario.metadados)


def test_prever_devolve_classe_coerente(servico):
    resultado, ms = servico.prever(imagem_da_classe(4))
    assert resultado.classe_prevista == 4
    assert resultado.confianca == max(resultado.probabilidades)
    assert sum(resultado.probabilidades) == pytest.approx(1.0)
    assert 0.0 <= resultado.entropia <= 1.0
    assert ms >= 0
    assert (servico.nome_modelo, servico.versao_modelo) == ("ModeloTeste", "0.1.0")


def test_prever_lote_preserva_a_ordem(servico):
    lote = np.vstack([imagem_da_classe(c) for c in (9, 0, 5)])
    resultados, _ = servico.prever_lote(lote)
    assert [r.classe_prevista for r in resultados] == [9, 0, 5]


def test_imagem_vazia_e_entrada_invalida(servico):
    with pytest.raises(ErroEntradaInvalida, match="Nenhum traço") as erro:
        servico.prever(np.zeros((1, 784), dtype=np.float32))
    assert erro.value.detalhes == [
        {"campo": "pixels", "mensagem": "Todos os pixels são zero", "tipo": "imagem_vazia"}
    ]


def test_lote_aponta_as_amostras_vazias(servico):
    lote = np.vstack([imagem_da_classe(1), np.zeros((1, 784)), np.zeros((1, 784))])
    with pytest.raises(ErroEntradaInvalida) as erro:
        servico.prever_lote(lote)
    assert [d["campo"] for d in erro.value.detalhes] == ["amostras.1", "amostras.2"]


@pytest.mark.parametrize("forma", [(784,), (1, 783), (0, 784), (2, 28, 28)])
def test_formato_errado_e_entrada_invalida(servico, forma):
    with pytest.raises(ErroEntradaInvalida, match="Formato inválido"):
        servico.prever_lote(np.ones(forma, dtype=np.float32))


def test_prever_recusa_mais_de_uma_imagem(servico):
    with pytest.raises(ErroEntradaInvalida, match="única imagem"):
        servico.prever(np.vstack([imagem_da_classe(1), imagem_da_classe(2)]))


def test_servico_com_o_modelo_real_reconhece_o_exemplo_do_swagger():
    from api.schemas.exemplos import PIXELS_DIGITO_7

    real = joblib.load(RAIZ_PROJETO / "models" / "modelo_mnist_v1.joblib")
    servico = ServicoInferencia(real["pipeline"], real["metadados"])
    resultado, _ = servico.prever(np.asarray([PIXELS_DIGITO_7], dtype=np.float32))
    assert resultado.classe_prevista == 7
    assert resultado.entropia < 0.05


@pytest.mark.parametrize(
    ("distribuicao", "esperado"),
    [([1.0] + [0.0] * 9, 0.0), ([0.1] * 10, 1.0)],
)
def test_entropia_normalizada_nos_extremos(distribuicao, esperado):
    assert entropia_normalizada(np.array(distribuicao)) == pytest.approx(esperado, abs=1e-6)


# ── Desacoplamento do HTTP (Fase 2) ──


@pytest.mark.parametrize("modulo", ["servico_inferencia.py", "carregador_artefato.py"])
def test_services_nao_importam_fastapi_nem_starlette(modulo):
    arvore = ast.parse((RAIZ_PROJETO / "api" / "services" / modulo).read_text(encoding="utf-8"))
    importados = {
        (n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
        for n in ast.walk(arvore)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not importados & {"fastapi", "starlette"}


def test_metricas_reais_tem_o_formato_que_o_carregador_entrega():
    dados = json.loads(
        (RAIZ_PROJETO / "models" / "metricas_modelo_v1.json").read_text(encoding="utf-8")
    )
    assert {"modelo", "validacao", "teste"} <= set(dados)
