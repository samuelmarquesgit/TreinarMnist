"""Testes do tratamento centralizado de erros (TM-040, Fase 5)."""

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from api.core.config import Configuracoes
from api.core.excecoes import ErroEntradaInvalida
from api.core.manipuladores import CABECALHO_ID, traduzir_erros_validacao
from api.dependencias import obter_servico
from api.main import criar_aplicacao
from api.schemas.entrada import EntradaPixels
from api.schemas.erro import SaidaErro


@pytest.fixture
def cliente(tmp_path):
    """App sem modelo (degradado) com rotas auxiliares que provocam cada tipo de erro."""
    app = criar_aplicacao(Configuracoes(caminho_modelo=tmp_path / "ausente.joblib"))

    @app.post("/teste/validacao")
    def validacao(entrada: EntradaPixels) -> dict:
        return {"ok": True}

    @app.get("/teste/entrada-invalida")
    def entrada_invalida() -> None:
        raise ErroEntradaInvalida(
            "Nenhum traço detectado.",
            detalhes=[
                {"campo": "pixels", "mensagem": "Todos os pixels são zero", "tipo": "imagem_vazia"}
            ],
        )

    @app.get("/teste/precisa-modelo")
    def precisa_modelo(servico=Depends(obter_servico)) -> None:  # noqa: B008
        return None

    @app.get("/teste/inesperado")
    def inesperado() -> None:
        raise ZeroDivisionError("detalhe interno secreto")

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def validar_contrato(resposta, codigo: int, erro: str) -> SaidaErro:
    assert resposta.status_code == codigo
    corpo = SaidaErro.model_validate(resposta.json())
    assert (corpo.codigo, corpo.erro) == (codigo, erro)
    assert resposta.headers[CABECALHO_ID] == corpo.id_requisicao
    return corpo


# ── 422 ──


def test_422_pixel_fora_da_faixa_com_detalhe_em_portugues(cliente):
    corpo = validar_contrato(
        cliente.post("/teste/validacao", json={"pixels": [2.0] + [0.0] * 783}),
        422,
        "VALIDACAO_FALHOU",
    )
    detalhe = corpo.detalhes[0]
    assert detalhe.campo == "pixels.0"
    assert detalhe.mensagem == "O valor deve ser menor ou igual a 1.0"
    assert detalhe.tipo == "less_than_equal"
    assert detalhe.valor_recebido == 2.0


@pytest.mark.parametrize(
    ("corpo", "tipo", "campo"),
    [
        ({"pixels": ["abc"] + [0.0] * 783}, "float_type", "pixels.0"),
        ({"pixels": [0.0] * 783}, "too_short", "pixels"),
        ({"pixels": [0.0] * 784, "extra": 1}, "extra_forbidden", "extra"),
        ({}, "missing", "pixels"),
    ],
)
def test_422_cobre_os_casos_do_edital(cliente, corpo, tipo, campo):
    resposta = validar_contrato(
        cliente.post("/teste/validacao", json=corpo), 422, "VALIDACAO_FALHOU"
    )
    assert (resposta.detalhes[0].tipo, resposta.detalhes[0].campo) == (tipo, campo)


def test_422_json_malformado(cliente):
    resposta = cliente.post(
        "/teste/validacao", content=b"{pixels: ", headers={"Content-Type": "application/json"}
    )
    corpo = validar_contrato(resposta, 422, "VALIDACAO_FALHOU")
    assert corpo.detalhes[0].tipo == "json_invalid"


def test_422_nao_ecoa_listas_grandes(cliente):
    corpo = validar_contrato(
        cliente.post("/teste/validacao", json={"pixels": [0.0] * 10}), 422, "VALIDACAO_FALHOU"
    )
    assert corpo.detalhes[0].valor_recebido is None


# ── 400 / 503 ──


def test_400_erro_de_dominio_com_detalhes(cliente):
    corpo = validar_contrato(cliente.get("/teste/entrada-invalida"), 400, "ENTRADA_INVALIDA")
    assert corpo.mensagem == "Nenhum traço detectado."
    assert corpo.detalhes[0].tipo == "imagem_vazia"


def test_503_sem_modelo_em_memoria(cliente):
    corpo = validar_contrato(cliente.get("/teste/precisa-modelo"), 503, "MODELO_INDISPONIVEL")
    assert "não está carregado" in corpo.mensagem


# ── 500 ──


def test_500_generico_sem_stack_trace_e_api_continua_de_pe(cliente):
    resposta = cliente.get("/teste/inesperado")
    corpo = validar_contrato(resposta, 500, "ERRO_INTERNO")
    assert "detalhe interno secreto" not in resposta.text
    assert "Traceback" not in resposta.text and "ZeroDivisionError" not in resposta.text
    assert corpo.detalhes is None
    assert cliente.get("/health").status_code == 200  # não caiu


def test_500_registra_o_erro_completo_no_log(cliente, caplog):
    with caplog.at_level("ERROR", logger="api_mnist"):
        corpo = cliente.get("/teste/inesperado").json()
    assert corpo["id_requisicao"] in caplog.text
    assert "ZeroDivisionError" in caplog.text


# ── 404 / 405 e id de requisição ──


def test_404_e_405_no_formato_padronizado(cliente):
    validar_contrato(cliente.get("/rota/inexistente"), 404, "NAO_ENCONTRADO")
    validar_contrato(cliente.delete("/health"), 405, "METODO_NAO_PERMITIDO")


def test_id_de_requisicao_recebido_e_reaproveitado(cliente):
    resposta = cliente.get("/teste/entrada-invalida", headers={CABECALHO_ID: "meu-id-123"})
    assert resposta.json()["id_requisicao"] == "meu-id-123"
    assert resposta.headers[CABECALHO_ID] == "meu-id-123"


# ── Rota de diagnóstico ──


def test_rota_de_diagnostico_desligada_por_padrao(cliente):
    validar_contrato(cliente.get("/v1/_diagnostico/erro-interno"), 404, "NAO_ENCONTRADO")


def test_rota_de_diagnostico_ligada_gera_500_padronizado(tmp_path):
    app = criar_aplicacao(
        Configuracoes(caminho_modelo=tmp_path / "x.joblib", habilitar_diagnostico=True)
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        resposta = c.get("/v1/_diagnostico/erro-interno")
        corpo = validar_contrato(resposta, 500, "ERRO_INTERNO")
        assert "proposital" not in resposta.text
        assert corpo.id_requisicao
        assert "/v1/_diagnostico/erro-interno" not in app.openapi()["paths"]


# ── Tradução ──


def test_traducao_usa_mensagem_original_para_tipos_desconhecidos():
    detalhes = traduzir_erros_validacao(
        [
            {
                "type": "tipo_novo",
                "loc": ("body", "x"),
                "msg": "mensagem original",
                "input": float("nan"),
            }
        ]
    )
    assert detalhes[0].mensagem == "mensagem original"
    assert detalhes[0].campo == "x"
    assert detalhes[0].valor_recebido == "nan"


def test_traducao_tolera_contexto_incompleto_e_textos_longos():
    detalhes = traduzir_erros_validacao(
        [{"type": "less_than_equal", "loc": ("body",), "msg": "msg", "ctx": {}, "input": "a" * 300}]
    )
    assert detalhes[0].mensagem == "msg"
    assert detalhes[0].campo == "body"
    assert detalhes[0].valor_recebido.endswith("…")
