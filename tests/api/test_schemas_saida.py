"""Testes dos schemas de saída e do contrato de erro (TM-032)."""

import json

import pytest
from pydantic import ValidationError

from api.core.config import RAIZ_PROJETO
from api.schemas.erro import RESPOSTAS_ERRO_PREDICAO, DetalheErro, SaidaErro
from api.schemas.saida import (
    MetricasResumo,
    SaidaLote,
    SaidaModelo,
    SaidaPredicao,
    SaidaSaude,
)

METRICAS_REAIS = RAIZ_PROJETO / "models" / "metricas_modelo_v1.json"


def predicao(**alteracoes):
    base = {
        "classe_prevista": 7,
        "confianca": 0.91,
        "probabilidades": [0.01] * 7 + [0.91] + [0.01] * 2,  # soma exatamente 1
        "entropia": 0.2,
        "alerta_falsa_certeza": False,
        "confiavel": True,
        "nome_modelo": "PerceptronMulticamadas",
        "versao_modelo": "1.0.0",
        "tempo_inferencia_ms": 0.5,
    }
    return {**base, **alteracoes}


# ── Exemplos do Swagger precisam ser válidos ──


@pytest.mark.parametrize("schema", [SaidaPredicao, SaidaSaude, SaidaErro])
def test_exemplos_documentados_validam_no_proprio_schema(schema):
    for exemplo in schema.model_json_schema()["examples"]:
        schema.model_validate(exemplo)


# ── SaidaPredicao ──


def test_predicao_coerente_e_aceita():
    assert SaidaPredicao(**predicao()).classe_prevista == 7


@pytest.mark.parametrize(
    ("alteracao", "trecho"),
    [
        ({"classe_prevista": 10}, "less than or equal"),
        ({"confianca": 1.2, "probabilidades": [0.0] * 7 + [1.2] + [0.0] * 2}, "less than or equal"),
        ({"probabilidades": [0.1] * 9}, "at least 10"),
        ({"probabilidades": [0.2] * 10}, "somam"),
        ({"classe_prevista": 3}, "maior probabilidade"),
        ({"confianca": 0.5}, "confianca deve ser igual"),
        ({"entropia": -0.1}, "greater than or equal"),
        ({"entropia": 1.5}, "less than or equal"),
        ({"versao_modelo": "v1"}, "pattern"),
        ({"nome_modelo": ""}, "at least 1"),
        ({"tempo_inferencia_ms": -1}, "greater than or equal"),
    ],
)
def test_predicao_incoerente_e_recusada(alteracao, trecho):
    with pytest.raises(ValidationError, match=trecho):
        SaidaPredicao(**predicao(**alteracao))


def test_soma_dentro_da_tolerancia_de_arredondamento_e_aceita():
    probabilidades = [0.0] * 7 + [0.995] + [0.0] * 2
    SaidaPredicao(**predicao(probabilidades=probabilidades, confianca=0.995))


# ── SaidaLote ──


def test_lote_coerente():
    lote = SaidaLote(predicoes=[predicao(), predicao()], quantidade=2, tempo_total_ms=1.0)
    assert lote.quantidade == 2


def test_lote_com_quantidade_divergente_e_recusado():
    with pytest.raises(ValidationError, match="quantidade deve ser igual"):
        SaidaLote(predicoes=[predicao()], quantidade=2, tempo_total_ms=1.0)


def test_lote_vazio_e_recusado():
    with pytest.raises(ValidationError):
        SaidaLote(predicoes=[], quantidade=1, tempo_total_ms=0.0)


# ── SaidaSaude ──


def test_saude_ok_e_degradado():
    ok = SaidaSaude(
        status="ok",
        modelo_carregado=True,
        nome_modelo="PerceptronMulticamadas",
        versao_modelo="1.0.0",
        versao_api="2.0.0",
        tempo_atividade_segundos=1.0,
    )
    degradado = SaidaSaude(
        status="degradado", modelo_carregado=False, versao_api="2.0.0", tempo_atividade_segundos=0
    )
    assert ok.status == "ok" and degradado.nome_modelo is None


@pytest.mark.parametrize(("status", "carregado"), [("ok", False), ("degradado", True)])
def test_saude_contraditoria_e_recusada(status, carregado):
    with pytest.raises(ValidationError, match="status 'ok' exige"):
        SaidaSaude(
            status=status,
            modelo_carregado=carregado,
            versao_api="2.0.0",
            tempo_atividade_segundos=0,
        )


def test_saude_recusa_status_desconhecido():
    with pytest.raises(ValidationError):
        SaidaSaude(
            status="parado", modelo_carregado=False, versao_api="2.0.0", tempo_atividade_segundos=0
        )


# ── SaidaModelo com o JSON real gerado na TM-029 ──


def test_saida_modelo_aceita_as_metricas_reais():
    relatorio = json.loads(METRICAS_REAIS.read_text(encoding="utf-8"))
    modelo = SaidaModelo(
        **{
            k: relatorio["modelo"][k]
            for k in (
                "nome_modelo",
                "versao_modelo",
                "versao_sklearn",
                "data_treino",
                "hiperparametros",
                "amostras",
            )
        },
        metricas_validacao=MetricasResumo.model_validate(relatorio["validacao"]),
        metricas_teste=MetricasResumo.model_validate(relatorio["teste"]),
    )
    assert modelo.metricas_teste.acuracia > 0.97
    assert modelo.amostras["teste"] == modelo.metricas_teste.n_amostras


def test_metricas_fora_da_faixa_sao_recusadas():
    with pytest.raises(ValidationError):
        MetricasResumo(
            acuracia=1.5,
            precisao_macro=0.9,
            recall_macro=0.9,
            f1_macro=0.9,
            roc_auc_ovr=0.9,
            log_loss=0.1,
            n_amostras=10,
        )


# ── SaidaErro ──


def test_erro_completo_com_detalhes():
    erro = SaidaErro(
        codigo=400,
        erro="ENTRADA_INVALIDA",
        mensagem="Nenhum traço detectado na imagem.",
        detalhes=[DetalheErro(campo="arquivo", mensagem="imagem em branco", tipo="sem_traco")],
        id_requisicao="abc-123",
    )
    assert erro.detalhes[0].valor_recebido is None


@pytest.mark.parametrize(
    "alteracao",
    [
        {"codigo": 200},
        {"codigo": 600},
        {"erro": "entrada invalida"},
        {"mensagem": ""},
        {"id_requisicao": ""},
    ],
)
def test_erro_invalido_e_recusado(alteracao):
    base = {"codigo": 400, "erro": "ENTRADA_INVALIDA", "mensagem": "x", "id_requisicao": "1"}
    with pytest.raises(ValidationError):
        SaidaErro(**{**base, **alteracao})


def test_respostas_documentadas_cobrem_400_422_500_503():
    assert set(RESPOSTAS_ERRO_PREDICAO) == {400, 422, 500, 503}
    assert all(r["model"] is SaidaErro for r in RESPOSTAS_ERRO_PREDICAO.values())


def test_cada_status_de_erro_tem_exemplo_proprio_e_valido():
    """Regressão: o Swagger mostrava o exemplo de 422 também na resposta 400."""
    from api.core.config import Configuracoes
    from api.main import criar_aplicacao

    operacao = criar_aplicacao(Configuracoes()).openapi()["paths"]["/v1/predict"]["post"]
    for status in ("400", "422", "500", "503"):
        exemplo = operacao["responses"][status]["content"]["application/json"]["example"]
        assert SaidaErro.model_validate(exemplo).codigo == int(status)
