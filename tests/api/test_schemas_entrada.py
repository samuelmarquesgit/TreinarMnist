"""Testes dos schemas de entrada, sem subir a API (TM-031)."""

import json
import math

import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from api.core.config import RAIZ_PROJETO
from api.schemas.entrada import LIMITE_LOTE, N_PIXELS, EntradaLote, EntradaPixels
from api.schemas.exemplos import PIXELS_DIGITO_7, ROTULO_DIGITO_7

VALIDO = [0.0] * N_PIXELS
ARTEFATO = RAIZ_PROJETO / "models" / "modelo_mnist_v1.joblib"


def tipos_de_erro(erro: ValidationError) -> set[str]:
    return {e["type"] for e in erro.errors()}


# ── Casos válidos ──


def test_aceita_784_pixels_normalizados():
    entrada = EntradaPixels(pixels=PIXELS_DIGITO_7)
    assert len(entrada.pixels) == N_PIXELS


def test_aceita_inteiros_0_e_1_do_json():
    entrada = EntradaPixels.model_validate_json(json.dumps({"pixels": [0, 1] * 392}))
    assert entrada.pixels[:2] == [0.0, 1.0]


def test_limites_0_e_1_sao_inclusivos():
    EntradaPixels(pixels=[0.0] * 783 + [1.0])


def test_para_matriz_gera_float32_1x784():
    matriz = EntradaPixels(pixels=PIXELS_DIGITO_7).para_matriz()
    assert matriz.shape == (1, N_PIXELS)
    assert matriz.dtype == np.float32
    assert matriz[0, 216] == pytest.approx(PIXELS_DIGITO_7[216])


# ── Casos que o edital exige que virem 422 ──


@pytest.mark.parametrize(
    ("pixel", "tipo"),
    [
        ("abc", "float_type"),  # texto em campo numérico
        ("0.5", "float_type"),  # número como string: recusado pelo modo estrito
        (True, "float_type"),  # booleano
        (None, "float_type"),
        (1.01, "less_than_equal"),  # acima da faixa
        (-0.01, "greater_than_equal"),  # abaixo da faixa
        (math.nan, "finite_number"),
        (math.inf, "finite_number"),
    ],
)
def test_recusa_pixel_invalido(pixel, tipo):
    with pytest.raises(ValidationError) as erro:
        EntradaPixels(pixels=[pixel] + [0.0] * 783)
    assert tipo in tipos_de_erro(erro.value)
    assert erro.value.errors()[0]["loc"] == ("pixels", 0)


@pytest.mark.parametrize(
    "corpo", ['{"pixels": [NaN' + ", 0" * 783 + "]}", '{"pixels": [Infinity' + ", 0" * 783 + "]}"]
)
def test_recusa_nan_e_infinity_vindos_do_json(corpo):
    with pytest.raises(ValidationError):
        EntradaPixels.model_validate_json(corpo)


@pytest.mark.parametrize(
    ("tamanho", "tipo"), [(783, "too_short"), (785, "too_long"), (0, "too_short")]
)
def test_recusa_quantidade_errada_de_pixels(tamanho, tipo):
    with pytest.raises(ValidationError) as erro:
        EntradaPixels(pixels=[0.0] * tamanho)
    assert tipo in tipos_de_erro(erro.value)


def test_recusa_campo_desconhecido():
    with pytest.raises(ValidationError) as erro:
        EntradaPixels(pixels=VALIDO, classe=7)
    assert "extra_forbidden" in tipos_de_erro(erro.value)


def test_recusa_corpo_sem_pixels_ou_com_nome_errado():
    with pytest.raises(ValidationError) as erro:
        EntradaPixels.model_validate({"pixel": VALIDO})
    assert {"missing", "extra_forbidden"} <= tipos_de_erro(erro.value)


def test_recusa_pixels_que_nao_sao_lista():
    with pytest.raises(ValidationError) as erro:
        EntradaPixels.model_validate({"pixels": "0,0,0"})
    assert "list_type" in tipos_de_erro(erro.value)


# ── Lote ──


def test_lote_valido_e_conversao_para_matriz():
    lote = EntradaLote(amostras=[{"pixels": VALIDO}, {"pixels": PIXELS_DIGITO_7}])
    matriz = lote.para_matriz()
    assert matriz.shape == (2, N_PIXELS)
    assert matriz.dtype == np.float32


@pytest.mark.parametrize(("quantidade", "tipo"), [(0, "too_short"), (LIMITE_LOTE + 1, "too_long")])
def test_lote_fora_dos_limites(quantidade, tipo):
    with pytest.raises(ValidationError) as erro:
        EntradaLote(amostras=[{"pixels": VALIDO}] * quantidade)
    assert tipo in tipos_de_erro(erro.value)


def test_lote_aponta_a_amostra_e_o_pixel_invalidos():
    with pytest.raises(ValidationError) as erro:
        EntradaLote(amostras=[{"pixels": VALIDO}, {"pixels": [0.0] * 3 + [2.0] + [0.0] * 780}])
    assert erro.value.errors()[0]["loc"] == ("amostras", 1, "pixels", 3)


def test_lote_recusa_campo_desconhecido():
    with pytest.raises(ValidationError) as erro:
        EntradaLote(amostras=[{"pixels": VALIDO}], extra=1)
    assert "extra_forbidden" in tipos_de_erro(erro.value)


# ── Documentação OpenAPI e exemplo real ──


def test_json_schema_documenta_restricoes_e_exemplo():
    esquema = EntradaPixels.model_json_schema()
    campo = esquema["properties"]["pixels"]
    assert (campo["minItems"], campo["maxItems"]) == (N_PIXELS, N_PIXELS)
    assert (campo["items"]["minimum"], campo["items"]["maximum"]) == (0.0, 1.0)
    assert esquema["additionalProperties"] is False
    assert esquema["examples"][0]["pixels"] == PIXELS_DIGITO_7
    assert EntradaLote.model_json_schema()["properties"]["amostras"]["maxItems"] == LIMITE_LOTE


def test_exemplo_do_swagger_e_um_7_reconhecido_pelo_modelo_real():
    pipeline = joblib.load(ARTEFATO)["pipeline"]
    matriz = EntradaPixels(pixels=PIXELS_DIGITO_7).para_matriz()
    assert int(pipeline.predict(matriz)[0]) == ROTULO_DIGITO_7
