"""Testes do relatório de métricas do pipeline offline (TM-029)."""

import json

import numpy as np
import pytest

from pipeline_offline import relatorio_metricas as rm
from pipeline_offline import treinar_e_serializar as tes
from pipeline_offline.preparar_dados import dividir_treino_validacao_teste
from src.pre_processamento import pre_processar_dados

METRICAS_REAIS = tes.DIR_MODELOS / rm.NOME_METRICAS


@pytest.fixture(scope="module")
def dados():
    """1 000 amostras separáveis: cada classe acende um bloco diferente de pixels."""
    gerador = np.random.default_rng(3)
    y = np.repeat(np.arange(10, dtype=np.int32), 100)
    X = gerador.random((1000, 784), dtype=np.float32) * 0.1
    for classe in range(10):
        X[y == classe, classe * 70 : classe * 70 + 70] += 0.9
    return np.clip(X, 0, 1), y


@pytest.fixture(scope="module")
def particoes(dados):
    return dividir_treino_validacao_teste(*dados)


@pytest.fixture(scope="module")
def busca(particoes):
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(tes.PARAMETROS_FIXOS, "max_iter", 30)
        return tes.buscar_hiperparametros(
            particoes, {"hidden_layer_sizes": [(16,)], "alpha": [1e-3]}
        )


@pytest.fixture(scope="module")
def metadados(busca, particoes):
    return tes.montar_metadados(busca, particoes)


@pytest.fixture(scope="module")
def relatorio(busca, metadados, particoes):
    return rm.gerar_relatorio(busca.pipeline, metadados, particoes)


# ── Mesmo conjunto de teste do Módulo 1 ──


def test_teste_do_modulo_2_e_identico_ao_do_modulo_1(dados):
    X, y = dados
    _, X_teste_m1, _, y_teste_m1, escalonador = pre_processar_dados(X, y)
    particoes_m2 = dividir_treino_validacao_teste(X, y)
    np.testing.assert_array_equal(y_teste_m1, particoes_m2.y_teste)
    np.testing.assert_allclose(
        escalonador.inverse_transform(X_teste_m1), particoes_m2.X_teste, atol=1e-5
    )


# ── Pares mais confundidos ──


def test_pares_mais_confundidos_ignora_diagonal_e_ordena():
    matriz = [[50, 3, 0], [1, 40, 7], [0, 2, 60]]
    pares = rm.pares_mais_confundidos(matriz, quantidade=3)
    assert pares == [
        {"real": 1, "previsto": 2, "erros": 7},
        {"real": 0, "previsto": 1, "erros": 3},
        {"real": 2, "previsto": 1, "erros": 2},
    ]


def test_pares_mais_confundidos_para_quando_acabam_os_erros():
    assert rm.pares_mais_confundidos([[5, 1], [0, 5]], quantidade=5) == [
        {"real": 0, "previsto": 1, "erros": 1}
    ]
    assert rm.pares_mais_confundidos([[5, 0], [0, 5]]) == []


# ── Avaliação ──


def test_avaliar_particao_calcula_todas_as_metricas(busca, particoes):
    m = rm.avaliar_particao(busca.pipeline, particoes.X_teste, particoes.y_teste)
    assert m["n_amostras"] == len(particoes.y_teste)
    for chave in ("acuracia", "precisao_macro", "recall_macro", "f1_macro", "roc_auc_ovr"):
        assert 0.9 <= m[chave] <= 1.0
    assert m["log_loss"] >= 0
    assert np.array(m["matriz_confusao"]).shape == (10, 10)
    assert np.array(m["matriz_confusao"]).sum() == len(particoes.y_teste)
    assert set(m["por_classe"]) == {str(c) for c in range(10)}
    assert sum(v["suporte"] for v in m["por_classe"].values()) == len(particoes.y_teste)


def test_relatorio_tem_validacao_teste_e_comparacao(relatorio):
    assert set(relatorio) == {"modelo", "validacao", "teste", "comparacao_modulo_1"}
    comp = relatorio["comparacao_modulo_1"]
    esperado = round((relatorio["teste"]["acuracia"] - comp["acuracia"]) * 100, 2)
    assert comp["ganho_acuracia_pp"] == esperado
    json.dumps(relatorio)  # serializável


def test_rejeita_particoes_incompativeis_com_o_artefato(busca, metadados, particoes):
    divergente = {**metadados, "amostras": {**metadados["amostras"], "teste": 1}}
    with pytest.raises(ValueError, match="incompatíveis"):
        rm.gerar_relatorio(busca.pipeline, divergente, particoes)


# ── Carregamento verificado ──


def test_carrega_artefato_com_checksum_valido(busca, metadados, tmp_path):
    caminho, _ = tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    pipeline, carregados = rm.carregar_artefato_verificado(caminho)
    assert carregados == metadados
    assert pipeline.predict_proba(np.zeros((1, 784), dtype=np.float32)).shape == (1, 10)


def test_recusa_artefato_adulterado(busca, metadados, tmp_path):
    caminho, caminho_sha = tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    caminho_sha.write_text("0" * 64 + "  " + caminho.name + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 não confere"):
        rm.carregar_artefato_verificado(caminho)


def test_recusa_artefato_ausente(tmp_path):
    with pytest.raises(FileNotFoundError):
        rm.carregar_artefato_verificado(tmp_path / "nao_existe.joblib")


# ── Saídas ──


def test_salvar_json_em_utf8_com_lf(relatorio, tmp_path):
    caminho = rm.salvar_json(relatorio, tmp_path / "sub" / "m.json")
    bruto = caminho.read_bytes()
    assert b"\r" not in bruto
    assert json.loads(bruto.decode("utf-8")) == relatorio


def test_salvar_matriz_confusao_gera_png(relatorio, tmp_path):
    caminho = rm.salvar_matriz_confusao(
        relatorio["teste"]["matriz_confusao"], tmp_path / "img" / "m.png", "teste"
    )
    assert caminho.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_markdown_tem_todas_as_secoes(relatorio):
    texto = rm.gerar_markdown(relatorio, "img/m.png")
    for secao in ("## 1. Modelo", "## 2. Métricas", "## 3. Comparação", "## 4.", "## 5.", "## 6."):
        assert secao in texto
    assert "![Matriz de confusão no teste](img/m.png)" in texto


@pytest.mark.parametrize(
    ("pares", "trecho"),
    [
        (
            [{"real": 4, "previsto": 9, "erros": 30}, {"real": 8, "previsto": 3, "erros": 10}],
            "mais que o dobro",
        ),
        (
            [{"real": 4, "previsto": 9, "erros": 12}, {"real": 8, "previsto": 3, "erros": 10}],
            "Não há um único par dominante",
        ),
        ([], "Não houve erros"),
    ],
)
def test_leitura_dos_pares_depende_dos_dados(relatorio, pares, trecho):
    teste = dict(relatorio["teste"], pares_mais_confundidos=pares)
    erros = sum(p["erros"] for p in pares)
    # Matriz coerente com os pares: os acertos ficam na diagonal, os erros fora dela
    matriz = np.diag([teste["n_amostras"] - erros] + [0] * 9)
    for p in pares:
        matriz[p["real"], p["previsto"]] += p["erros"]
    teste["matriz_confusao"] = matriz.tolist()
    texto = rm.gerar_markdown({**relatorio, "teste": teste}, "m.png")
    assert trecho in texto


def test_main_grava_json_imagem_e_markdown(busca, metadados, particoes, tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "carregar_artefato_verificado", lambda: (busca.pipeline, metadados))
    monkeypatch.setattr(
        "pipeline_offline.preparar_dados.carregar_e_dividir", lambda semente: particoes
    )
    monkeypatch.setattr(rm, "DIR_MODELOS", tmp_path / "models")
    monkeypatch.setattr(rm, "CAMINHO_RELATORIO_MD", tmp_path / "docs" / "R.md")
    monkeypatch.setattr(rm, "CAMINHO_MATRIZ_PNG", tmp_path / "docs" / "img" / "m.png")
    (tmp_path / "docs").mkdir()
    rm.main()
    assert (tmp_path / "models" / rm.NOME_METRICAS).exists()
    assert (tmp_path / "docs" / "img" / "m.png").exists()
    assert "](img/m.png)" in (tmp_path / "docs" / "R.md").read_text(encoding="utf-8")


# ── JSON real versionado em models/ (lido pela API em GET /v1/modelo) ──


@pytest.mark.skipif(not METRICAS_REAIS.exists(), reason="métricas reais ainda não geradas")
def test_json_real_tem_contrato_esperado_pela_api():
    dados_json = json.loads(METRICAS_REAIS.read_text(encoding="utf-8"))
    for particao in ("validacao", "teste"):
        for chave in ("acuracia", "f1_macro", "roc_auc_ovr", "matriz_confusao"):
            assert chave in dados_json[particao]
    assert dados_json["teste"]["acuracia"] > 0.97
    assert dados_json["teste"]["n_amostras"] == dados_json["modelo"]["amostras"]["teste"]
    assert dados_json["modelo"]["versao_modelo"] == tes.VERSAO_MODELO
