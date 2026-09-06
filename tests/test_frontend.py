"""Testes do frontend Streamlit — streamlit e plotly completamente mockados."""

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Infra: mock de context manager reutilizável ────────────────────────────


def _mk_ctx():
    """Cria um MagicMock compatível com 'with' statement."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


# ── Mock completo do Streamlit (instalado ANTES de qualquer import frontend) ──

class _SessionStateMock(dict):
    """Simula st.session_state como dict com acesso por atributo."""
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value


_mock_st = MagicMock()
_mock_st.session_state = _SessionStateMock()


def _make_columns(n, *args, **kwargs):
    count = n if isinstance(n, int) else len(n)
    cols = []
    for _ in range(count):
        c = _mk_ctx()
        # Widgets embutidos em colunas retornam valores escalares seguros
        c.selectbox.return_value = 0
        c.number_input.return_value = 5
        c.slider.return_value = 5
        c.radio.return_value = "Brutos [0–255]"
        c.checkbox.return_value = False
        c.toggle.return_value = False
        cols.append(c)
    return cols


def _make_tabs(labels, *args, **kwargs):
    return [_mk_ctx() for _ in range(len(labels))]


_mock_st.columns.side_effect = _make_columns
_mock_st.tabs.side_effect = _make_tabs
_mock_st.expander.return_value = _mk_ctx()
_mock_st.form.return_value = _mk_ctx()
_mock_st.spinner.return_value = _mk_ctx()
_mock_st.container.return_value = _mk_ctx()
_mock_st.sidebar = _mk_ctx()
_mock_st.checkbox.return_value = False
_mock_st.toggle.return_value = False
_mock_st.selectbox.return_value = None
_mock_st.radio.return_value = "Brutos [0–255]"
_mock_st.multiselect.return_value = []
_mock_st.number_input.return_value = 20
_mock_st.slider.return_value = 5
_mock_st.text_input.return_value = ""
_mock_st.text_area.return_value = ""
_mock_st.button.return_value = False
_mock_st.form_submit_button.return_value = False
_mock_st.file_uploader.return_value = None
_mock_st.chat_input.return_value = None
_mock_st.color_picker.return_value = "#FFFFFF"
_mock_st.progress.return_value = MagicMock()

# Mock plotly (não instalado na VM Linux)
_mock_px = MagicMock()
_mock_go = MagicMock()
_mock_plotly = MagicMock()

# Instala mocks no sys.modules
sys.modules['streamlit'] = _mock_st
sys.modules['plotly'] = _mock_plotly
sys.modules['plotly.express'] = _mock_px
sys.modules['plotly.graph_objects'] = _mock_go
sys.modules['streamlit_drawable_canvas'] = MagicMock()

# Limpa cache de módulos frontend para garantir import com mock
for _mod in list(sys.modules):
    if _mod.startswith('src.frontend'):
        del sys.modules[_mod]

# ── Importações dos módulos frontend (agora com streamlit mockado) ─────────

from src.frontend.estilos import (  # noqa: E402
    kpi_tile, badge, aplicar_estilos, card, titulo_secao,
)
from src.frontend.painel_robustez_ood import (  # noqa: E402
    _simular_softmax,
    _simular_softmax_in_dist,
    _entropia_shannon,
    _avaliar_lote,
    renderizar as renderizar_ood,
)
from src.frontend.painel_laboratorio_visao import (  # noqa: E402
    ordenar_probabilidades_por_bolha,
    _pipeline_visual,
    _renderizar_pipeline_e_inferencia,
    renderizar as renderizar_lab,
)
from src.frontend.painel_benchmarks import (  # noqa: E402
    _formatar_tabela, _renderizar_kpis, _renderizar_graficos,
    _renderizar_matriz_confusao, _executar_benchmark,
    renderizar as renderizar_bench,
)
from src.frontend.painel_assistente_rag import (  # noqa: E402
    _resposta_fallback,
    renderizar as renderizar_rag,
)
from src.frontend.painel_analise_estatistica import (  # noqa: E402
    _diagnostico_assimetria,
    renderizar as renderizar_analise,
)
from src.frontend.painel_eda import renderizar as renderizar_eda  # noqa: E402
from src.frontend.painel_bancos_dados import (  # noqa: E402
    _obter_experimentos_postgres,
    _obter_artefatos_mongodb,
    renderizar as renderizar_bancos,
)
from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza  # noqa: E402


# ── helpers de fachada ─────────────────────────────────────────────────────


def _fachada_mock():
    """Cria FachadaPipelineIA simulada para testes de renderização."""
    f = MagicMock()
    f.dados_inicializados.return_value = True
    f.listar_modelos_treinados.return_value = ["RegressaoLogistica"]
    f.X_treino = np.ones((100, 784), dtype=np.float32)
    f.y_treino = np.zeros(100, dtype=np.int32)
    f.X_teste = np.ones((20, 784), dtype=np.float32)
    f.y_teste = np.zeros(20, dtype=np.int32)
    f.avaliar_modelo.return_value = {
        "acuracia": 0.95, "precisao": 0.94, "recall": 0.94,
        "f1": 0.94, "tempo_treino": 0.5,
        "matriz_confusao": [[0] * 10 for _ in range(10)],
    }
    f.obter_estatisticas_dados.return_value = {
        "media": 0.1, "mediana": 0.0, "desvio_padrao": 0.3,
        "variancia": 0.09, "minimo": 0.0, "maximo": 1.0,
        "assimetria": 0.2, "curtose": -0.1,
    }
    # Amostra EDA: 10 imagens
    f.amostras_por_classe.return_value = {
        i: np.zeros((28, 28), dtype=np.uint8) for i in range(10)
    }
    return f


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: estilos.py ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_kpi_tile_retorna_html_com_valor_e_label():
    html = kpi_tile("0.95", "Acurácia")
    assert "0.95" in html
    assert "Acurácia" in html
    assert "kpi-tile" in html


def test_kpi_tile_retorna_string():
    assert isinstance(kpi_tile("42", "Teste"), str)


def test_badge_ok_contem_classe_correta():
    html = badge("Ativo", "ok")
    assert "badge-ok" in html
    assert "Ativo" in html


def test_badge_erro_contem_classe_erro():
    html = badge("Falhou", "erro")
    assert "badge-erro" in html


def test_badge_aviso_contem_classe_aviso():
    html = badge("Atenção", "aviso")
    assert "badge-aviso" in html


def test_badge_tipo_desconhecido_usa_ok():
    html = badge("X", "desconhecido")
    assert "badge-ok" in html


def test_aplicar_estilos_chama_st_markdown():
    _mock_st.reset_mock()
    aplicar_estilos()
    _mock_st.markdown.assert_called_once()


def test_card_chama_st_markdown():
    _mock_st.reset_mock()
    card("<p>conteúdo</p>")
    _mock_st.markdown.assert_called_once()
    call_arg = _mock_st.markdown.call_args[0][0]
    assert "glass-card" in call_arg


def test_titulo_secao_chama_st_markdown():
    _mock_st.reset_mock()
    titulo_secao("Minha Seção")
    _mock_st.markdown.assert_called_once()
    call_arg = _mock_st.markdown.call_args[0][0]
    assert "Minha Seção" in call_arg


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_robustez_ood.py ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_simular_softmax_shape():
    probs = _simular_softmax(5, [0, 1, 2, 3, 5, 6, 8, 9])
    assert probs.shape == (5, 10)


def test_simular_softmax_soma_um():
    probs = _simular_softmax(4, [0, 1, 2])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_simular_softmax_reproducivel_com_seed():
    p1 = _simular_softmax(3, [1, 2], seed=77)
    p2 = _simular_softmax(3, [1, 2], seed=77)
    np.testing.assert_array_equal(p1, p2)


def test_simular_softmax_in_dist_shape():
    probs = _simular_softmax_in_dist(6)
    assert probs.shape == (6, 10)


def test_simular_softmax_in_dist_soma_um():
    probs = _simular_softmax_in_dist(4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_entropia_shannon_uniforme_maxima():
    prob = np.ones(10) / 10
    H = _entropia_shannon(prob)
    esperado = float(-10 * 0.1 * np.log(0.1))
    assert H == pytest.approx(esperado, rel=1e-5)


def test_entropia_shannon_deterministica_proxima_de_zero():
    prob = np.zeros(10)
    prob[5] = 1.0
    assert _entropia_shannon(prob) < 0.01


def test_entropia_shannon_retorna_float():
    assert isinstance(_entropia_shannon(np.ones(5) / 5), float)


def test_avaliar_lote_retorna_dataframe():
    probs = np.ones((3, 10)) / 10
    val = ValidadorFalsaCerteza()
    df = _avaliar_lote(probs, list(range(10)), val)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_avaliar_lote_colunas_presentes():
    probs = np.ones((2, 10)) / 10
    val = ValidadorFalsaCerteza()
    df = _avaliar_lote(probs, list(range(10)), val)
    for col in ("Amostra", "Classe Prevista", "Confiança", "Entropia", "Alerta OOD", "Confiável"):
        assert col in df.columns


def test_renderizar_ood_nao_levanta_excecao():
    """renderizar() do painel OOD deve executar sem erros com st mockado."""
    fachada = _fachada_mock()
    renderizar_ood(fachada)  # não deve levantar


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_laboratorio_visao.py ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_ordenar_por_bolha_ordem_decrescente():
    probs = [(0, 0.1), (1, 0.5), (2, 0.3)]
    ordenado = ordenar_probabilidades_por_bolha(probs)
    valores = [v for _, v in ordenado]
    assert valores == sorted(valores, reverse=True)


def test_ordenar_por_bolha_lista_vazia():
    assert ordenar_probabilidades_por_bolha([]) == []


def test_ordenar_por_bolha_um_elemento():
    assert ordenar_probabilidades_por_bolha([(3, 0.9)]) == [(3, 0.9)]


def test_ordenar_por_bolha_preserva_todos_elementos():
    probs = [(i, float(i) / 10) for i in range(10)]
    ordenado = ordenar_probabilidades_por_bolha(probs)
    assert len(ordenado) == 10
    assert set(c for c, _ in ordenado) == set(range(10))


def test_pipeline_visual_retorna_quatro_arrays():
    img = (np.random.rand(28, 28) * 255).astype(np.uint8)
    resultado = _pipeline_visual(img)
    assert len(resultado) == 4


def test_pipeline_visual_imagem_rgb():
    img = (np.random.rand(28, 28, 3) * 255).astype(np.uint8)
    gray, invertida, bbox, canvas = _pipeline_visual(img)
    assert canvas.shape == (28, 28)


def test_pipeline_visual_canvas_28x28():
    img = np.zeros((28, 28), dtype=np.uint8)
    img[10:20, 10:20] = 200
    _, _, _, canvas = _pipeline_visual(img)
    assert canvas.shape == (28, 28)


def test_renderizar_lab_nao_levanta_excecao():
    fachada = _fachada_mock()
    fachada.modelos = {}
    renderizar_lab(fachada)


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_benchmarks.py ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def _tres_modelos():
    """Retorna dict com 3 modelos (mínimo para as medalhas não quebrarem)."""
    return {
        "ModeloA": {"acuracia": 0.95, "precisao": 0.94, "recall": 0.94, "f1": 0.94, "tempo_treino": 1.2},
        "ModeloB": {"acuracia": 0.88, "precisao": 0.87, "recall": 0.87, "f1": 0.87, "tempo_treino": 0.8},
        "ModeloC": {"acuracia": 0.82, "precisao": 0.81, "recall": 0.81, "f1": 0.81, "tempo_treino": 0.6},
    }


def test_formatar_tabela_retorna_dataframe():
    df = _formatar_tabela(_tres_modelos())
    assert isinstance(df, pd.DataFrame)
    assert "Modelo" in df.columns
    assert "Acurácia" in df.columns


def test_formatar_tabela_ordena_por_acuracia():
    resultados = {
        "Fraco": {"acuracia": 0.70, "precisao": 0.70, "recall": 0.70, "f1": 0.70, "tempo_treino": 0.5},
        "Forte": {"acuracia": 0.95, "precisao": 0.95, "recall": 0.95, "f1": 0.95, "tempo_treino": 1.0},
        "Medio": {"acuracia": 0.80, "precisao": 0.80, "recall": 0.80, "f1": 0.80, "tempo_treino": 0.7},
    }
    df = _formatar_tabela(resultados)
    assert df.iloc[0]["Modelo"] == "Forte"


def test_formatar_tabela_medalhas():
    df = _formatar_tabela(_tres_modelos())
    assert "🥇" in df["🏅"].values


def test_renderizar_bench_sem_modelos_nao_levanta():
    fachada = _fachada_mock()
    _mock_st.multiselect.return_value = []
    _mock_st.session_state["bench_resultados"] = {}
    renderizar_bench(fachada)


def test_renderizar_bench_com_resultados_na_sessao():
    fachada = _fachada_mock()
    _mock_st.session_state["bench_resultados"] = {
        "RegressaoLogistica": {
            "acuracia": 0.95, "precisao": 0.94, "recall": 0.94, "f1": 0.94,
            "tempo_treino": 0.5, "matriz_confusao": [[0] * 10 for _ in range(10)],
        }
    }
    renderizar_bench(fachada)


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_assistente_rag.py ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_resposta_fallback_palavra_chave_mnist():
    resp = _resposta_fallback("O que é MNIST?")
    assert "offline" in resp.lower() or "mnist" in resp.lower()


def test_resposta_fallback_palavra_desconhecida():
    resp = _resposta_fallback("pergunta completamente aleatória xyzxyz")
    assert "offline" in resp.lower() or "Não encontrei" in resp


def test_resposta_fallback_retorna_string():
    assert isinstance(_resposta_fallback("teste"), str)


def test_resposta_fallback_modos_modelo():
    resp = _resposta_fallback("quais modelos estão disponíveis?")
    assert isinstance(resp, str)
    assert len(resp) > 0


def test_renderizar_rag_sem_historico_nao_levanta():
    _mock_st.session_state.clear()
    renderizar_rag()


def test_renderizar_rag_com_historico_existente():
    _mock_st.session_state["historico_chat"] = [
        {"papel": "usuario", "conteudo": "Olá"},
        {"papel": "assistente", "conteudo": "Olá! Como posso ajudar?"},
    ]
    _mock_st.session_state["rag_pronto"] = False
    renderizar_rag()


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_analise_estatistica.py ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_diagnostico_assimetria_simetrica():
    diag = _diagnostico_assimetria(0.1, 0.2)
    assert "simétrica" in diag.lower()


def test_diagnostico_assimetria_direita():
    diag = _diagnostico_assimetria(1.5, 0.0)
    assert "direita" in diag.lower()


def test_diagnostico_assimetria_esquerda():
    diag = _diagnostico_assimetria(-1.5, 0.0)
    assert "esquerda" in diag.lower()


def test_diagnostico_assimetria_leptocurtica():
    diag = _diagnostico_assimetria(0.0, 1.0)
    assert "leptocúrtica" in diag.lower()


def test_diagnostico_assimetria_platicurtica():
    diag = _diagnostico_assimetria(0.0, -1.0)
    assert "platicúrtica" in diag.lower()


def test_diagnostico_assimetria_retorna_string():
    assert isinstance(_diagnostico_assimetria(0.0, 0.0), str)


def test_renderizar_analise_nao_levanta():
    fachada = _fachada_mock()
    _mock_st.radio.return_value = "Brutos [0–255]"
    _mock_st.toggle.return_value = False
    _mock_st.selectbox.return_value = 0
    renderizar_analise(fachada)


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_eda.py ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_renderizar_eda_nao_levanta_excecao():
    fachada = _fachada_mock()
    renderizar_eda(fachada)


def test_renderizar_eda_chama_aplicar_estilos():
    fachada = _fachada_mock()
    _mock_st.reset_mock()
    renderizar_eda(fachada)
    # st.markdown deve ter sido chamado (por aplicar_estilos e outros)
    assert _mock_st.markdown.called


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes: painel_bancos_dados.py ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_obter_experimentos_postgres_retorna_dataframe():
    """Deve retornar DataFrame vazio em ambiente sem banco configurado."""
    df = _obter_experimentos_postgres()
    assert isinstance(df, pd.DataFrame)


def test_obter_artefatos_mongodb_retorna_lista():
    """Deve retornar lista (possivelmente vazia) sem banco MongoDB."""
    artefatos = _obter_artefatos_mongodb()
    assert isinstance(artefatos, list)


def test_renderizar_bancos_nao_levanta():
    renderizar_bancos()


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes adicionais: painel_benchmarks.py ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_executar_benchmark_sucesso():
    """_executar_benchmark deve popular resultados com métricas em sucesso."""
    fachada = _fachada_mock()
    resultados = {}
    _mock_st.progress.return_value = MagicMock()
    _executar_benchmark(fachada, ["RegressaoLogistica"], resultados)
    assert "RegressaoLogistica" in resultados
    assert resultados["RegressaoLogistica"]["acuracia"] == pytest.approx(0.95)


def test_executar_benchmark_erro_modelo():
    """_executar_benchmark deve registrar erro sem lançar exceção."""
    fachada = _fachada_mock()
    fachada.treinar_modelo.side_effect = RuntimeError("falhou")
    resultados = {}
    _mock_st.progress.return_value = MagicMock()
    _executar_benchmark(fachada, ["ModeloRuim"], resultados)
    assert "ModeloRuim" in resultados
    assert resultados["ModeloRuim"]["acuracia"] == 0
    assert "erro" in resultados["ModeloRuim"]


def test_renderizar_kpis_chama_columns():
    """_renderizar_kpis deve criar 3 colunas e exibir tiles."""
    df = _formatar_tabela(_tres_modelos())
    _mock_st.reset_mock()
    _renderizar_kpis(df)
    _mock_st.columns.assert_called()


def test_renderizar_graficos_sem_plotly_retorna():
    """_renderizar_graficos com PLOTLY_OK=False não deve chamar st.plotly_chart."""
    import src.frontend.painel_benchmarks as pb
    original = pb.PLOTLY_OK
    pb.PLOTLY_OK = False
    _mock_st.reset_mock()
    try:
        df = _formatar_tabela(_tres_modelos())
        _renderizar_graficos(df)
        _mock_st.plotly_chart.assert_not_called()
    finally:
        pb.PLOTLY_OK = original


def test_renderizar_matriz_sem_mat_exibe_info():
    """_renderizar_matriz_confusao com matriz ausente deve chamar st.info."""
    resultados = {"ModeloA": {"acuracia": 0.9}}
    _mock_st.selectbox.return_value = "ModeloA"
    _mock_st.toggle.return_value = False
    _mock_st.reset_mock()
    _renderizar_matriz_confusao(resultados)
    _mock_st.info.assert_called()


def test_renderizar_matriz_com_mat_sem_plotly():
    """_renderizar_matriz_confusao com PLOTLY_OK=False deve sair antes de plotar."""
    import src.frontend.painel_benchmarks as pb
    original = pb.PLOTLY_OK
    pb.PLOTLY_OK = False
    _mock_st.selectbox.return_value = "ModeloA"
    _mock_st.toggle.return_value = False
    _mock_st.reset_mock()
    resultados = {"ModeloA": {"matriz_confusao": [[i == j for j in range(10)] for i in range(10)]}}
    try:
        _renderizar_matriz_confusao(resultados)
        _mock_st.plotly_chart.assert_not_called()
    finally:
        pb.PLOTLY_OK = original


def test_renderizar_benchmarks_com_resultados_em_session():
    """renderizar deve exibir tabela quando resultados_benchmark estiver em session_state."""
    from itertools import cycle
    fachada = _fachada_mock()
    _mock_st.session_state["resultados_benchmark"] = _tres_modelos()
    _mock_st.button.return_value = False
    _mock_st.multiselect.return_value = ["RegressaoLogistica"]
    # selectbox é chamado: primeiro para ordenar ("Acurácia"), depois para modelo da matriz ("ModeloA")
    _mock_st.selectbox.side_effect = cycle(["Acurácia", "ModeloA"])
    _mock_st.toggle.return_value = False
    _mock_st.reset_mock()
    _mock_st.session_state["resultados_benchmark"] = _tres_modelos()
    _mock_st.selectbox.side_effect = cycle(["Acurácia", "ModeloA"])
    _mock_st.toggle.return_value = False
    with patch.object(pd.DataFrame, 'style', new_callable=lambda: property(lambda self: MagicMock())):
        renderizar_bench(fachada)
    assert _mock_st.dataframe.called
    _mock_st.selectbox.side_effect = None


def test_renderizar_benchmarks_vazio_exibe_info():
    """renderizar deve exibir st.info quando não há resultados."""
    fachada = _fachada_mock()
    _mock_st.session_state.pop("resultados_benchmark", None)
    _mock_st.button.return_value = False
    _mock_st.multiselect.return_value = []
    _mock_st.reset_mock()
    renderizar_bench(fachada)
    _mock_st.info.assert_called()


def test_renderizar_benchmarks_executa_quando_botao():
    """renderizar deve chamar _executar_benchmark quando executar=True."""
    fachada = _fachada_mock()
    _mock_st.session_state.pop("resultados_benchmark", None)
    _mock_st.button.return_value = True
    modelos = ["RegressaoLogistica", "FlorestaAleatoria", "SVM"]
    _mock_st.multiselect.return_value = modelos
    _mock_st.selectbox.side_effect = iter(["Acurácia", "RegressaoLogistica"])
    _mock_st.toggle.return_value = False
    _mock_st.progress.return_value = MagicMock()
    with patch.object(pd.DataFrame, "style", new_callable=lambda: property(lambda self: MagicMock())):
        renderizar_bench(fachada)
    fachada.treinar_modelo.assert_called()
    _mock_st.selectbox.side_effect = None


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes adicionais: painel_laboratorio_visao.py ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_inferir_com_modelo_com_predict_proba():
    """_inferir_com_modelo deve usar predict_proba quando disponível."""
    from src.frontend.painel_laboratorio_visao import _inferir_com_modelo
    mock_sklearn = MagicMock()
    mock_sklearn.predict_proba.return_value = np.array([[0.1] * 10])
    mock_modelo = MagicMock()
    mock_modelo.modelo = mock_sklearn
    fachada = MagicMock()
    fachada.modelos = {"SVM": mock_modelo}
    vetor = np.zeros((1, 784))
    resultado = _inferir_com_modelo(fachada, vetor)
    assert resultado is not None
    assert len(resultado) == 10


def test_inferir_com_modelo_sem_predict_proba():
    """_inferir_com_modelo usa prever() quando predict_proba não existe."""
    from src.frontend.painel_laboratorio_visao import _inferir_com_modelo
    mock_sklearn = MagicMock(spec=[])  # sem predict_proba
    mock_modelo = MagicMock()
    mock_modelo.modelo = mock_sklearn
    mock_modelo.prever.return_value = np.array([3])
    fachada = MagicMock()
    fachada.modelos = {"KNN": mock_modelo}
    resultado = _inferir_com_modelo(fachada, np.zeros((1, 784)))
    assert resultado is not None
    assert resultado[3][1] == 1.0


def test_inferir_com_modelo_sem_modelos_retorna_none():
    """_inferir_com_modelo retorna None quando fachada.modelos está vazio."""
    from src.frontend.painel_laboratorio_visao import _inferir_com_modelo
    fachada = MagicMock()
    fachada.modelos = {}
    assert _inferir_com_modelo(fachada, np.zeros((1, 784))) is None


def test_inferir_com_modelo_excecao_continua():
    """_inferir_com_modelo deve tentar próximo modelo se o atual lançar exceção."""
    from src.frontend.painel_laboratorio_visao import _inferir_com_modelo
    mock_ruim = MagicMock()
    mock_ruim.modelo.predict_proba.side_effect = RuntimeError("erro")
    fachada = MagicMock()
    fachada.modelos = {"Ruim": mock_ruim}
    assert _inferir_com_modelo(fachada, np.zeros((1, 784))) is None


def test_grafico_topk_sem_plotly_usa_bar_chart():
    """_grafico_topk com PLOTLY_OK=False deve chamar st.bar_chart."""
    import src.frontend.painel_laboratorio_visao as plv
    original = plv.PLOTLY_OK
    plv.PLOTLY_OK = False
    _mock_st.reset_mock()
    ranking = [(i, 0.1) for i in range(10)]
    try:
        from src.frontend.painel_laboratorio_visao import _grafico_topk
        _grafico_topk(ranking)
        _mock_st.bar_chart.assert_called()
    finally:
        plv.PLOTLY_OK = original


def test_renderizar_pipeline_e_inferencia_sem_modelo():
    """_renderizar_pipeline_e_inferencia deve exibir st.info sem modelo treinado."""
    fachada = MagicMock()
    fachada.modelos = {}
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    _mock_st.reset_mock()
    _renderizar_pipeline_e_inferencia(fachada, img)
    _mock_st.info.assert_called()


def test_renderizar_pipeline_e_inferencia_com_modelo():
    """_renderizar_pipeline_e_inferencia deve chamar kpi_tile com modelo treinado."""
    mock_sklearn = MagicMock()
    mock_sklearn.predict_proba.return_value = np.array([[0.1] * 10])
    mock_modelo = MagicMock()
    mock_modelo.modelo = mock_sklearn
    fachada = MagicMock()
    fachada.modelos = {"SVM": mock_modelo}
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    _mock_st.reset_mock()
    _renderizar_pipeline_e_inferencia(fachada, img)
    assert _mock_st.markdown.called


def test_renderizar_lab_sem_modelos_avisa():
    """renderizar do laboratório deve chamar st.warning sem modelos treinados."""
    fachada = MagicMock()
    fachada.modelos = {}
    _mock_st.radio.return_value = "✍️ Canvas (Desenho)"
    # canvas retorna image_data=None para que _renderizar_modo_canvas retorne None
    import sys as _sys
    _sys.modules["streamlit_drawable_canvas"].st_canvas.return_value.image_data = None
    _mock_st.reset_mock()
    renderizar_lab(fachada)
    _mock_st.warning.assert_called()


def test_renderizar_lab_canvas_nenhum_dado():
    """renderizar com Canvas retornando None não deve chamar _renderizar_pipeline."""
    import sys as _sys
    fachada = MagicMock()
    fachada.modelos = {"SVM": MagicMock()}
    _mock_st.radio.return_value = "✍️ Canvas (Desenho)"
    _sys.modules["streamlit_drawable_canvas"].st_canvas.return_value.image_data = None
    _mock_st.reset_mock()
    renderizar_lab(fachada)
    # _renderizar_pipeline_e_inferencia não é chamado (img_array is None)
    _mock_st.divider.assert_not_called()


def test_renderizar_lab_upload_sem_arquivo():
    """renderizar com Upload retornando None (sem arquivo) não chama pipeline."""
    import sys as _sys
    fachada = MagicMock()
    fachada.modelos = {"SVM": MagicMock()}
    _mock_st.radio.return_value = "📷 Upload de Imagem"
    _sys.modules["streamlit_drawable_canvas"].st_canvas.return_value.image_data = None
    _mock_st.file_uploader.return_value = None
    _mock_st.reset_mock()
    renderizar_lab(fachada)
    _mock_st.divider.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes adicionais: painel_robustez_ood.py ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _resultado_ood_mock():
    """Cria resultado OOD simulado para testes do painel."""
    probs_ood = np.ones((10, 10)) / 10
    probs_ind = np.ones((10, 10)) / 10
    val = ValidadorFalsaCerteza()
    classes = list(range(10))
    df_ood = _avaliar_lote(probs_ood, classes, val)
    df_ind = _avaliar_lote(probs_ind, classes, val)
    return {
        "df_ood": df_ood, "df_ind": df_ind,
        "probs_ood": probs_ood, "probs_ind": probs_ind,
        "classes_mascaradas": [4, 7],
        "classes_conhecidas": [0, 1, 2, 3, 5, 6, 8, 9],
        "fonte": "simulação",
    }


def test_renderizar_ood_sem_sessao_exibe_info():
    """renderizar OOD deve exibir st.info quando nenhum experimento foi executado."""
    _mock_st.session_state.pop("resultado_ood", None)
    _mock_st.button.return_value = False
    _mock_st.multiselect.return_value = [4, 7]
    _mock_st.slider.return_value = 200
    _mock_st.reset_mock()
    fachada = _fachada_mock()
    renderizar_ood(fachada)
    _mock_st.info.assert_called()


def test_renderizar_ood_com_resultado_em_sessao():
    """renderizar OOD deve mostrar KPIs quando resultado_ood está em session_state."""
    res = _resultado_ood_mock()
    _mock_st.session_state["resultado_ood"] = res
    _mock_st.button.return_value = False
    _mock_st.multiselect.return_value = [4, 7]
    _mock_st.slider.return_value = 200
    _mock_st.toggle.return_value = False
    _mock_st.reset_mock()
    fachada = _fachada_mock()
    renderizar_ood(fachada)
    # KPIs renderizados → st.markdown deve ter sido chamado
    assert _mock_st.markdown.called


def test_renderizar_ood_botao_executar():
    """renderizar OOD com executar=True deve popular session_state.resultado_ood."""
    _mock_st.session_state.pop("resultado_ood", None)
    _mock_st.button.return_value = True
    _mock_st.multiselect.return_value = [4, 7]
    _mock_st.slider.return_value = 50
    _mock_st.toggle.return_value = False
    fachada = _fachada_mock()
    # Faz executar_experimento_ood levantar exceção → usa simulação
    with patch("src.robustez_ood.executar_experimento_ood", side_effect=RuntimeError("sem dados")):
        renderizar_ood(fachada)
    assert "resultado_ood" in _mock_st.session_state


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes adicionais: painel_assistente_rag.py ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def test_inicializar_estado_cria_chaves():
    """_inicializar_estado deve criar historico_chat e rag_pronto na sessão."""
    from src.frontend.painel_assistente_rag import _inicializar_estado
    _mock_st.session_state.pop("historico_chat", None)
    _mock_st.session_state.pop("rag_pronto", None)
    _inicializar_estado()
    assert "historico_chat" in _mock_st.session_state
    assert "rag_pronto" in _mock_st.session_state


def test_carregar_assistente_sem_rag_retorna_none():
    """_carregar_assistente deve retornar None se AssistenteRAG falhar."""
    from src.frontend.painel_assistente_rag import _carregar_assistente
    with patch("src.rag.assistente.AssistenteRAG", side_effect=ImportError("sem rag")):
        resultado = _carregar_assistente()
    assert resultado is None


def test_processar_pergunta_fallback():
    """_processar_pergunta sem RAG deve adicionar resposta fallback ao histórico."""
    from src.frontend.painel_assistente_rag import _processar_pergunta
    _mock_st.session_state["historico_chat"] = []
    _mock_st.session_state["rag_pronto"] = False
    _mock_st.session_state["assistente"] = None
    _mock_st.spinner.return_value = _mk_ctx()
    _processar_pergunta("O que é MNIST?")
    historico = _mock_st.session_state["historico_chat"]
    assert len(historico) == 2
    assert historico[0]["papel"] == "usuario"
    assert historico[1]["papel"] == "assistente"


def test_processar_pergunta_com_rag():
    """_processar_pergunta com RAG ativo deve usar assistente.perguntar()."""
    from src.frontend.painel_assistente_rag import _processar_pergunta
    mock_assistente = MagicMock()
    mock_assistente.perguntar.return_value = {"resposta": "70.000 imagens", "fontes": ["mnist.md"]}
    _mock_st.session_state["historico_chat"] = []
    _mock_st.session_state["rag_pronto"] = True
    _mock_st.session_state["assistente"] = mock_assistente
    _mock_st.spinner.return_value = _mk_ctx()
    _processar_pergunta("Quantas imagens tem o MNIST?")
    historico = _mock_st.session_state["historico_chat"]
    assert historico[1]["conteudo"] == "70.000 imagens"
    assert historico[1]["fontes"] == ["mnist.md"]


def test_renderizar_rag_inicializa_estado():
    """renderizar do assistente RAG deve chamar _inicializar_estado."""
    from src.frontend.painel_assistente_rag import renderizar as renderizar_rag2
    _mock_st.session_state.pop("historico_chat", None)
    _mock_st.session_state.pop("rag_pronto", None)
    _mock_st.form.return_value = _mk_ctx()
    _mock_st.form_submit_button = MagicMock(return_value=False)
    renderizar_rag2()
    assert "historico_chat" in _mock_st.session_state


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes adicionais: painel_bancos_dados.py (cobertura expandida) ───────────
# ══════════════════════════════════════════════════════════════════════════════


def test_renderizar_bancos_com_postgres_nao_vazio():
    """renderizar bancos deve exibir KPIs quando postgres retorna DataFrame com linhas."""
    import pandas as pd

    df_fake = pd.DataFrame([{
        "ID": 1,
        "Modelo": "SVM",
        "Acurácia": "0.9700",
        "Tempo Treino (s)": "1.23",
        "Data de Execução": "01/01/2025 10:00:00",
    }])

    cols = _make_columns(3)
    _mock_st.columns.return_value = cols

    with patch("src.frontend.painel_bancos_dados._obter_experimentos_postgres",
               return_value=df_fake), \
         patch("src.frontend.painel_bancos_dados._obter_artefatos_mongodb",
               return_value=[]):
        renderizar_bancos()

    # Com dados, st.dataframe deve ser chamado
    _mock_st.dataframe.assert_called()


def test_renderizar_bancos_postgres_nao_vazio_sem_plotly():
    """renderizar bancos com plotly ausente nao deve levantar excecao."""
    import pandas as pd
    import sys

    df_fake = pd.DataFrame([{
        "ID": i, "Modelo": f"M{i}",
        "Acurácia": f"{0.9 + i * 0.01:.4f}",
        "Tempo Treino (s)": "1.00",
        "Data de Execução": "01/01/2025 10:00:00",
    } for i in range(2)])

    cols = _make_columns(3)
    _mock_st.columns.return_value = cols

    # Simula ausencia de plotly
    plotly_backup = sys.modules.pop("plotly", None)
    plotly_express_backup = sys.modules.pop("plotly.express", None)
    sys.modules["plotly"] = None  # type: ignore
    sys.modules["plotly.express"] = None  # type: ignore

    try:
        with patch("src.frontend.painel_bancos_dados._obter_experimentos_postgres",
                   return_value=df_fake), \
             patch("src.frontend.painel_bancos_dados._obter_artefatos_mongodb",
                   return_value=[]):
            renderizar_bancos()
    finally:
        if plotly_backup is not None:
            sys.modules["plotly"] = plotly_backup
        else:
            sys.modules.pop("plotly", None)
        if plotly_express_backup is not None:
            sys.modules["plotly.express"] = plotly_express_backup
        else:
            sys.modules.pop("plotly.express", None)


def test_renderizar_bancos_com_mongodb_nao_vazio():
    """renderizar bancos deve exibir lista quando mongodb retorna artefatos."""
    artefatos_fake = [
        {"nome": "matriz_svm", "dados": {"acuracia": 0.95}, "salvo_em": "01/01/2025 10:00:00"},
        {"nome": "relatorio_ood", "dados": {"matriz": [[1, 0], [0, 1]]}, "salvo_em": "02/01/2025 12:00:00"},
    ]

    cols = _make_columns(2)
    _mock_st.columns.return_value = cols
    _mock_st.expander.return_value = _mk_ctx()

    with patch("src.frontend.painel_bancos_dados._obter_experimentos_postgres",
               return_value=__import__("pandas").DataFrame()), \
         patch("src.frontend.painel_bancos_dados._obter_artefatos_mongodb",
               return_value=artefatos_fake):
        renderizar_bancos()

    # Com artefatos, st.expander deve ser chamado para cada um
    assert _mock_st.expander.call_count >= 2


def test_renderizar_bancos_mongodb_com_matriz_confusao():
    """renderizar bancos com dados de matriz deve tentar exibir DataFrame."""
    import pandas as pd

    artefatos_fake = [
        {"nome": "matriz_lr", "dados": {"matriz_confusao": [[9, 1], [2, 8]]}, "salvo_em": "—"},
    ]

    cols = _make_columns(2)
    _mock_st.columns.return_value = cols
    _mock_st.expander.return_value = _mk_ctx()

    with patch("src.frontend.painel_bancos_dados._obter_experimentos_postgres",
               return_value=pd.DataFrame()), \
         patch("src.frontend.painel_bancos_dados._obter_artefatos_mongodb",
               return_value=artefatos_fake):
        renderizar_bancos()

    _mock_st.expander.assert_called()


def test_obter_artefatos_mongodb_enriquece_com_timestamp(tmp_path, monkeypatch):
    """_obter_artefatos_mongodb em modo local deve adicionar salvo_em se arquivo existe."""
    import json

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "artefato_ts.json").write_text(json.dumps({"k": "v"}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with patch("src.frontend.painel_bancos_dados.ConexaoMongoDB") as mock_cls:
        mock_conn = MagicMock()
        mock_conn.usar_local = True
        mock_conn.listar_colecao.return_value = [{"nome": "artefato_ts", "dados": {"k": "v"}}]
        mock_cls.return_value = mock_conn

        from src.frontend.painel_bancos_dados import _obter_artefatos_mongodb
        resultado = _obter_artefatos_mongodb()

    assert len(resultado) == 1
    assert "salvo_em" in resultado[0]


def test_obter_experimentos_postgres_registros_vazios():
    """_obter_experimentos_postgres com sessao sem registros deve retornar DataFrame vazio — linha 24-25."""
    from src.frontend.painel_bancos_dados import _obter_experimentos_postgres

    with patch("src.frontend.painel_bancos_dados.ConexaoPostgres") as mock_cls:
        mock_db = MagicMock()
        mock_cls.return_value = mock_db
        mock_sessao = MagicMock()
        mock_db.obter_sessao.return_value.__enter__ = MagicMock(return_value=mock_sessao)
        mock_db.obter_sessao.return_value.__exit__ = MagicMock(return_value=False)
        mock_sessao.query.return_value.order_by.return_value.all.return_value = []

        resultado = _obter_experimentos_postgres()

    import pandas as pd
    assert isinstance(resultado, pd.DataFrame)
    assert resultado.empty


def test_renderizar_bancos_mongodb_matriz_malformada():
    """Artefato com matriz invalida deve cair no except e chamar st.json — linhas 201-202."""
    import pandas as pd

    artefatos_fake = [
        # "matriz" existe mas nao pode virar DataFrame (None invalido)
        {"nome": "quebrado", "dados": {"matriz": 42}, "salvo_em": "—"},
    ]

    _mock_st.expander.return_value = _mk_ctx()
    cols = _make_columns(2)
    _mock_st.columns.return_value = cols

    with patch("src.frontend.painel_bancos_dados._obter_experimentos_postgres",
               return_value=pd.DataFrame()), \
         patch("src.frontend.painel_bancos_dados._obter_artefatos_mongodb",
               return_value=artefatos_fake):
        renderizar_bancos()

    # st.json deve ter sido chamado no fallback
    _mock_st.json.assert_called()


def test_obter_experimentos_postgres_com_registros():
    """_obter_experimentos_postgres com registros deve retornar DataFrame populado — linha 26."""
    from datetime import datetime
    from src.frontend.painel_bancos_dados import _obter_experimentos_postgres

    registro = MagicMock()
    registro.id = 1
    registro.modelo = "SVM"
    registro.acuracia = 0.97
    registro.tempo_treino = 1.23
    registro.data_execucao = datetime(2025, 1, 1, 10, 0, 0)

    with patch("src.frontend.painel_bancos_dados.ConexaoPostgres") as mock_cls:
        mock_db = MagicMock()
        mock_cls.return_value = mock_db
        mock_sessao = MagicMock()
        mock_db.obter_sessao.return_value.__enter__ = MagicMock(return_value=mock_sessao)
        mock_db.obter_sessao.return_value.__exit__ = MagicMock(return_value=False)
        mock_sessao.query.return_value.order_by.return_value.all.return_value = [registro]

        resultado = _obter_experimentos_postgres()

    import pandas as pd
    assert not resultado.empty
    assert resultado.iloc[0]["Modelo"] == "SVM"
    assert "Acurácia" in resultado.columns


# ══════════════════════════════════════════════════════════════════════════════
# ── Testes adicionais: painel_laboratorio_visao.py (cobertura expandida) ──────
# ══════════════════════════════════════════════════════════════════════════════


def test_pipeline_visual_sem_contornos_usa_invertida():
    """_pipeline_visual com imagem sem contornos deve usar invertida como bbox_crop — linha 119."""
    # Imagem all-white → apos inversao fica all-black → findNonZero retorna None
    img_branca = np.ones((28, 28, 3), dtype=np.uint8) * 255
    gray, invertida, bbox_crop, canvas_28 = _pipeline_visual(img_branca)
    # bbox_crop deve ser igual a invertida (o else: bbox_crop = invertida)
    assert np.array_equal(bbox_crop, invertida)
    assert canvas_28.shape == (28, 28)


def test_renderizar_modo_canvas_com_image_data():
    """_renderizar_modo_canvas com image_data nao-None deve retornar array RGB — linhas 150-151."""
    from src.frontend.painel_laboratorio_visao import _renderizar_modo_canvas

    img_data = np.zeros((280, 280, 4), dtype=np.uint8)  # RGBA
    img_data[100:180, 100:180, :3] = 200  # bright region
    img_data[:, :, 3] = 255  # alpha

    sys.modules['streamlit_drawable_canvas'].st_canvas.return_value.image_data = img_data
    cols = _make_columns(2)
    _mock_st.columns.return_value = cols
    _mock_st.slider.return_value = 20
    _mock_st.color_picker.return_value = "#FFFFFF"

    resultado = _renderizar_modo_canvas()

    # Restaura para None para outros testes
    sys.modules['streamlit_drawable_canvas'].st_canvas.return_value.image_data = None
    assert resultado is not None
    assert resultado.shape[2] == 3  # RGB, sem alpha


def test_renderizar_pipeline_e_inferencia_pipeline_falha():
    """_renderizar_pipeline_e_inferencia com _pipeline_visual falhando deve retornar — linhas 203-205."""
    fachada = _fachada_mock()
    img = np.zeros((28, 28, 3), dtype=np.uint8)
    img[10:20, 10:20] = 100

    with patch("src.frontend.painel_laboratorio_visao._pipeline_visual",
               side_effect=RuntimeError("cv2 indisponivel")):
        _renderizar_pipeline_e_inferencia(fachada, img)

    _mock_st.error.assert_called()


def test_renderizar_pipeline_e_inferencia_overconfidence():
    """Predicao com alerta_overconfidence=True deve exibir st.warning — linha 225."""
    fachada = _fachada_mock()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[15:35, 15:35] = 200

    probs_mock = [(i, 1.0 if i == 0 else 0.0) for i in range(10)]
    cols = _make_columns(4)
    _mock_st.columns.return_value = cols
    _mock_st.expander.return_value = _mk_ctx()

    with patch("src.frontend.painel_laboratorio_visao._inferir_com_modelo",
               return_value=probs_mock), \
         patch("guardrails.validador_falsa_certeza.ValidadorFalsaCerteza") as mock_val:
        mock_inst = MagicMock()
        mock_val.return_value = mock_inst
        mock_inst.avaliar_predicao.return_value = {"alerta_overconfidence": True}
        _renderizar_pipeline_e_inferencia(fachada, img)

    _mock_st.warning.assert_called()


def test_renderizar_lab_com_imagem_upload():
    """renderizar laboratorio em modo upload com imagem valida chama _renderizar_pipeline_e_inferencia — linha 256."""
    fachada = _fachada_mock()
    img_fake = np.zeros((50, 50, 3), dtype=np.uint8)
    img_fake[10:40, 10:40] = 128

    _mock_st.radio.return_value = "📷 Upload de Imagem"
    cols = _make_columns(4)
    _mock_st.columns.return_value = cols
    _mock_st.expander.return_value = _mk_ctx()

    with patch("src.frontend.painel_laboratorio_visao._renderizar_modo_upload",
               return_value=img_fake), \
         patch("src.frontend.painel_laboratorio_visao._renderizar_pipeline_e_inferencia") as mock_pipe:
        renderizar_lab(fachada)

    mock_pipe.assert_called_once_with(fachada, img_fake)
