"""Painel 3 — Benchmarks: comparação dinâmica, KPIs e matrizes de confusão 10×10.

Fluxo de renderização
---------------------
1. ``renderizar(fachada)`` é chamado pelo ``app.py`` com uma instância de
   ``FachadaPipelineIA`` já configurada.
2. A lista de modelos é obtida *exclusivamente* via
   ``FabricaModelos.listar_disponiveis()``, eliminando qualquer hardcode.
3. O benchmark executa cada modelo dentro de ``try/except``; falhas são
   marcadas com ``_falhou=True`` e nunca geram métricas zeradas silenciosas.
4. Resultados são separados em **válidos** (executaram com sucesso) e
   **com falha** (levantaram exceção); cada grupo recebe tratamento visual
   distinto.
5. Gráficos, KPIs e a matriz de confusão são renderizados apenas sobre o
   subconjunto de resultados válidos.

Contrato com a fábrica
-----------------------
- ``FabricaModelos.listar_disponiveis() -> list[str]``:
  retorna as chaves canônicas aceitas por ``criar_modelo()``.
- ``FachadaPipelineIA.treinar_modelo(nome: str)``
- ``FachadaPipelineIA.avaliar_modelo(nome: str) -> dict``
  → chaves obrigatórias: ``acuracia``, ``precisao``, ``recall``,
  ``f1``, ``matriz_confusao``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.frontend.estilos import aplicar_estilos, kpi_tile, titulo_secao
from src.modelos.fabrica_modelos import FabricaModelos

logger = logging.getLogger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _PLOTLY_OK = True
except ImportError:  # pragma: no cover
    _PLOTLY_OK = False

_TEMA_PLOTLY: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "template": "plotly_dark",
}

# Sentinel interno: nunca exposto ao usuário final.
_CHAVE_FALHOU = "_falhou"
_CHAVE_ERRO = "erro"


# ──────────────────────────────────────────────────────────────────────────────
# Descoberta dinâmica de modelos
# ──────────────────────────────────────────────────────────────────────────────


def _obter_modelos_disponiveis() -> list[str]:
    """Consulta o registro oficial da fábrica e retorna os nomes suportados.

    Nunca usa lista estática — qualquer alteração em ``FabricaModelos``
    é automaticamente refletida aqui.

    Returns:
        list[str]: Chaves válidas para ``FabricaModelos.criar_modelo()``.
    """
    try:
        return FabricaModelos.listar_disponiveis()
    except Exception as exc:  # pragma: no cover
        logger.error("[Benchmarks] Falha ao consultar FabricaModelos: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Execução do benchmark
# ──────────────────────────────────────────────────────────────────────────────


def _executar_benchmark(
    fachada: Any,
    modelos_sel: list[str],
    resultados: dict[str, dict[str, Any]],
) -> None:
    """Treina cada modelo selecionado e armazena métricas reais na sessão.

    Modelos que falham recebem ``{_CHAVE_FALHOU: True, _CHAVE_ERRO: <msg>}``
    e *nunca* recebem métricas numéricas zeradas que possam mascarar a falha.

    Args:
        fachada: Instância de ``FachadaPipelineIA``.
        modelos_sel: Lista de nomes de modelos a executar.
        resultados: Dicionário de sessão; alterado in-place.
    """
    modelos_validos = set(FabricaModelos.listar_disponiveis())
    barra = st.progress(0, text="Iniciando benchmark…")

    for i, nome in enumerate(modelos_sel):
        barra.progress(i / len(modelos_sel), text=f"⏳ Treinando **{nome}**…")

        # Valida contra o registro antes de qualquer I/O
        if nome not in modelos_validos:
            msg = f"Modelo '{nome}' não está registrado na fábrica."
            logger.warning("[Benchmarks] %s", msg)
            resultados[nome] = {_CHAVE_FALHOU: True, _CHAVE_ERRO: msg}
            continue

        try:
            t0 = time.perf_counter()
            fachada.treinar_modelo(nome)
            metricas = fachada.avaliar_modelo(nome)
            metricas["tempo_treino"] = round(time.perf_counter() - t0, 2)
            metricas[_CHAVE_FALHOU] = False
            resultados[nome] = metricas
            logger.info("[Benchmarks] '%s' concluído — acurácia=%.4f", nome, metricas["acuracia"])

        except Exception as exc:
            msg = str(exc)
            logger.error("[Benchmarks] Falha ao executar '%s': %s", nome, msg)
            # Armazena APENAS a falha — zero métricas numéricas para não enganar
            resultados[nome] = {_CHAVE_FALHOU: True, _CHAVE_ERRO: msg}

    barra.progress(1.0, text="✅ Benchmark concluído!")


# ──────────────────────────────────────────────────────────────────────────────
# Separação de resultados
# ──────────────────────────────────────────────────────────────────────────────


def _separar_resultados(
    resultados: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Divide os resultados em válidos e com falha.

    Args:
        resultados: Dicionário completo da sessão.

    Returns:
        Tupla ``(validos, falhas)`` onde ``falhas`` mapeia nome → mensagem de erro.
    """
    validos: dict[str, dict[str, Any]] = {}
    falhas: dict[str, str] = {}
    for nome, dados in resultados.items():
        if dados.get(_CHAVE_FALHOU, False):
            falhas[nome] = dados.get(_CHAVE_ERRO, "Erro desconhecido.")
        else:
            validos[nome] = dados
    return validos, falhas


# ──────────────────────────────────────────────────────────────────────────────
# Formatação da tabela comparativa
# ──────────────────────────────────────────────────────────────────────────────


def _formatar_tabela(validos: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Constrói o DataFrame comparativo apenas com modelos que executaram com sucesso.

    Args:
        validos: Subconjunto de ``resultados`` sem entradas de falha.

    Returns:
        ``pd.DataFrame`` ordenado por acurácia decrescente com coluna de medalha.
    """
    linhas = []
    for nome, m in validos.items():
        linhas.append(
            {
                "Modelo": nome,
                "Acurácia": round(m.get("acuracia", 0.0), 4),
                "Precisão": round(m.get("precisao", 0.0), 4),
                "Recall": round(m.get("recall", 0.0), 4),
                "F1-Score": round(m.get("f1", 0.0), 4),
                "Tempo (s)": round(m.get("tempo_treino", 0.0), 2),
            }
        )
    df = (
        pd.DataFrame(linhas)
        .sort_values("Acurácia", ascending=False)
        .reset_index(drop=True)
    )
    medalhas = ["🥇", "🥈", "🥉"] + [""] * max(0, len(df) - 3)
    df.insert(0, "🏅", medalhas)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Componentes visuais
# ──────────────────────────────────────────────────────────────────────────────


def _renderizar_alertas_falha(falhas: dict[str, str]) -> None:
    """Exibe um bloco de erro para cada modelo que não pôde ser executado.

    Garante que o usuário saiba exatamente *por que* cada modelo está ausente
    da comparação, em vez de aparecer com métricas zeradas.

    Args:
        falhas: Dicionário ``{nome_modelo: mensagem_de_erro}``.
    """
    if not falhas:
        return
    st.divider()
    titulo_secao("⚠️ Modelos com Falha de Execução")
    st.caption(
        "Os modelos abaixo **não geraram métricas** — a falha foi registrada "
        "em log e exibida aqui para diagnóstico."
    )
    for nome, msg in falhas.items():
        with st.expander(f"❌ {nome}", expanded=True):
            st.error(f"**Motivo:** {msg}", icon="🚨")
            st.info(
                "Este modelo foi excluído da tabela comparativa, dos gráficos "
                "e da matriz de confusão para não distorcer os resultados.",
                icon="ℹ️",
            )


def _renderizar_kpis(df: pd.DataFrame) -> None:
    """Exibe KPI tiles de campeão, maior acurácia e modelo mais rápido.

    Args:
        df: DataFrame formatado por ``_formatar_tabela()``.
    """
    melhor = df.iloc[0]
    mais_rapido = df.loc[df["Tempo (s)"].idxmin()]
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_tile(melhor["Modelo"], "🥇 Modelo Campeão"), unsafe_allow_html=True)
    k2.markdown(kpi_tile(f'{melhor["Acurácia"]:.4f}', "Maior Acurácia"), unsafe_allow_html=True)
    k3.markdown(
        kpi_tile(
            f'{mais_rapido["Tempo (s)"]}s',
            f'⚡ Mais Rápido — {mais_rapido["Modelo"]}',
        ),
        unsafe_allow_html=True,
    )


def _renderizar_graficos(df_ord: pd.DataFrame) -> None:
    """Renderiza gráfico de barras agrupadas ou radar multimétrico com Plotly.

    Só é chamado quando ``df_ord`` contém pelo menos um modelo válido;
    nunca renderiza dados zerados de modelos com falha.

    Args:
        df_ord: DataFrame já ordenado pela coluna escolhida pelo usuário.
    """
    if not _PLOTLY_OK:
        st.warning("Plotly não instalado — gráficos indisponíveis.")
        return

    st.divider()
    titulo_secao("Comparação Visual")
    tipo_graf = st.radio(
        "Tipo de gráfico",
        ["Barras — Acurácia × F1", "Radar — Multimétrica"],
        horizontal=True,
    )

    if "Barras" in tipo_graf:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Acurácia",
                x=df_ord["Modelo"],
                y=df_ord["Acurácia"],
                marker_color="#58a6ff",
            )
        )
        fig.add_trace(
            go.Bar(
                name="F1-Score",
                x=df_ord["Modelo"],
                y=df_ord["F1-Score"],
                marker_color="#3fb950",
            )
        )
        fig.update_layout(
            **_TEMA_PLOTLY,
            barmode="group",
            height=350,
            margin=dict(t=10, b=80),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Radar
        categorias = ["Acurácia", "Precisão", "Recall", "F1-Score"]
        fig = go.Figure()
        for _, row in df_ord.iterrows():
            vals = [row[c] for c in categorias] + [row[categorias[0]]]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=categorias + [categorias[0]],
                    fill="toself",
                    name=row["Modelo"],
                )
            )
        fig.update_layout(**_TEMA_PLOTLY, height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


def _renderizar_matriz_confusao(validos: dict[str, dict[str, Any]]) -> None:
    """Exibe a matriz de confusão 10×10 interativa com diagnóstico automático.

    Só oferece modelos que executaram com sucesso no seletor; modelos com
    falha são excluídos para evitar exibição de matriz zerada.

    Args:
        validos: Subconjunto de resultados sem falhas.
    """
    if not validos:
        return

    st.divider()
    titulo_secao("Matriz de Confusão 10×10")

    opcoes = [nome for nome, m in validos.items() if m.get("matriz_confusao")]
    if not opcoes:
        st.info("Nenhum modelo retornou matriz de confusão.", icon="ℹ️")
        return

    modelo_mat = st.selectbox("Selecione o modelo", options=opcoes)
    mat = validos[modelo_mat].get("matriz_confusao")

    if not _PLOTLY_OK:
        st.warning("Plotly não instalado — visualização indisponível.")
        return

    mat_np = np.array(mat)
    normalizar = st.toggle("Normalizar por linha (% por classe verdadeira)", value=False)

    if normalizar:
        soma = mat_np.sum(axis=1, keepdims=True)
        soma[soma == 0] = 1
        mat_plot = np.round(mat_np / soma * 100, 1)
        sufixo = "%"
    else:
        mat_plot = mat_np
        sufixo = ""

    fig_cm = px.imshow(
        mat_plot,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Previsto", y="Real", color=f"Contagem{sufixo}"),
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)],
    )
    fig_cm.update_layout(**_TEMA_PLOTLY, height=500, margin=dict(t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    erros = mat_np.sum(axis=1) - np.diag(mat_np)
    classe_pior = int(np.argmax(erros))
    st.markdown(
        f"🔎 **Diagnóstico:** A classe com mais erros é o dígito **{classe_pior}** "
        f"({int(erros[classe_pior])} erros). Verifique a linha {classe_pior} da matriz acima."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────────────────────────────────────


def renderizar(fachada: Any) -> None:
    """Ponto de entrada do Painel 3 — recebe ``FachadaPipelineIA`` inicializada.

    A lista de modelos é obtida dinamicamente via ``FabricaModelos.listar_disponiveis()``;
    modelos inexistentes ou com falha de execução jamais geram métricas zeradas
    silenciosas — cada falha produz alerta visual explícito na interface.

    Args:
        fachada: Instância de ``FachadaPipelineIA`` pronta para uso.
    """
    aplicar_estilos()
    st.markdown("## 🏆 Benchmarks & Comparação de Modelos")
    st.caption(
        "Treine e compare os algoritmos disponíveis. "
        "A lista é sincronizada automaticamente com o registro da fábrica de modelos."
    )

    # ── Descoberta dinâmica ──────────────────────────────────────────────────
    catalogo = _obter_modelos_disponiveis()
    if not catalogo:
        st.error(
            "Não foi possível obter a lista de modelos da fábrica. "
            "Verifique `FabricaModelos.listar_disponiveis()` e os logs.",
            icon="🚨",
        )
        return

    # ── Configuração do benchmark ────────────────────────────────────────────
    titulo_secao("Configuração do Benchmark")
    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        padroes = [m for m in ["RegressaoLogistica", "FlorestaAleatoria", "SVM", "KNN"] if m in catalogo]
        modelos_sel: list[str] = st.multiselect(
            "Modelos para incluir no benchmark",
            options=catalogo,
            default=padroes,
            help="Lista obtida do registro oficial de `FabricaModelos`.",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        executar = st.button("▶ Executar Benchmark", type="primary", use_container_width=True)

    if "resultados_benchmark" not in st.session_state:
        st.session_state.resultados_benchmark = {}

    if executar:
        if not modelos_sel:
            st.warning("Selecione ao menos um modelo antes de executar.", icon="⚠️")
        else:
            # Limpa resultados anteriores dos modelos que serão re-executados
            for nome in modelos_sel:
                st.session_state.resultados_benchmark.pop(nome, None)
            _executar_benchmark(fachada, modelos_sel, st.session_state.resultados_benchmark)

    resultados: dict[str, dict[str, Any]] = st.session_state.resultados_benchmark
    if not resultados:
        st.info("Selecione os modelos e clique em **Executar Benchmark** para começar.")
        return

    # ── Separação: válidos × com falha ──────────────────────────────────────
    validos, falhas = _separar_resultados(resultados)

    # Alertas de falha sempre visíveis, no topo da seção de resultados
    _renderizar_alertas_falha(falhas)

    if not validos:
        st.error(
            "Nenhum modelo executou com sucesso. "
            "Consulte os alertas acima e os logs do servidor.",
            icon="🚨",
        )
        return

    # ── KPIs ────────────────────────────────────────────────────────────────
    st.divider()
    titulo_secao("Destaques")
    df = _formatar_tabela(validos)
    _renderizar_kpis(df)

    # ── Tabela comparativa ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    titulo_secao("Tabela Comparativa")

    if falhas:
        st.caption(
            f"ℹ️ {len(falhas)} modelo(s) excluído(s) por falha de execução: "
            + ", ".join(f"**{n}**" for n in falhas)
        )

    col_ord, _ = st.columns([2, 4])
    with col_ord:
        coluna_ord: str = st.selectbox(
            "Ordenar por", ["Acurácia", "F1-Score", "Precisão", "Recall", "Tempo (s)"]
        )

    df_ord = df.sort_values(
        coluna_ord, ascending=(coluna_ord == "Tempo (s)")
    ).reset_index(drop=True)

    st.dataframe(
        df_ord.style.background_gradient(subset=["Acurácia", "F1-Score"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Gráficos ─────────────────────────────────────────────────────────────
    _renderizar_graficos(df_ord)

    # ── Matriz de confusão ───────────────────────────────────────────────────
    _renderizar_matriz_confusao(validos)
