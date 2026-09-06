"""Painel 3 — Benchmarks: tabela comparativa, KPIs e matrizes de confusão 10×10."""

import numpy as np
import pandas as pd
import streamlit as st

from src.frontend.estilos import aplicar_estilos, kpi_tile, titulo_secao

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

_TEMA = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark")


def _formatar_tabela(resultados: dict) -> pd.DataFrame:
    linhas = []
    for nome, m in resultados.items():
        linhas.append({
            "Modelo": nome,
            "Acurácia": round(m.get("acuracia", 0), 4),
            "Precisão": round(m.get("precisao", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "F1-Score": round(m.get("f1", 0), 4),
            "Tempo (s)": round(m.get("tempo_treino", 0), 2),
        })
    df = pd.DataFrame(linhas).sort_values(
        "Acurácia",
        ascending=False).reset_index(
        drop=True)
    df.insert(0, "🏅", ["🥇", "🥈", "🥉"] + [""] * max(0, len(df) - 3))
    return df


def _executar_benchmark(fachada, modelos_sel: list, resultados: dict) -> None:
    """Treina os modelos selecionados e armazena métricas na sessão."""
    import time
    barra = st.progress(0, text="Iniciando benchmark...")
    for i, nome in enumerate(modelos_sel):
        barra.progress((i) / len(modelos_sel), text=f"Treinando {nome}...")
        try:
            t0 = time.perf_counter()
            fachada.treinar_modelo(nome)
            metricas = fachada.avaliar_modelo(nome)
            metricas["tempo_treino"] = round(time.perf_counter() - t0, 2)
            resultados[nome] = metricas
        except Exception as e:
            resultados[nome] = {
                "acuracia": 0, "precisao": 0, "recall": 0, "f1": 0,
                "tempo_treino": 0, "erro": str(e),
                "matriz_confusao": [[0] * 10 for _ in range(10)],
            }
    barra.progress(1.0, text="✅ Benchmark concluído!")


def _renderizar_kpis(df) -> None:
    """Exibe os KPI tiles de campeão, acurácia e mais rápido."""
    melhor = df.iloc[0]
    mais_rapido = df.loc[df["Tempo (s)"].idxmin()]
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_tile(melhor["Modelo"], "🥇 Modelo Campeão"), unsafe_allow_html=True)
    k2.markdown(kpi_tile(f'{melhor["Acurácia"]:.4f}', "Maior Acurácia"), unsafe_allow_html=True)
    k3.markdown(
        kpi_tile(f'{mais_rapido["Tempo (s)"]}s', f'⚡ Mais Rápido — {mais_rapido["Modelo"]}'),
        unsafe_allow_html=True,
    )


def _renderizar_graficos(df_ord) -> None:
    """Renderiza gráficos de barras ou radar com Plotly."""
    if not PLOTLY_OK:
        return
    st.divider()
    titulo_secao("Comparação Visual")
    tipo_graf = st.radio("Tipo", ["Barras — Acurácia vs Tempo", "Radar — Multimétrica"], horizontal=True)
    if "Barras" in tipo_graf:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Acurácia", x=df_ord["Modelo"], y=df_ord["Acurácia"], marker_color="#58a6ff"))
        fig.add_trace(go.Bar(name="F1-Score", x=df_ord["Modelo"], y=df_ord["F1-Score"], marker_color="#3fb950"))
        fig.update_layout(**_TEMA, barmode="group", height=350, margin=dict(t=10, b=80), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        categorias = ["Acurácia", "Precisão", "Recall", "F1-Score"]
        fig = go.Figure()
        for _, row in df_ord.iterrows():
            vals = [row[c] for c in categorias] + [row[categorias[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categorias + [categorias[0]],
                fill="toself", name=row["Modelo"]))
        fig.update_layout(**_TEMA, height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


def _renderizar_matriz_confusao(resultados: dict) -> None:
    """Exibe a matriz de confusão 10×10 interativa com diagnóstico."""
    st.divider()
    titulo_secao("Matriz de Confusão 10×10")
    modelo_mat = st.selectbox("Selecione o modelo", options=list(resultados.keys()))
    mat = resultados[modelo_mat].get("matriz_confusao")
    if not mat or not PLOTLY_OK:
        if not mat:
            st.info("Matriz de confusão não disponível para este modelo.")
        return
    mat_np = np.array(mat)
    with_norm = st.toggle("Normalizar por linha (% por classe verdadeira)", value=False)
    if with_norm:
        soma = mat_np.sum(axis=1, keepdims=True)
        soma[soma == 0] = 1
        mat_plot = np.round(mat_np / soma * 100, 1)
        fmt_label = "%"
    else:
        mat_plot = mat_np
        fmt_label = ""
    fig_cm = px.imshow(
        mat_plot, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Previsto", y="Real", color=f"Contagem{fmt_label}"),
        x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
    )
    fig_cm.update_layout(**_TEMA, height=500, margin=dict(t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)
    erros_por_classe = mat_np.sum(axis=1) - np.diag(mat_np)
    classe_pior = int(np.argmax(erros_por_classe))
    st.markdown(
        f'🔎 **Diagnóstico:** A classe com mais erros é o dígito **{classe_pior}** '
        f'({int(erros_por_classe[classe_pior])} erros). Verifique a linha {classe_pior} da matriz acima.'
    )


def renderizar(fachada) -> None:
    """Ponto de entrada do Painel 3 — recebe FachadaPipelineIA já inicializada."""
    aplicar_estilos()
    st.markdown("## 🏆 Benchmarks & Comparação de Modelos")
    st.caption("Treine e compare os 12 algoritmos. Clique em um modelo para ver sua matriz de confusão.")

    MODELOS_DISPONIVEIS = [
        "RegressaoLogistica", "ArvoreDecisao", "FlorestaAleatoria",
        "ImpulsionamentoGradiente", "SVM", "KNN", "NaiveBayes",
        "PerceptronMulticamadas", "VisionTransformer"
    ]

    titulo_secao("Configuração do Benchmark")
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        modelos_sel = st.multiselect(
            "Modelos para incluir no benchmark",
            options=MODELOS_DISPONIVEIS,
            default=["RegressaoLogistica", "FlorestaAleatoria", "SVM", "KNN"],
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        executar = st.button("▶ Executar Benchmark", type="primary", use_container_width=True)

    if "resultados_benchmark" not in st.session_state:
        st.session_state.resultados_benchmark = {}

    if executar and modelos_sel:
        _executar_benchmark(fachada, modelos_sel, st.session_state.resultados_benchmark)

    resultados = st.session_state.resultados_benchmark
    if not resultados:
        st.info("Selecione os modelos e clique em **Executar Benchmark** para começar.")
        return

    st.divider()
    titulo_secao("Destaques")
    df = _formatar_tabela(resultados)
    _renderizar_kpis(df)

    st.markdown("<br>", unsafe_allow_html=True)
    titulo_secao("Tabela Comparativa")
    col_ord, _ = st.columns([2, 4])
    with col_ord:
        coluna_ord = st.selectbox("Ordenar por", ["Acurácia", "F1-Score", "Precisão", "Recall", "Tempo (s)"])
    df_ord = df.sort_values(coluna_ord, ascending=(coluna_ord == "Tempo (s)")).reset_index(drop=True)
    st.dataframe(
        df_ord.style.background_gradient(subset=["Acurácia", "F1-Score"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )

    _renderizar_graficos(df_ord)
    _renderizar_matriz_confusao(resultados)
