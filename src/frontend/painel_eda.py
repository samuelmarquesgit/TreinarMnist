"""Painel 1 — Análise Exploratória de Dados (EDA) interativa do MNIST."""

import numpy as np
import pandas as pd
import streamlit as st

from src.frontend.estilos import aplicar_estilos, kpi_tile, titulo_secao

try:
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

_TEMA = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark")


def renderizar(fachada) -> None:
    """Ponto de entrada do Painel 1 — recebe a FachadaPipelineIA já inicializada."""
    aplicar_estilos()
    st.markdown("## 📊 Análise Exploratória de Dados (EDA)")
    st.caption(
        "Inspeção visual e estatística do dataset MNIST (70 000 imagens, dígitos 0–9).")

    X_treino = fachada.X_treino
    X_teste = fachada.X_teste
    y_treino = fachada.y_treino
    y_teste = fachada.y_teste
    X_full = np.vstack([X_treino, X_teste])
    y_full = np.concatenate([y_treino, y_teste])

    # ── KPIs ──────────────────────────────────────────────────────────────
    titulo_secao("Visão Geral do Dataset")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_tile(f"{len(X_full):,}",
                "Total de Amostras"), unsafe_allow_html=True)
    k2.markdown(kpi_tile(f"{len(X_treino):,}",
                "Treino (80%)"), unsafe_allow_html=True)
    k3.markdown(kpi_tile(f"{len(X_teste):,}",
                "Teste (20%)"), unsafe_allow_html=True)
    k4.markdown(kpi_tile("784", "Features (28×28px)"), unsafe_allow_html=True)

    st.divider()

    # ── Balanceamento de classes ───────────────────────────────────────────
    titulo_secao("Distribuição e Balanceamento de Classes")
    classes, contagens = np.unique(y_full, return_counts=True)
    df_bal = pd.DataFrame({"Dígito": classes.astype(str),
                           "Quantidade": contagens,
                           "Proporção (%)": (contagens / len(y_full) * 100).round(2)})

    col_tab, col_graf = st.columns([1, 2])
    with col_tab:
        st.dataframe(df_bal, use_container_width=True, hide_index=True)
    with col_graf:
        if PLOTLY_OK:
            fig = px.bar(
                df_bal,
                x="Dígito",
                y="Quantidade",
                color="Quantidade",
                color_continuous_scale="Blues",
                text="Quantidade",
                labels={
                    "Quantidade": "Nº de amostras"})
            fig.update_traces(textposition="outside")
            fig.update_layout(**_TEMA, coloraxis_showscale=False,
                              margin=dict(t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df_bal.set_index("Dígito")["Quantidade"])

    st.divider()

    # ── Grade de amostras 2×5 ─────────────────────────────────────────────
    titulo_secao("Grade de Amostras (um exemplo por dígito)")
    st.caption(
        "Primeira amostra encontrada no conjunto de treino para cada classe 0–9.")

    n_cols = st.slider(
        "Colunas na grade",
        min_value=2,
        max_value=10,
        value=5,
        step=1)
    int(np.ceil(10 / n_cols))

    # Encontra um índice por classe
    indices_exemplo = {}
    for cls in range(10):
        idxs = np.where(y_treino == cls)[0]
        if len(idxs):
            indices_exemplo[cls] = idxs[0]

    colunas = st.columns(n_cols)
    for i, (cls, idx) in enumerate(sorted(indices_exemplo.items())):
        img = X_treino[idx].reshape(28, 28)
        with colunas[i % n_cols]:
            st.image(img, caption=f"Dígito {cls}", width=80, clamp=True)

    st.divider()

    # ── Inspetor de dígito individual ─────────────────────────────────────
    titulo_secao("Inspetor de Dígito Individual")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        digito_sel = st.selectbox(
            "Escolha o dígito (classe)", options=list(
                range(10)), index=0)
    with col_sel2:
        idxs_cls = np.where(y_treino == digito_sel)[0]
        amostra_n = st.slider(
            "Nº da amostra dentro da classe", 0, min(
                len(idxs_cls) - 1, 99), 0)

    idx_insp = idxs_cls[amostra_n]
    img_insp = X_treino[idx_insp].reshape(28, 28)

    col_img, col_heat, col_hist = st.columns(3)

    with col_img:
        st.markdown("**Imagem Original (28×28)**")
        st.image(img_insp, width=120, clamp=True)

    with col_heat:
        st.markdown("**Mapa de Intensidade de Pixel**")
        if PLOTLY_OK:
            # Reconverte para [0,255] para exibição intuitiva
            img_255 = (img_insp * 255).astype(int)
            fig_h = px.imshow(img_255, color_continuous_scale="Blues",
                              labels=dict(color="Intensidade"),
                              zmin=0, zmax=255)
            fig_h.update_layout(**_TEMA, margin=dict(t=5, b=5), height=220,
                                coloraxis_showscale=True)
            fig_h.update_xaxes(showticklabels=False)
            fig_h.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.write("Instale plotly para o heatmap.")

    with col_hist:
        st.markdown("**Distribuição de Brilho (histograma)**")
        if PLOTLY_OK:
            pixels = img_insp.flatten()
            fig_bar = px.histogram(
                x=pixels,
                nbins=32,
                labels={
                    "x": "Intensidade [0,1]",
                    "y": "Frequência"},
                color_discrete_sequence=["#58a6ff"])
            fig_bar.update_layout(**_TEMA, margin=dict(t=5, b=5), height=220,
                                  showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("Instale plotly para o histograma.")

    # Matriz 28×28 numérica (opcional)
    with st.expander("🔢 Ver matriz 28×28 de intensidades brutas"):
        img_255_df = pd.DataFrame((img_insp * 255).astype(int))
        st.dataframe(img_255_df, use_container_width=True)
