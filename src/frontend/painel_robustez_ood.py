"""Painel 4 — Robustez OOD e Falsa Certeza (Overconfidence / Saturação Softmax)."""

import numpy as np
import pandas as pd
import streamlit as st

from src.frontend.estilos import aplicar_estilos, titulo_secao, kpi_tile
from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza

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


def _simular_softmax(
        n: int,
        classes_conhecidas: list[int],
        seed: int = 42) -> np.ndarray:
    """
    Gera vetores softmax simulados para amostras OOD (classes desconhecidas).
    Simula o comportamento de saturação: alta confiança em classes erradas.
    """
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(n):
        # Sorteia uma classe conhecida para receber a maior probabilidade
        classe_destino = rng.choice(classes_conhecidas)
        p = rng.dirichlet(np.ones(10) * 0.5)
        # Força saturação: concentra ~80-95% na classe destino
        fator = rng.uniform(0.80, 0.97)
        p = p * (1 - fator)
        p[classe_destino] += fator
        probs.append(p)
    return np.array(probs)


def _simular_softmax_in_dist(n: int, seed: int = 0) -> np.ndarray:
    """Gera vetores softmax para amostras In-Distribution (distribuição normal de confiança)."""
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(n):
        p = rng.dirichlet(np.ones(10) * 1.5)
        probs.append(p)
    return np.array(probs)


def _avaliar_lote(
        probs: np.ndarray,
        classes_conhecidas: list[int],
        validador: ValidadorFalsaCerteza) -> pd.DataFrame:
    linhas = []
    for i, p in enumerate(probs):
        res = validador.avaliar_predicao(p, classes_conhecidas)
        linhas.append({
            "Amostra": i,
            "Classe Prevista": res["classe_prevista"],
            "Confiança": round(res["confianca"], 4),
            "Entropia": round(res["entropia"], 4),
            "Alerta OOD": "⚠️ Sim" if res["alerta_overconfidence"] else "✅ Não",
            "Confiável": res["confiavel"],
        })
    return pd.DataFrame(linhas)


def renderizar(fachada) -> None:
    """Ponto de entrada do Painel 4 — recebe FachadaPipelineIA já inicializada."""
    aplicar_estilos()
    st.markdown("## 🧪 Testes de Robustez & Generalização OOD")
    st.caption(
        "Experimento de **Class Masking**: classes são ocultadas do treino e apresentadas "
        "ao modelo na inferência para expor o fenômeno de **Falsa Certeza (Overconfidence)**.")

    # ── Configuração do experimento ────────────────────────────────────────
    titulo_secao("Configuração do Experimento")
    col1, col2, col3 = st.columns(3)
    with col1:
        classes_mascaradas = st.multiselect(
            "Classes a mascarar (retiradas do treino)",
            options=list(range(10)),
            default=[4, 7],
        )
    with col2:
        n_amostras_ood = st.slider(
            "Amostras OOD a avaliar", 50, 500, 200, step=50)
    with col3:
        limiar_overconf = st.slider(
            "Limiar de alerta de overconfidence",
            0.70,
            0.99,
            0.85,
            step=0.01)

    classes_conhecidas = [c for c in range(10) if c not in classes_mascaradas]
    executar = st.button("▶ Executar Experimento OOD", type="primary")

    if not executar and "resultado_ood" not in st.session_state:
        st.info("Configure os parâmetros e clique em **Executar Experimento OOD**.")
        return

    if executar:
        with st.spinner("Simulando inferência OOD e avaliando overconfidence..."):
            # Tenta usar o módulo real; cai em simulação se não existir
            try:
                from src.robustez_ood import executar_experimento_ood  # type: ignore
                probs_ood = executar_experimento_ood(
                    fachada, classes_mascaradas=classes_mascaradas, n_amostras=n_amostras_ood)
            except (ImportError, Exception):
                probs_ood = _simular_softmax(
                    n_amostras_ood, classes_conhecidas)

            probs_ind = _simular_softmax_in_dist(n_amostras_ood)

            validador = ValidadorFalsaCerteza(
                limiar_alerta_certeza=limiar_overconf)
            df_ood = _avaliar_lote(probs_ood, classes_conhecidas, validador)
            df_ind = _avaliar_lote(probs_ind, list(range(10)), validador)

            st.session_state.resultado_ood = {
                "df_ood": df_ood, "df_ind": df_ind,
                "probs_ood": probs_ood, "probs_ind": probs_ind,
                "classes_mascaradas": classes_mascaradas,
                "classes_conhecidas": classes_conhecidas,
            }

    res = st.session_state.resultado_ood
    df_ood = res["df_ood"]
    df_ind = res["df_ind"]
    probs_ood = res["probs_ood"]
    probs_ind = res["probs_ind"]

    n_alertas = (df_ood["Alerta OOD"] == "⚠️ Sim").sum()
    taxa_overconf = n_alertas / len(df_ood) * 100

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.divider()
    titulo_secao("Resultados do Experimento")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_tile(str(len(df_ood)),
                         "Amostras OOD Avaliadas"),
                unsafe_allow_html=True)
    k2.markdown(
        kpi_tile(
            f"{n_alertas}",
            "Alertas de Overconfidence"),
        unsafe_allow_html=True)
    k3.markdown(kpi_tile(f"{taxa_overconf:.1f}%",
                "Taxa de Falsa Certeza"), unsafe_allow_html=True)
    k4.markdown(kpi_tile(
        f'{df_ood["Confiança"].mean():.3f}',
        "Confiança Média OOD"
    ), unsafe_allow_html=True)

    if taxa_overconf > 50:
        st.error(
            f"⚠️ **Alta Falsa Certeza detectada!** {taxa_overconf:.1f}%"
            f" das amostras OOD receberam alertas de overconfidence.")
    elif taxa_overconf > 20:
        st.warning(
            f"🟡 Falsa Certeza moderada: {taxa_overconf:.1f}% das amostras OOD com alerta.")
    else:
        st.success(
            f"✅ Baixa Falsa Certeza: apenas {taxa_overconf:.1f}% das amostras OOD com alerta.")

    # ── Distribuição de confiança: OOD vs In-Distribution ─────────────────
    if PLOTLY_OK:
        st.divider()
        titulo_secao("Distribuição de Confiança: In-Distribution vs OOD")
        st.caption(
            "In-Distribution = amostras de classes conhecidas. "
            "OOD = amostras das classes mascaradas nunca vistas no treino."
        )

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_ind["Confiança"], name="In-Distribution",
            marker_color="#3fb950", opacity=0.7, nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=df_ood["Confiança"], name="OOD (classes mascaradas)",
            marker_color="#f78166", opacity=0.7, nbinsx=30,
        ))
        fig.add_vline(
            x=limiar_overconf,
            line_dash="dash",
            line_color="#e3b341",
            annotation_text=f"Limiar ({limiar_overconf})",
            annotation_position="top right")
        fig.update_layout(
            **_TEMA,
            barmode="overlay",
            height=350,
            xaxis_title="Confiança Máxima (Softmax)",
            yaxis_title="Nº de Amostras",
            margin=dict(
                t=10,
                b=40))
        st.plotly_chart(fig, use_container_width=True)

        # ── Distribuição de entropia ───────────────────────────────────────
        titulo_secao("Entropia de Shannon: In-Distribution vs OOD")
        st.caption(
            "Entropia baixa + classe desconhecida = sinal claro de Falsa Certeza.")

        fig_e = go.Figure()
        fig_e.add_trace(go.Histogram(
            x=df_ind["Entropia"], name="In-Distribution",
            marker_color="#58a6ff", opacity=0.7, nbinsx=30,
        ))
        fig_e.add_trace(go.Histogram(
            x=df_ood["Entropia"], name="OOD",
            marker_color="#f78166", opacity=0.7, nbinsx=30,
        ))
        fig_e.update_layout(
            **_TEMA,
            barmode="overlay",
            height=320,
            xaxis_title="Entropia de Shannon",
            yaxis_title="Nº de Amostras",
            margin=dict(
                t=10,
                b=40))
        st.plotly_chart(fig_e, use_container_width=True)

        # ── Para quais classes o modelo mapeou as OODs ────────────────────
        st.divider()
        titulo_secao("Mapeamento: para qual classe as OODs foram enviadas")
        contagem_prev = df_ood["Classe Prevista"].value_counts().reset_index()
        contagem_prev.columns = ["Classe Prevista", "Contagem"]
        fig_map = px.bar(
            contagem_prev,
            x="Classe Prevista",
            y="Contagem",
            color="Contagem",
            color_continuous_scale="Reds",
            labels={
                "Classe Prevista": "Dígito Previsto (classe conhecida)"},
            text="Contagem")
        fig_map.update_traces(textposition="outside")
        fig_map.update_layout(
            **_TEMA,
            coloraxis_showscale=False,
            height=320,
            margin=dict(
                t=10,
                b=40),
            xaxis=dict(
                tickmode="array",
                tickvals=list(
                    range(10)),
                ticktext=[
                    str(i) for i in range(10)]))
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Tabela detalhada ───────────────────────────────────────────────────
    st.divider()
    titulo_secao("Detalhes das Amostras OOD")
    apenas_alertas = st.toggle(
        "Exibir apenas amostras com alerta de overconfidence",
        value=False)
    df_exib = (
        df_ood[df_ood["Alerta OOD"] == "⚠️ Sim"]
        if apenas_alertas else df_ood
    )
    st.dataframe(
        df_exib.drop(
            columns=["Confiável"]),
        use_container_width=True,
        hide_index=True)
