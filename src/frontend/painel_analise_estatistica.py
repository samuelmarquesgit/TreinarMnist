"""Painel 2 — Análise Estatística Interativa: Dados Brutos vs Tratados + Testes de Hipótese."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

from src.analise_estatistica import CalculadorEstatistico
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


# ── Helpers ────────────────────────────────────────────────────────────────

def _obter_dados(fachada, modo: str, particao: str,
                 digito: int | None) -> np.ndarray:
    """Devolve vetor 1D de pixels conforme modo (Bruto/Tratado), partição e filtro de classe."""
    X = fachada.X_treino if particao == "Treino" else fachada.X_teste
    y = fachada.y_treino if particao == "Treino" else fachada.y_teste

    if digito is not None:
        X = X[y == digito]

    if modo == "Brutos [0–255]":
        return (X * 255).flatten().astype(float)
    else:
        return X.flatten().astype(float)


def _card_metricas(stats_dict: dict) -> None:
    cols = st.columns(4)
    mapa = [
        ("média", "Média (μ)"),
        ("mediana", "Mediana"),
        ("desvio_padrao", "Desvio Padrão (σ)"),
        ("variancia", "Variância (σ²)"),
        ("minimo", "Mínimo"),
        ("maximo", "Máximo"),
        ("assimetria", "Assimetria (Skew)"),
        ("curtose", "Curtose (Kurt)"),
    ]
    for i, (chave, label) in enumerate(mapa):
        val = stats_dict.get(chave, 0)
        cols[i % 4].markdown(
            kpi_tile(f"{val:.4f}", label), unsafe_allow_html=True)
        if (i + 1) % 4 == 0 and i < len(mapa) - 1:
            cols = st.columns(4)


def _diagnostico_assimetria(assimetria: float, curtose: float) -> str:
    diag_ass = ("simétrica" if abs(assimetria) < 0.5
                else ("levemente assimétrica à direita" if assimetria > 0
                      else "levemente assimétrica à esquerda"))
    if abs(assimetria) > 1:
        diag_ass = "fortemente assimétrica à " + \
            ("direita" if assimetria > 0 else "esquerda")
    diag_kurt = ("mesocúrtica (normal)" if abs(curtose) < 0.5
                 else ("leptocúrtica (caudas pesadas)" if curtose > 0
                       else "platicúrtica (caudas leves)"))
    return f"Distribuição **{diag_ass}** · **{diag_kurt}**."


# ── Painel principal ───────────────────────────────────────────────────────

def renderizar(fachada) -> None:
    """Ponto de entrada do Painel 2 — recebe FachadaPipelineIA já inicializada."""
    aplicar_estilos()
    st.markdown("## 📈 Análise Estatística Interativa")
    st.caption(
        "Explore as distribuições de pixels com alternância instantânea entre dados brutos e tratados.")

    # ── Filtros globais ────────────────────────────────────────────────────
    titulo_secao("Filtros")
    col1, col2, col3 = st.columns(3)
    with col1:
        modo = st.radio(
            "Tipo de dado", [
                "Brutos [0–255]", "Tratados [0–1] MinMax"], horizontal=True)
    with col2:
        particao = st.radio("Partição", ["Treino", "Teste"], horizontal=True)
    with col3:
        usar_filtro_classe = st.toggle("Filtrar por dígito", value=False)
        digito_sel = st.selectbox("Dígito", list(
            range(10)), disabled=not usar_filtro_classe)
        digito_filtro = digito_sel if usar_filtro_classe else None

    dados = _obter_dados(fachada, modo, particao, digito_filtro)
    calc = CalculadorEstatistico()
    try:
        stats_dict = calc.estatisticas_descritivas(dados)
    except ValueError as e:
        st.error(f"Erro ao calcular estatísticas: {e}")
        return

    _classe = 'Dígito ' + str(digito_filtro) if digito_filtro is not None else 'Todas as classes'
    label_ctx = f"{_classe} · {particao} · {modo}"
    st.caption(
        f"Contexto: **{label_ctx}** · {len(dados):,} valores analisados")

    # ── Medidas descritivas ────────────────────────────────────────────────
    st.divider()
    titulo_secao("Medidas Descritivas")
    _card_metricas(stats_dict)

    ass, kurt = stats_dict["assimetria"], stats_dict["curtose"]
    st.markdown(f"> 🔍 {_diagnostico_assimetria(ass, kurt)}")

    # Quartis e IQR adicionais
    q1, q2, q3 = np.percentile(dados, [25, 50, 75])
    iqr = q3 - q1
    cv = (
        stats_dict["desvio_padrao"]
        / stats_dict["media"]
        * 100) if stats_dict["media"] != 0 else 0
    cq1, cq2, cq3, cqr, ccv = st.columns(5)
    cq1.metric("Q1 (25%)", f"{q1:.4f}")
    cq2.metric("Q2 (50%)", f"{q2:.4f}")
    cq3.metric("Q3 (75%)", f"{q3:.4f}")
    cqr.metric("IQR", f"{iqr:.4f}")
    ccv.metric("CV (%)", f"{cv:.2f}%")

    # ── Visualizações ──────────────────────────────────────────────────────
    if PLOTLY_OK:
        st.divider()
        titulo_secao("Visualizações Gráficas")

        abas = st.tabs(["Histograma + KDE", "Boxplot por Classe",
                       "Q-Q Plot", "Heatmap Espacial 28×28"])

        # ── Histograma + KDE ───────────────────────────────────────────────
        with abas[0]:
            amostra = dados if len(dados) <= 50_000 else np.random.choice(
                dados, 50_000, replace=False)
            nbins = int((max(amostra) - min(amostra)) / (25 if "Brutos" in modo else 0.02)) if len(amostra) > 0 else 50
            nbins = max(nbins, 10)
            df_amostra = pd.DataFrame({"Intensidade": amostra})
            fig_h = px.histogram(
                df_amostra, x="Intensidade", nbins=nbins,
                histnorm='density', color_discrete_sequence=["#58a6ff"]
            )
            fig_h.update_layout(
                **_TEMA,
                height=380,
                margin=dict(
                    t=10,
                    b=40),
                xaxis_title="Intensidade",
                yaxis_title="Densidade")
            st.plotly_chart(fig_h, use_container_width=True)

        # ── Boxplot por classe ─────────────────────────────────────────────
        with abas[1]:
            X_base = fachada.X_treino if particao == "Treino" else fachada.X_teste
            y_base = fachada.y_treino if particao == "Treino" else fachada.y_teste
            fator = 255 if "Brutos" in modo else 1

            amostras_box = []
            for cls in range(10):
                mask = y_base == cls
                medias_cls = X_base[mask].mean(axis=1) * fator
                amostra_cls = medias_cls if len(medias_cls) <= 2000 else \
                    np.random.choice(medias_cls, 2000, replace=False)
                for v in amostra_cls:
                    amostras_box.append(
                        {"Dígito": str(cls), "Intensidade Média": v})

            df_box = pd.DataFrame(amostras_box)
            fig_box = px.box(
                df_box,
                x="Dígito",
                y="Intensidade Média",
                color="Dígito",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_box.update_layout(**_TEMA, height=380, margin=dict(t=10, b=40),
                                  showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Q-Q Plot ───────────────────────────────────────────────────────
        with abas[2]:
            amostra_qq = dados if len(dados) <= 5000 else np.random.choice(
                dados, 5000, replace=False)
            qq_teor, qq_obs = scipy_stats.probplot(amostra_qq, dist="norm")[:2]
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=qq_teor[0], y=qq_teor[1],
                                        mode="markers", name="Observado",
                                        marker=dict(color="#58a6ff", size=3)))
            x_line = np.linspace(min(qq_teor[0]), max(qq_teor[0]), 100)
            slope, intercept = np.polyfit(qq_teor[0], qq_teor[1], 1)
            fig_qq.add_trace(
                go.Scatter(
                    x=x_line,
                    y=slope * x_line + intercept,
                    mode="lines",
                    name="Normal teórica",
                    line=dict(
                        color="#f78166",
                        dash="dash")))
            fig_qq.update_layout(**_TEMA, height=380, margin=dict(t=10, b=40),
                                 xaxis_title="Quantis Teóricos (Normal)",
                                 yaxis_title="Quantis Observados")
            st.plotly_chart(fig_qq, use_container_width=True)

        # ── Heatmap espacial 28×28 ─────────────────────────────────────────
        with abas[3]:
            X_base = fachada.X_treino if particao == "Treino" else fachada.X_teste
            y_base = fachada.y_treino if particao == "Treino" else fachada.y_teste
            fator = 255 if "Brutos" in modo else 1

            if digito_filtro is not None:
                mask = y_base == digito_filtro
                media_espacial = X_base[mask].mean(
                    axis=0).reshape(28, 28) * fator
                titulo_heat = f"Intensidade média por pixel — Dígito {digito_filtro}"
            else:
                media_espacial = X_base.mean(axis=0).reshape(28, 28) * fator
                titulo_heat = "Intensidade média por pixel — Todas as classes"

            fig_heat = px.imshow(
                media_espacial, color_continuous_scale="Blues", zmin=0, zmax=(
                    255 if "Brutos" in modo else 1), labels=dict(
                    color="Intensidade"))
            fig_heat.update_layout(
                **_TEMA,
                height=380,
                margin=dict(
                    t=30,
                    b=10),
                title=titulo_heat)
            fig_heat.update_xaxes(showticklabels=False)
            fig_heat.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Testes de Hipótese ─────────────────────────────────────────────────
    st.divider()
    titulo_secao("Inferência Estatística & Testes de Hipótese")

    abas_testes = st.tabs(["Normalidade",
                           "Teste t (par de dígitos)",
                           "ANOVA (10 classes)",
                           "Qui-Quadrado"])

    X_base = fachada.X_treino if particao == "Treino" else fachada.X_teste
    y_base = fachada.y_treino if particao == "Treino" else fachada.y_teste
    fator = 255 if "Brutos" in modo else 1

    # ── Normalidade ────────────────────────────────────────────────────────
    with abas_testes[0]:
        st.markdown(
            "**Shapiro-Wilk e Kolmogorov-Smirnov** sobre intensidade média por imagem.")
        amostra_norm = (X_base.mean(axis=1) * fator)
        amostra_norm = amostra_norm if len(amostra_norm) <= 5000 else \
            np.random.choice(amostra_norm, 5000, replace=False)

        stat_sw, p_sw = scipy_stats.shapiro(amostra_norm[:3000])
        stat_ks, p_ks = scipy_stats.kstest(
            amostra_norm, "norm", args=(
                amostra_norm.mean(), amostra_norm.std()))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Shapiro-Wilk**")
            st.metric("Estatística W", f"{stat_sw:.5f}")
            st.metric("p-valor", f"{p_sw:.5e}")
            conclusao_sw = "✅ Normal (p > 0.05)" if p_sw > 0.05 else "❌ Não-normal (p ≤ 0.05)"
            st.markdown(f"Conclusão: {conclusao_sw}")
        with c2:
            st.markdown("**Kolmogorov-Smirnov**")
            st.metric("Estatística KS", f"{stat_ks:.5f}")
            st.metric("p-valor", f"{p_ks:.5e}")
            conclusao_ks = "✅ Normal (p > 0.05)" if p_ks > 0.05 else "❌ Não-normal (p ≤ 0.05)"
            st.markdown(f"Conclusão: {conclusao_ks}")

    # ── Teste t ────────────────────────────────────────────────────────────
    with abas_testes[1]:
        st.markdown(
            "**Teste t de Student** comparando brilho médio entre dois dígitos.")
        ct1, ct2 = st.columns(2)
        digito_a = ct1.selectbox(
            "Dígito A", list(
                range(10)), index=0, key="ta")
        digito_b = ct2.selectbox(
            "Dígito B", list(
                range(10)), index=1, key="tb")

        if digito_a == digito_b:
            st.warning("Selecione dígitos diferentes.")
        else:
            ga = X_base[y_base == digito_a].mean(axis=1) * fator
            gb = X_base[y_base == digito_b].mean(axis=1) * fator
            stat_t, p_t = scipy_stats.ttest_ind(ga, gb)
            st.metric(f"Média dígito {digito_a}", f"{ga.mean():.4f}")
            st.metric(f"Média dígito {digito_b}", f"{gb.mean():.4f}")
            st.metric("Estatística t", f"{stat_t:.4f}")
            st.metric("p-valor (bilateral)", f"{p_t:.5e}")
            concl = ("✅ Médias **não diferem** significativamente (p > 0.05)"
                     if p_t > 0.05 else
                     f"❌ Médias diferem significativamente — dígito {digito_a} ≠ dígito {digito_b} (p ≤ 0.05)")
            st.markdown(concl)

    # ── ANOVA ──────────────────────────────────────────────────────────────
    with abas_testes[2]:
        st.markdown(
            "**ANOVA de 1 Fator** — brilho médio entre as 10 classes de dígitos.")
        grupos = [X_base[y_base == cls].mean(
            axis=1) * fator for cls in range(10)]
        stat_f, p_f = scipy_stats.f_oneway(*grupos)
        st.metric("Estatística F", f"{stat_f:.4f}")
        st.metric("p-valor", f"{p_f:.5e}")
        concl_anova = ("✅ Não há diferença significativa entre as classes (p > 0.05)"
                       if p_f > 0.05 else
                       "❌ Há diferença significativa de brilho entre as 10 classes de dígitos (p ≤ 0.05)")
        st.markdown(concl_anova)

        if PLOTLY_OK:
            medias = [g.mean() for g in grupos]
            fig_an = px.bar(x=list(range(10)), y=medias,
                            labels={"x": "Dígito", "y": "Brilho Médio"},
                            color=medias, color_continuous_scale="Blues",
                            template="plotly_dark")
            fig_an.update_layout(
                **_TEMA,
                height=300,
                coloraxis_showscale=False,
                margin=dict(
                    t=10,
                    b=40))
            st.plotly_chart(fig_an, use_container_width=True)

    # ── Qui-Quadrado ───────────────────────────────────────────────────────
    with abas_testes[3]:
        st.markdown(
            "**Qui-Quadrado de aderência** — distribuição observada de classes vs. distribuição uniforme esperada.")
        _, contagens = np.unique(y_base, return_counts=True)
        esperado = np.full(len(contagens), contagens.mean())
        stat_chi, p_chi = scipy_stats.chisquare(contagens, f_exp=esperado)
        st.metric("Estatística χ²", f"{stat_chi:.4f}")
        st.metric("p-valor", f"{p_chi:.5e}")
        concl_chi = ("✅ Distribuição **uniforme** (balanceada) — p > 0.05"
                     if p_chi > 0.05 else
                     "❌ Distribuição **não uniforme** entre classes — p ≤ 0.05")
        st.markdown(concl_chi)
