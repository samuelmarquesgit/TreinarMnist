"""Painel 6 — Monitor de Bancos de Dados: PostgreSQL (SQLAlchemy) + MongoDB (fallback JSON)."""

import os
import json
import glob
from datetime import datetime

import pandas as pd
import streamlit as st

from src.frontend.estilos import aplicar_estilos, titulo_secao, badge, kpi_tile
from src.banco_dados.conexao_postgres import ConexaoPostgres, Experimento
from src.banco_dados.conexao_mongodb import ConexaoMongoDB


# ── helpers ────────────────────────────────────────────────────────────────


def _obter_experimentos_postgres() -> pd.DataFrame:
    """Lê todos os experimentos da tabela PostgreSQL/SQLite e devolve DataFrame."""
    try:
        db = ConexaoPostgres()
        with db.obter_sessao() as sessao:
            registros = sessao.query(Experimento).order_by(
                Experimento.data_execucao.desc()).all()
        if not registros:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ID": r.id,
            "Modelo": r.modelo,
            "Acurácia": f"{r.acuracia:.4f}" if r.acuracia is not None else "—",
            "Tempo Treino (s)": f"{r.tempo_treino:.2f}" if r.tempo_treino is not None else "—",
            "Data de Execução": r.data_execucao.strftime("%d/%m/%Y %H:%M:%S") if r.data_execucao else "—",
        } for r in registros])
    except Exception as erro:
        st.error(f"Erro ao consultar PostgreSQL: {erro}")
        return pd.DataFrame()


def _obter_artefatos_mongodb() -> list[dict]:
    """
    Tenta ler artefatos do MongoDB. Em modo fallback local, lista os JSON
    em reports/ e devolve até 20 documentos mais recentes.
    """
    mongo = ConexaoMongoDB()
    artefatos = []

    if mongo.usar_local:
        padrao = os.path.join("reports", "*.json")
        arquivos = sorted(
            glob.glob(padrao),
            key=os.path.getmtime,
            reverse=True)[
            :20]
        for caminho in arquivos:
            try:
                with open(caminho, encoding="utf-8") as f:
                    dados = json.load(f)
                nome = os.path.splitext(os.path.basename(caminho))[0]
                mtime = datetime.fromtimestamp(
                    os.path.getmtime(caminho)).strftime("%d/%m/%Y %H:%M:%S")
                artefatos.append(
                    {"nome": nome, "dados": dados, "salvo_em": mtime})
            except Exception:
                pass
    else:
        try:
            for doc in mongo.colecao.find(
                {}, {
                    "_id": 0}).sort(
                    "_id", -1).limit(20):
                artefatos.append(doc)
        except Exception as erro:
            st.error(f"Erro ao consultar MongoDB: {erro}")

    return artefatos


# ── painel principal ───────────────────────────────────────────────────────


def renderizar() -> None:
    """Ponto de entrada do Painel 6 — chamado pelo app.py."""
    aplicar_estilos()

    st.markdown("## 🗄️ Monitor de Bancos de Dados")
    st.caption(
        "Visualização em tempo real das tabelas PostgreSQL e documentos MongoDB/JSON.")

    aba_pg, aba_mongo = st.tabs(
        ["🐘 PostgreSQL / SQLite", "🍃 MongoDB / JSON Local"])

    # ── Aba PostgreSQL ────────────────────────────────────────────────────
    with aba_pg:
        titulo_secao("Tabela de Experimentos")

        col_att, col_esp = st.columns([1, 5])
        with col_att:
            st.button("🔄 Atualizar", key="att_pg")

        df = _obter_experimentos_postgres()

        if df.empty:
            st.info(
                "Nenhum experimento registrado ainda. Execute o pipeline para popular o banco.")
        else:
            # KPIs
            total = len(df)
            melhor_acc = df["Acurácia"].replace(
                "—", None).dropna().astype(float).max()
            k1, k2, k3 = st.columns(3)
            k1.markdown(
                kpi_tile(
                    str(total),
                    "Experimentos"),
                unsafe_allow_html=True)
            k2.markdown(kpi_tile(f"{melhor_acc:.4f}" if not pd.isna(
                melhor_acc) else "—", "Melhor Acurácia"), unsafe_allow_html=True)
            k3.markdown(
                kpi_tile(
                    df["Modelo"].iloc[0],
                    "Último Modelo"),
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            titulo_secao("Histórico Completo")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Gráfico de acurácia por modelo
            try:
                import plotly.express as px
                df_plot = df.copy()
                df_plot["Acurácia_num"] = pd.to_numeric(
                    df_plot["Acurácia"], errors="coerce")
                df_plot = df_plot.dropna(subset=["Acurácia_num"])
                if not df_plot.empty:
                    titulo_secao("Evolução de Acurácia por Execução")
                    fig = px.bar(
                        df_plot,
                        x="Modelo",
                        y="Acurácia_num",
                        color="Acurácia_num",
                        color_continuous_scale="Blues",
                        labels={"Acurácia_num": "Acurácia"},
                        template="plotly_dark",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        margin=dict(t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.caption("Instale plotly para ver o gráfico de barras.")

        # Status da conexão
        _url = os.getenv("DATABASE_URL", "sqlite:///reports/banco_local.db")
        modo = "PostgreSQL" if "postgresql" in _url else "SQLite local"
        st.markdown(
            f'<div style="margin-top:1rem; font-size:.8rem; color:#8b949e;">'
            f'Conexão ativa: {badge(modo, "ok")} &nbsp;·&nbsp; '
            f'<code>{_url.split("@")[-1] if "@" in _url else _url}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Aba MongoDB ───────────────────────────────────────────────────────
    with aba_mongo:
        titulo_secao("Artefatos Documentais")

        col_att2, _ = st.columns([1, 5])
        with col_att2:
            st.button("🔄 Atualizar", key="att_mongo")

        artefatos = _obter_artefatos_mongodb()
        mongo_modo = "MongoDB Atlas" if not ConexaoMongoDB(
        ).usar_local else "JSON Local (fallback)"
        modo_badge = "ok" if "Atlas" in mongo_modo else "aviso"

        st.markdown(
            f'<div style="margin-bottom:1rem; font-size:.8rem; color:#8b949e;">'
            f'Modo: {badge(mongo_modo, modo_badge)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not artefatos:
            st.info(
                "Nenhum artefato encontrado. Execute avaliações para gerar matrizes de confusão e relatórios.")
        else:
            k1, k2 = st.columns(2)
            k1.markdown(kpi_tile(str(len(artefatos)), "Artefatos"),
                        unsafe_allow_html=True)

            titulo_secao("Lista de Artefatos")
            for art in artefatos:
                nome = art.get("nome", "sem-nome")
                salvo_em = art.get("salvo_em", "—")
                with st.expander(f"📄 {nome}  ·  {salvo_em}"):
                    dados = art.get("dados", art)
                    # Se for matriz de confusão, tenta renderizar como
                    # DataFrame
                    if "matriz" in dados or "matriz_confusao" in dados:
                        chave = "matriz" if "matriz" in dados else "matriz_confusao"
                        try:
                            df_mat = pd.DataFrame(dados[chave])
                            st.dataframe(df_mat, use_container_width=True)
                        except Exception:
                            st.json(dados)
                    else:
                        st.json(dados)
