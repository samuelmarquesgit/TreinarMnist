"""Injeção de CSS customizado: Dark Mode profissional com efeito Glassmorphism."""

import streamlit as st


CSS_GLOBAL = """
<style>
/* ── Variáveis de tema ─────────────────────────────────────────── */
:root {
    --cor-fundo:        #0d1117;
    --cor-superfice:    #161b22;
    --cor-card:         rgba(22, 27, 34, 0.85);
    --cor-borda:        rgba(255, 255, 255, 0.08);
    --cor-primaria:     #58a6ff;
    --cor-secundaria:   #3fb950;
    --cor-alerta:       #f78166;
    --cor-aviso:        #e3b341;
    --cor-texto:        #e6edf3;
    --cor-texto-suave:  #8b949e;
    --raio-card:        12px;
    --sombra-card:      0 4px 24px rgba(0,0,0,0.4);
    --blur-glass:       12px;
}

/* ── Reset e fundo global ──────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--cor-fundo) !important;
    color: var(--cor-texto) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--cor-superfice) !important;
    border-right: 1px solid var(--cor-borda);
}

/* ── Cards Glassmorphism ───────────────────────────────────────── */
.glass-card {
    background: var(--cor-card);
    border: 1px solid var(--cor-borda);
    border-radius: var(--raio-card);
    backdrop-filter: blur(var(--blur-glass));
    -webkit-backdrop-filter: blur(var(--blur-glass));
    padding: 1.25rem 1.5rem;
    box-shadow: var(--sombra-card);
    margin-bottom: 1rem;
}

/* ── KPI tiles ─────────────────────────────────────────────────── */
.kpi-tile {
    background: var(--cor-card);
    border: 1px solid var(--cor-borda);
    border-radius: var(--raio-card);
    backdrop-filter: blur(var(--blur-glass));
    padding: 1rem 1.25rem;
    text-align: center;
    box-shadow: var(--sombra-card);
    transition: transform .15s ease, box-shadow .15s ease;
}
.kpi-tile:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
.kpi-valor { font-size: 2rem; font-weight: 700; color: var(--cor-primaria); line-height: 1.1; }
.kpi-label { font-size: .8rem; color: var(--cor-texto-suave); margin-top: .25rem; letter-spacing: .04em; text-transform: uppercase; }  # noqa: E501

/* ── Badges de status ──────────────────────────────────────────── */
.badge-ok    { background: rgba(63,185,80,.15); color: var(--cor-secundaria); border: 1px solid rgba(63,185,80,.3); padding: .2rem .6rem; border-radius: 20px; font-size: .75rem; font-weight: 600; }  # noqa: E501
.badge-erro  { background: rgba(247,129,102,.15); color: var(--cor-alerta);   border: 1px solid rgba(247,129,102,.3); padding: .2rem .6rem; border-radius: 20px; font-size: .75rem; font-weight: 600; }  # noqa: E501
.badge-aviso { background: rgba(227,179,65,.15); color: var(--cor-aviso);    border: 1px solid rgba(227,179,65,.3); padding: .2rem .6rem; border-radius: 20px; font-size: .75rem; font-weight: 600; }  # noqa: E501

/* ── Títulos de seção ──────────────────────────────────────────── */
.secao-titulo {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--cor-texto);
    border-left: 3px solid var(--cor-primaria);
    padding-left: .75rem;
    margin: 1.25rem 0 .75rem;
}

/* ── Tabelas ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] table {
    background: transparent !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(88,166,255,.1) !important;
    color: var(--cor-primaria) !important;
    font-weight: 600;
    font-size: .82rem;
    text-transform: uppercase;
    letter-spacing: .05em;
}
[data-testid="stDataFrame"] td {
    color: var(--cor-texto) !important;
    border-bottom: 1px solid var(--cor-borda) !important;
}

/* ── Botões ────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {
    background: var(--cor-primaria) !important;
    color: #0d1117 !important;
    border: none !important;
    font-weight: 600;
    border-radius: 8px;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid var(--cor-borda) !important;
    color: var(--cor-texto) !important;
    border-radius: 8px;
}

/* ── Inputs e Selects ──────────────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > div {
    background: var(--cor-superfice) !important;
    border: 1px solid var(--cor-borda) !important;
    border-radius: 8px;
    color: var(--cor-texto) !important;
}

/* ── Métricas nativas Streamlit ────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--cor-card);
    border: 1px solid var(--cor-borda);
    border-radius: var(--raio-card);
    padding: .75rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--cor-primaria) !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--cor-texto-suave) !important; font-size: .8rem; }

/* ── Scrollbar fina ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--cor-fundo); }
::-webkit-scrollbar-thumb { background: var(--cor-borda); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cor-texto-suave); }

/* ── Divider ───────────────────────────────────────────────────── */
hr { border-color: var(--cor-borda) !important; margin: 1.5rem 0; }
</style>
"""


def aplicar_estilos() -> None:
    """Injeta o CSS global de Dark Mode e Glassmorphism na página Streamlit."""
    st.markdown(CSS_GLOBAL, unsafe_allow_html=True)


def card(conteudo_html: str) -> None:
    """Renderiza um bloco de conteúdo HTML dentro de um card glassmorphism."""
    st.markdown(
        f'<div class="glass-card">{conteudo_html}</div>',
        unsafe_allow_html=True)


def kpi_tile(valor: str, label: str) -> str:
    """Retorna o HTML de um KPI tile para uso com st.markdown."""
    return (
        f'<div class="kpi-tile">'
        f'  <div class="kpi-valor">{valor}</div>'
        f'  <div class="kpi-label">{label}</div>'
        f'</div>'
    )


def badge(texto: str, tipo: str = "ok") -> str:
    """Retorna HTML de um badge colorido. tipo: 'ok' | 'erro' | 'aviso'."""
    classe = {"ok": "badge-ok", "erro": "badge-erro",
              "aviso": "badge-aviso"}.get(tipo, "badge-ok")
    return f'<span class="{classe}">{texto}</span>'


def titulo_secao(texto: str) -> None:
    """Renderiza um título de seção com barra lateral colorida."""
    st.markdown(
        f'<div class="secao-titulo">{texto}</div>',
        unsafe_allow_html=True)
