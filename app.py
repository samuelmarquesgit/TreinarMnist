"""
Ponto de entrada do Frontend Web — Plataforma Empresarial MNIST.

Execução:
    streamlit run app.py
    python main.py --modo web
"""

from src.frontend.estilos import aplicar_estilos
import streamlit as st

# ── Configuração da página (deve ser a primeira chamada Streamlit) ─────────
st.set_page_config(
    page_title="Plataforma MNIST",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Importações dos painéis ───────────────────────────────────────────────

try:
    from src.fachada import FachadaPipelineIA
except ImportError as e:
    st.error(f"Erro ao importar a fachada do projeto: {e}")
    st.info("Execute o app a partir da raiz do repositório: `streamlit run app.py`")
    st.stop()

# ── Importações dos painéis (com fallback gracioso) ───────────────────────


def _importar_painel(modulo: str):
    try:
        import importlib
        return importlib.import_module(modulo)
    except ImportError:
        return None


painel_eda = _importar_painel("src.frontend.painel_eda")
painel_estatistica = _importar_painel("src.frontend.painel_analise_estatistica")
painel_benchmarks = _importar_painel("src.frontend.painel_benchmarks")
painel_ood = _importar_painel("src.frontend.painel_robustez_ood")
painel_visao = _importar_painel("src.frontend.painel_laboratorio_visao")
painel_bancos = _importar_painel("src.frontend.painel_bancos_dados")
painel_rag = _importar_painel("src.frontend.painel_assistente_rag")

# ── Cache da fachada ──────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def carregar_fachada() -> FachadaPipelineIA:
    f = FachadaPipelineIA()
    f.inicializar_dados()
    return f


# ── Estilos globais ───────────────────────────────────────────────────────
aplicar_estilos()

# ── Sidebar — Navegação ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: .5rem 0 1rem;'>"
        "<span style='font-size:2.2rem;'>🧠</span><br>"
        "<span style='font-size:1.1rem; font-weight:700; color:#e6edf3;'>Plataforma MNIST</span><br>"
        "<span style='font-size:.75rem; color:#8b949e;'>Enterprise AI · pt-BR</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    MENU = {
        "📊 Análise Exploratória (EDA)": "eda",
        "📈 Análise Estatística": "estatistica",
        "🏆 Benchmarks & Modelos": "benchmarks",
        "🧪 Robustez OOD": "ood",
        "✍️ Laboratório de Visão": "visao",
        "🗄️ Monitor de Bancos de Dados": "bancos",
        "💬 Assistente RAG": "rag",
    }

    pagina_label = st.radio(
        "Navegação",
        list(MENU.keys()),
        label_visibility="collapsed",
    )
    pagina = MENU[pagina_label]

    st.divider()
    st.caption("Versão 1.0.0 · develop")

# ── Carregamento da fachada com spinner ───────────────────────────────────
# Painéis que precisam dos dados
_REQUER_DADOS = {"eda", "estatistica", "benchmarks", "ood", "visao"}

if pagina in _REQUER_DADOS:
    if "fachada" not in st.session_state:
        with st.spinner("⏳ Carregando dataset MNIST... Isso pode demorar na primeira execução."):
            try:
                st.session_state.fachada = carregar_fachada()
                st.success("✅ MNIST carregado com sucesso!")
            except Exception as e:
                st.error(f"Falha ao carregar os dados: {e}")
                st.stop()
    fachada = st.session_state.fachada
else:
    fachada = None

# ── Roteamento de páginas ─────────────────────────────────────────────────


def _painel_indisponivel(nome: str) -> None:
    st.warning(f"⚠️ Painel **{nome}** não pôde ser importado. Verifique as dependências.")


def _rotear_pagina(pagina: str, fachada) -> None:
    """Despacha para o painel correto. Complexidade mantida abaixo de C901-10."""
    _COM_FACHADA = {
        "eda": (painel_eda, "Análise Exploratória"),
        "estatistica": (painel_estatistica, "Análise Estatística"),
        "benchmarks": (painel_benchmarks, "Benchmarks"),
        "ood": (painel_ood, "Robustez OOD"),
        "visao": (painel_visao, "Laboratório de Visão"),
    }
    _SEM_FACHADA = {
        "bancos": (painel_bancos, "Monitor de Bancos"),
        "rag": (painel_rag, "Assistente RAG"),
    }
    if pagina in _COM_FACHADA:
        modulo, nome = _COM_FACHADA[pagina]
        if modulo:
            modulo.renderizar(fachada)
        else:
            _painel_indisponivel(nome)
    elif pagina in _SEM_FACHADA:
        modulo, nome = _SEM_FACHADA[pagina]
        if modulo:
            modulo.renderizar()
        else:
            _painel_indisponivel(nome)


_rotear_pagina(pagina, fachada)
