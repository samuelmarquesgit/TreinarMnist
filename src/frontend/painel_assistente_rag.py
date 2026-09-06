"""Painel 7 — Chatbot Assistente RAG: perguntas em linguagem natural sobre o projeto."""

import streamlit as st

from src.frontend.estilos import aplicar_estilos, titulo_secao, badge


# ── Estado da sessão ───────────────────────────────────────────────────────

import json
import os


def _inicializar_estado() -> None:
    if "historico_chat" not in st.session_state:
        caminho_hist = 'reports/historico_chat.json'
        historico_inicial = []
        if os.path.exists(caminho_hist):
            try:
                with open(caminho_hist, 'r', encoding='utf-8') as f:
                    historico_inicial = json.load(f)
            except Exception:
                pass
        st.session_state.historico_chat = historico_inicial

    if "rag_pronto" not in st.session_state:
        st.session_state.rag_pronto = False
        st.session_state.assistente = None


def _salvar_historico() -> None:
    os.makedirs('reports', exist_ok=True)
    with open('reports/historico_chat.json', 'w', encoding='utf-8') as f:
        json.dump(st.session_state.historico_chat, f, indent=4, ensure_ascii=False)


# ── Tentativa de carregar o RAG (gracioso se não estiver implementado) ─────


def _carregar_assistente():
    """Tenta importar e instanciar o AssistenteRAG. Retorna None se indisponível."""
    try:
        from src.rag.assistente import AssistenteRAG  # type: ignore
        assistente = AssistenteRAG()
        assistente.indexar_documentos()
        return assistente
    except (ImportError, Exception):
        return None


# ── Bolhas de chat ─────────────────────────────────────────────────────────

_CSS_CHAT = """
<style>
.chat-bolha-usuario {
    background: rgba(88,166,255,.15);
    border: 1px solid rgba(88,166,255,.25);
    border-radius: 12px 12px 2px 12px;
    padding: .65rem 1rem;
    margin: .4rem 0 .4rem 20%;
    color: #e6edf3;
    font-size: .92rem;
}
.chat-bolha-assistente {
    background: rgba(22,27,34,.9);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px 12px 12px 2px;
    padding: .65rem 1rem;
    margin: .4rem 20% .4rem 0;
    color: #e6edf3;
    font-size: .92rem;
}
.chat-fonte {
    font-size: .72rem;
    color: #8b949e;
    margin-top: .3rem;
}
.chat-area {
    max-height: 460px;
    overflow-y: auto;
    padding: .5rem 0;
    margin-bottom: 1rem;
}
</style>
"""


def _renderizar_historico() -> None:
    blocos = []
    for msg in st.session_state.historico_chat:
        if msg["papel"] == "usuario":
            blocos.append(
                f'<div class="chat-bolha-usuario">🧑 {msg["conteudo"]}</div>')
        else:
            fontes_html = ""
            if msg.get("fontes"):
                fontes_lista = " · ".join(
                    f'<code>{f}</code>' for f in msg["fontes"])
                fontes_html = f'<div class="chat-fonte">📎 Fontes: {fontes_lista}</div>'
            blocos.append(
                f'<div class="chat-bolha-assistente">🤖 {msg["conteudo"]}{fontes_html}</div>'
            )
    st.markdown(
        f'<div class="chat-area">{"".join(blocos)}</div>',
        unsafe_allow_html=True,
    )


# ── Sugestões de perguntas ─────────────────────────────────────────────────

PERGUNTAS_SUGERIDAS = [
    "Qual modelo obteve a melhor acurácia?",
    "Como funciona o Vision Transformer neste projeto?",
    "Explique o experimento de robustez OOD com as classes 4 e 7.",
    "O que é falsa certeza (overconfidence) e como foi detectada?",
    "Quais são as métricas de avaliação usadas nos benchmarks?",
    "Descreva o pipeline de visão computacional para imagens reais.",
]


# ── Resposta de fallback (sem RAG ativo) ──────────────────────────────────

_BASE_CONHECIMENTO = {
    "acurácia": "A acurácia mede a proporção de predições corretas. No projeto, é calculada por sklearn.metrics.accuracy_score.",  # noqa: E501
    "precisão": "A precisão (precision) indica a fração de positivos verdadeiros entre todos os positivos previstos.",
    "recall": "O recall mede a fração de positivos verdadeiros corretamente identificados pelo modelo.",
    "f1": "O F1-Score é a média harmônica entre precisão e recall, útil para conjuntos desbalanceados.",
    "ood": "O experimento OOD mascarou as classes 4 e 7 no treino e as apresentou na inferência para avaliar robustez.",
    "overconfidence": "Falsa certeza ocorre quando o modelo atribui alta probabilidade a uma classe mesmo sem ter aprendido sobre ela.",  # noqa: E501
    "mnist": "O MNIST contém 70.000 imagens 28×28 de dígitos manuscritos (0–9), com 60.000 para treino e 10.000 para teste.",  # noqa: E501
    "vision transformer": "O ViT adapta a arquitetura Transformer para patches de imagens. Neste projeto é uma implementação educacional em NumPy puro.",  # noqa: E501
    "rag": "RAG (Retrieval-Augmented Generation) combina busca semântica em ChromaDB com geração de resposta contextualizada.",  # noqa: E501
    "pipeline": "O pipeline de visão converte a imagem para grayscale, detecta bounding box, redimensiona para 20×20 e centraliza em 28×28.",  # noqa: E501
}


def _resposta_fallback(pergunta: str) -> str:
    """Retorna resposta baseada em palavras-chave quando o RAG não está ativo."""
    pergunta_lower = pergunta.lower()
    for chave, resposta in _BASE_CONHECIMENTO.items():
        if chave in pergunta_lower:
            return f"💡 (Modo offline — RAG não inicializado)\n\n{resposta}"
    return (
        "💡 (Modo offline — RAG não inicializado)\n\n"
        "Não encontrei correspondência direta. Inicialize o RAG clicando em **⚡ Inicializar RAG** "
        "para obter respostas semânticas completas sobre o projeto."
    )


# ── Painel principal ───────────────────────────────────────────────────────

def _renderizar_status_rag() -> None:
    """Exibe badge de status do RAG e botão de inicialização."""
    col_status, col_btn = st.columns([4, 1])
    with col_status:
        if st.session_state.rag_pronto:
            st.markdown(badge("RAG Ativo", "ok") + "&nbsp; Base de conhecimento indexada.", unsafe_allow_html=True)
        else:
            st.markdown(
                badge("RAG Inativo", "aviso") + "&nbsp; Módulo RAG ainda não implementado ou ChromaDB não configurado.",
                unsafe_allow_html=True,
            )
    with col_btn:
        if st.button("⚡ Inicializar RAG"):
            with st.spinner("Indexando documentos no ChromaDB..."):
                assistente = _carregar_assistente()
                if assistente:
                    st.session_state.assistente = assistente
                    st.session_state.rag_pronto = True
                    st.success("RAG inicializado com sucesso!")
                else:
                    st.warning("Módulo RAG (src/rag/assistente.py) ainda não implementado. Respondendo com base no conhecimento geral do projeto.")  # noqa: E501


def _renderizar_sugestoes() -> None:
    """Renderiza botões de perguntas sugeridas."""
    titulo_secao("Sugestões de Perguntas")
    cols = st.columns(3)
    for i, pergunta in enumerate(PERGUNTAS_SUGERIDAS):
        if cols[i % 3].button(pergunta, key=f"sug_{i}", use_container_width=True):
            st.session_state._pergunta_pendente = pergunta


def _renderizar_formulario_chat() -> str | None:
    """Renderiza o formulário de entrada e retorna a pergunta final ou None."""
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("form_chat", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            pergunta_digitada = st.text_input(
                "Sua pergunta",
                placeholder="Ex.: Qual modelo teve menor overfitting?",
                label_visibility="collapsed",
            )
        with col_send:
            enviado = st.form_submit_button("Enviar ➤", use_container_width=True)
    if enviado and pergunta_digitada:
        return pergunta_digitada
    return st.session_state.pop("_pergunta_pendente", None)


def _processar_pergunta(pergunta: str) -> None:
    """Executa a consulta RAG ou fallback e atualiza o histórico."""
    st.session_state.historico_chat.append({"papel": "usuario", "conteudo": pergunta})
    with st.spinner("Buscando resposta..."):
        if st.session_state.rag_pronto and st.session_state.assistente:
            try:
                resultado = st.session_state.assistente.perguntar(pergunta)
                resposta = resultado.get("resposta", "Não foi possível gerar uma resposta.")
                fontes = resultado.get("fontes", [])
            except Exception as e:
                resposta = f"Erro ao consultar o RAG: {e}"
                fontes = []
        else:
            resposta = _resposta_fallback(pergunta)
            fontes = []
    st.session_state.historico_chat.append({"papel": "assistente", "conteudo": resposta, "fontes": fontes})
    _salvar_historico()
    st.rerun()


def renderizar() -> None:
    """Ponto de entrada do Painel 7 — chamado pelo app.py."""
    aplicar_estilos()
    st.markdown(_CSS_CHAT, unsafe_allow_html=True)
    _inicializar_estado()

    st.markdown("## 💬 Assistente RAG")
    st.caption("Faça perguntas em linguagem natural sobre os experimentos, métricas e análises do projeto.")

    _renderizar_status_rag()
    st.divider()
    titulo_secao("Conversa")
    _renderizar_historico()
    _renderizar_sugestoes()

    pergunta_final = _renderizar_formulario_chat()
    if pergunta_final:
        _processar_pergunta(pergunta_final)
