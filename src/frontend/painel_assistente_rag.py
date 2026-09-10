"""Painel 7 — Chatbot Assistente RAG: perguntas em linguagem natural sobre o projeto."""

import json
import os

import streamlit as st
import streamlit.components.v1 as components

from src.frontend.estilos import aplicar_estilos, badge

# ── Estado da sessão ───────────────────────────────────────────────────────


def _inicializar_estado() -> None:
    if "historico_chat" not in st.session_state:
        caminho_hist = "reports/historico_chat.json"
        historico_inicial: list[dict] = []
        if os.path.exists(caminho_hist):
            try:
                with open(caminho_hist, encoding="utf-8") as f:
                    historico_inicial = json.load(f)
            except Exception:
                pass
        st.session_state.historico_chat = historico_inicial

    if "rag_pronto" not in st.session_state:
        st.session_state.rag_pronto = False
        st.session_state.assistente = None


def _salvar_historico() -> None:
    os.makedirs("reports", exist_ok=True)
    with open("reports/historico_chat.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.historico_chat, f, indent=4, ensure_ascii=False)


# ── Carregar RAG ──────────────────────────────────────────────────────────


def _carregar_assistente():
    try:
        from src.rag.assistente import AssistenteRAG  # type: ignore

        assistente = AssistenteRAG()
        assistente.indexar_documentos()
        return assistente
    except Exception as e:
        import traceback

        st.error(f"Erro ao carregar AssistenteRAG: {e}")
        st.code(traceback.format_exc())
        return None


# ── Fallback sem RAG ──────────────────────────────────────────────────────

_BASE_CONHECIMENTO = {
    "acurácia": "A acurácia mede a proporção de predições corretas. No projeto, é calculada por sklearn.metrics.accuracy_score.",
    "precisão": "A precisão (precision) indica a fração de positivos verdadeiros entre todos os positivos previstos.",
    "recall": "O recall mede a fração de positivos verdadeiros corretamente identificados pelo modelo.",
    "f1": "O F1-Score é a média harmônica entre precisão e recall, útil para conjuntos desbalanceados.",
    "ood": "O experimento OOD mascarou as classes 4 e 7 no treino e as apresentou na inferência para avaliar robustez.",
    "overconfidence": "Falsa certeza ocorre quando o modelo atribui alta probabilidade a uma classe mesmo sem ter aprendido sobre ela.",
    "mnist": "O MNIST contém 70.000 imagens 28×28 de dígitos manuscritos (0–9), com 60.000 para treino e 10.000 para teste.",
    "vision transformer": "O ViT adapta a arquitetura Transformer para patches de imagens. Neste projeto é uma implementação educacional em NumPy puro.",
    "rag": "RAG (Retrieval-Augmented Generation) combina busca semântica em ChromaDB com geração de resposta contextualizada.",
    "pipeline": "O pipeline de visão converte a imagem para grayscale, detecta bounding box, redimensiona para 20×20 e centraliza em 28×28.",
    "eda": "Análise Exploratória de Dados: abordagem para explorar características fundamentais de um conjunto de dados antes da modelagem.",
    "espaço latente": "Representação comprimida de dados num espaço de menor dimensão, gerada por algoritmos como PCA ou t-SNE.",
    "anova": "Análise de Variância: teste estatístico que compara médias de múltiplos grupos para determinar se há diferença significativa.",
    "heatmap": "Mapa de calor 28×28 que mostra a intensidade média de pixel de uma classe, revelando padrões espaciais nos dígitos.",
    "boxplot": "Diagrama de caixa que resume a distribuição de dados por quartis, comparável entre múltiplas classes.",
    "q-q plot": "Gráfico quantil-quantil para verificar se os dados seguem uma distribuição normal.",
}


def _resposta_fallback(pergunta: str) -> str:
    pergunta_lower = pergunta.lower()
    for chave, resposta in _BASE_CONHECIMENTO.items():
        if chave in pergunta_lower:
            return f"💡 *Modo offline — RAG não inicializado*\n\n{resposta}"
    return (
        "💡 *Modo offline — RAG não inicializado*\n\n"
        "Não encontrei correspondência direta. "
        "Inicialize o **RAG** (botão ⚡) para obter respostas semânticas completas."
    )


# ── Agrupador de pares pergunta/resposta ──────────────────────────────────


def _agrupar_pares() -> list[dict]:
    historico = st.session_state.historico_chat
    pares: list[dict] = []
    i = 0
    while i < len(historico):
        msg = historico[i]
        if msg["papel"] == "usuario":
            bot = historico[i + 1] if i + 1 < len(historico) else None
            pares.append(
                {
                    "idx_user": i,
                    "pergunta": msg["conteudo"],
                    "resposta": bot["conteudo"] if bot else "…",
                    "fontes": bot.get("fontes", []) if bot else [],
                    "idx_bot": i + 1 if bot else None,
                }
            )
            i += 2 if bot else 1
        else:
            i += 1
    return pares


def _excluir_par(idx_user: int, idx_bot: int | None) -> None:
    indices = sorted(
        [x for x in [idx_user, idx_bot] if x is not None],
        reverse=True,
    )
    for idx in indices:
        if idx < len(st.session_state.historico_chat):
            st.session_state.historico_chat.pop(idx)
    _salvar_historico()
    st.rerun()


# ── CSS ────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.chat-container {
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
    background: rgba(13,17,23,.6);
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.chat-msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 4px;
}
.chat-msg-user .bubble {
    background: linear-gradient(135deg, rgba(88,166,255,.25), rgba(56,139,253,.15));
    border: 1px solid rgba(88,166,255,.3);
    border-radius: 16px 16px 4px 16px;
    padding: .55rem 1rem;
    max-width: 75%;
    color: #e6edf3;
    font-size: .92rem;
    line-height: 1.55;
    word-break: break-word;
}
.chat-msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 2px;
}
.chat-msg-bot .bubble {
    background: rgba(22,27,34,.95);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 4px 16px 16px 16px;
    padding: .65rem 1.1rem;
    max-width: 88%;
    color: #c9d1d9;
    font-size: .92rem;
    line-height: 1.6;
    word-break: break-word;
}
.chat-label {
    font-size: .7rem;
    color: #6e7681;
    margin-bottom: 3px;
    padding: 0 4px;
}
.chat-label-user { text-align: right; }
.chat-fonte {
    font-size: .72rem;
    color: #6e7681;
    margin-top: .4rem;
    border-top: 1px solid rgba(255,255,255,.07);
    padding-top: .3rem;
}
.chat-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,.05);
    margin: .6rem 0;
}
.chat-empty {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #484f58;
}
.chat-empty .icon { font-size: 2.2rem; }
</style>
"""


def _scroll_js() -> None:
    """Injeta JS que rola a página principal do Streamlit para o final do chat.

    O ``components.html`` executa dentro de um <iframe>. Por isso usamos
    ``window.parent`` para alcançar o documento pai e encontrar o container
    de scroll do Streamlit (``[data-testid='stAppViewContainer']`` ou
    ``[data-testid='block-container']``).
    """
    scroll_html = """
    <script>
    (function scroll() {
        try {
            var selectors = [
                '[data-testid="stAppViewContainer"]',
                '[data-testid="block-container"]',
                '.main',
                'section.main'
            ];
            var doc = window.parent.document;
            for (var i = 0; i < selectors.length; i++) {
                var el = doc.querySelector(selectors[i]);
                if (el && el.scrollHeight > el.clientHeight) {
                    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                    return;
                }
            }
            // Fallback: rola o body do pai
            window.parent.scrollTo({ top: window.parent.document.body.scrollHeight, behavior: 'smooth' });
        } catch(e) {
            // Cross-origin: sem acesso — ignora silenciosamente
        }
    })();
    </script>
    """
    components.html(scroll_html, height=0, scrolling=False)


# ── Sugestões ─────────────────────────────────────────────────────────────

PERGUNTAS_SUGERIDAS = [
    "O que é EDA?",
    "O que é Vision Transformer?",
    "Explique o experimento OOD.",
    "O que é overconfidence?",
    "O que é Q-Q Plot?",
    "Como funciona o pipeline de visão?",
    "O que é ANOVA?",
    "O que é Espaço Latente?",
]


# ── Seções do painel ──────────────────────────────────────────────────────


def _renderizar_status_rag() -> None:
    col_status, col_btn, col_limpar = st.columns([4, 1.3, 1.3])
    with col_status:
        if st.session_state.rag_pronto:
            st.markdown(badge("RAG Ativo", "ok") + "&nbsp; Base indexada.", unsafe_allow_html=True)
        else:
            st.markdown(
                badge("RAG Inativo", "aviso") + "&nbsp; Modo offline (sem ChromaDB).",
                unsafe_allow_html=True,
            )
    with col_btn:
        if st.button("⚡ Inicializar RAG", use_container_width=True):
            with st.spinner("Indexando documentos…"):
                assistente = _carregar_assistente()
                if assistente:
                    st.session_state.assistente = assistente
                    st.session_state.rag_pronto = True
                    st.success("RAG inicializado!")
                else:
                    st.warning("RAG não disponível — usando modo offline.")
    with col_limpar:
        if st.button("🗑️ Limpar tudo", use_container_width=True):
            st.session_state.historico_chat = []
            _salvar_historico()
            st.rerun()


def _renderizar_chat() -> None:
    """Renderiza o histórico usando st.chat_message nativo + botão de exclusão."""
    pares = _agrupar_pares()

    if not pares:
        st.markdown(
            '<div class="chat-empty"><div class="icon">💬</div>'
            "Nenhuma conversa ainda. Use as sugestões abaixo!</div>",
            unsafe_allow_html=True,
        )
        return

    for par_idx, par in enumerate(pares):
        # ── Pergunta do usuário ──
        with st.chat_message("user"):
            st.write(par["pergunta"])

        # ── Resposta do assistente + botão excluir ──
        with st.chat_message("assistant"):
            st.write(par["resposta"])
            if par["fontes"]:
                st.caption("📎 Fontes: " + " · ".join(par["fontes"]))
            # Botão de exclusão DENTRO da bolha do assistente
            if st.button(
                "🗑️ Excluir esta resposta",
                key=f"del_{par_idx}",
                help="Remove esta pergunta e resposta do histórico",
            ):
                _excluir_par(par["idx_user"], par["idx_bot"])

    # Scroll automático: executa após renderizar tudo
    _scroll_js()


def _renderizar_sugestoes() -> None:
    st.markdown("**💡 Sugestões rápidas**")
    cols = st.columns(4)
    for i, pergunta in enumerate(PERGUNTAS_SUGERIDAS):
        if cols[i % 4].button(pergunta, key=f"sug_{i}", use_container_width=True):
            st.session_state._pergunta_pendente = pergunta


def _renderizar_input() -> str | None:
    with st.form("form_chat", clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            texto = st.text_input(
                "pergunta",
                placeholder="Ex.: Qual modelo teve menor overfitting? O que é t-SNE?",
                label_visibility="collapsed",
            )
        with col_btn:
            enviado = st.form_submit_button("➤ Enviar", use_container_width=True, type="primary")

    if enviado and texto.strip():
        return texto.strip()

    pendente = st.session_state.pop("_pergunta_pendente", None)
    return str(pendente) if pendente is not None else None


def _processar_pergunta(pergunta: str) -> None:
    st.session_state.historico_chat.append({"papel": "usuario", "conteudo": pergunta})
    with st.spinner("Buscando resposta…"):
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

    st.session_state.historico_chat.append(
        {"papel": "assistente", "conteudo": resposta, "fontes": fontes}
    )
    _salvar_historico()
    st.rerun()


# ── Ponto de entrada ──────────────────────────────────────────────────────


def renderizar() -> None:
    """Ponto de entrada do Painel 7 — chamado pelo app.py."""
    aplicar_estilos()
    st.markdown(_CSS, unsafe_allow_html=True)
    _inicializar_estado()

    st.markdown("## 💬 Assistente RAG")
    st.caption(
        "Faça perguntas em linguagem natural sobre experimentos, métricas e conceitos do projeto."
    )

    _renderizar_status_rag()
    st.divider()

    _renderizar_chat()

    st.divider()
    _renderizar_sugestoes()
    st.markdown("<br>", unsafe_allow_html=True)

    pergunta_final = _renderizar_input()
    if pergunta_final:
        _processar_pergunta(pergunta_final)
