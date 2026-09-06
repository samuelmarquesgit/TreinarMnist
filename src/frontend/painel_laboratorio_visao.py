"""Painel 5 — Laboratório de Visão Computacional: Canvas de desenho + Upload de imagem real."""

import numpy as np
import streamlit as st

from src.frontend.estilos import aplicar_estilos, titulo_secao, kpi_tile

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

_TEMA = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark")

# ── Bubble Sort (usado para ranking Top-K) ─────────────────────────────────


def ordenar_probabilidades_por_bolha(probs: list[tuple]) -> list[tuple]:
    """
    Ordena lista de (classe, probabilidade) em ordem decrescente via Bubble Sort.

    Args:
        probs: Lista de tuplas (classe: int, probabilidade: float).

    Returns:
        Lista ordenada do maior para o menor.
    """
    arr = list(probs)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j][1] < arr[j + 1][1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def _inferir_com_modelo(fachada, vetor: np.ndarray) -> list[tuple] | None:
    """
    Tenta obter probabilidades do modelo treinado.
    Retorna lista de (classe, prob) ou None se nenhum modelo estiver treinado.
    """
    for nome, modelo in fachada.modelos.items():
        try:
            modelo_sklearn = modelo.modelo
            if hasattr(modelo_sklearn, "predict_proba"):
                probs = modelo_sklearn.predict_proba(vetor)[0]
                classes = list(range(len(probs)))
                return list(zip(classes, probs.tolist()))
            else:
                pred = modelo.prever(vetor)[0]
                probs = [0.0] * 10
                probs[int(pred)] = 1.0
                return list(enumerate(probs))
        except Exception:
            continue
    return None


def _grafico_topk(ranking: list[tuple], k: int = 10) -> None:
    top = ranking[:k]
    rotulos = [f"Dígito {c}" for c, _ in top]
    valores = [round(p * 100, 2) for _, p in top]
    cores = [
        "#58a6ff" if i == 0
        else "#3fb950" if i == 1
        else "#8b949e"
        for i in range(len(top))
    ]

    if PLOTLY_OK:
        fig = go.Figure(go.Bar(
            x=valores, y=rotulos, orientation="h",
            marker_color=cores, text=[f"{v:.2f}%" for v in valores],
            textposition="outside",
        ))
        fig.update_layout(
            **_TEMA, height=320, margin=dict(t=10, b=10, l=80),
            xaxis_title="Probabilidade (%)", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        df_top = {r: v for r, v in zip(rotulos, valores)}
        st.bar_chart(df_top)


def _pipeline_visual(img_orig: np.ndarray) -> tuple:
    """
    Executa e devolve as 4 etapas do pipeline de visão para exibição.
    Retorna: (gray, invertida, bbox_crop, final_28x28)
    """
    import cv2

    # Etapa 1: grayscale
    if len(img_orig.shape) == 3:
        gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_orig.copy()

    # Etapa 2: inversão (fundo preto, dígito branco)
    invertida = 255 - gray

    # Etapa 3: bounding box
    _, bin_img = cv2.threshold(invertida, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(bin_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        bbox_crop = invertida[y:y + h, x:x + w]
    else:
        bbox_crop = invertida

    # Etapa 4: resize 20×20 → canvas centralizado 28×28
    resized = cv2.resize(bbox_crop, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized

    return gray, invertida, bbox_crop, canvas


# ── Painel principal ───────────────────────────────────────────────────────

def _renderizar_modo_canvas():
    """Renderiza o canvas de desenho e retorna img_array ou None."""
    try:
        from streamlit_drawable_canvas import st_canvas  # type: ignore
        titulo_secao("Desenhe o dígito abaixo")
        col_canvas, col_config = st.columns([2, 1])
        with col_config:
            espessura = st.slider("Espessura do traço", 10, 40, 20)
            cor_traço = st.color_picker("Cor do traço", "#FFFFFF")
        with col_canvas:
            resultado_canvas = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=espessura,
                stroke_color=cor_traço,
                background_color="#000000",
                height=280, width=280,
                drawing_mode="freedraw",
                key="canvas_digito",
            )
        if resultado_canvas.image_data is not None:
            return resultado_canvas.image_data[:, :, :3].astype(np.uint8)
    except ImportError:
        st.error(
            "Componente `streamlit-drawable-canvas` não instalado. "
            "Execute: `pip install streamlit-drawable-canvas`"
        )
    return None


def _renderizar_modo_upload():
    """Renderiza o uploader de imagem e retorna img_array ou None."""
    titulo_secao("Faça upload de uma foto do dígito")
    arquivo = st.file_uploader(
        "Formatos suportados: JPG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if not arquivo or not PIL_OK:
        return None
    try:
        from guardrails.validador_imagem_entrada import ValidadorImagemEntrada
        import tempfile
        import os
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(arquivo.name)[1]) as tmp:
                tmp.write(arquivo.getvalue())
                tmp_path = tmp.name
            ValidadorImagemEntrada.validar_arquivo(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        st.error(f"Imagem inválida: {e}")
        return None
    pil_img = Image.open(arquivo).convert("RGB")
    img_array = np.array(pil_img)
    st.image(pil_img, caption="Imagem carregada", width=200)
    return img_array


def _renderizar_pipeline_e_inferencia(fachada, img_array: np.ndarray) -> None:
    """Executa pipeline visual e inferência sobre img_array."""
    st.divider()
    titulo_secao("Pipeline de Transformação (4 Etapas)")
    try:
        gray, invertida, bbox_crop, canvas_28 = _pipeline_visual(img_array)
        col1, col2, col3, col4 = st.columns(4)
        col1.image(img_array, caption="① Original", width=110, clamp=True)
        col2.image(invertida, caption="② Grayscale/Invertida", width=110, clamp=True)
        col3.image(bbox_crop, caption="③ Bounding Box", width=110, clamp=True)
        col4.image(canvas_28, caption="④ 28×28 Centralizado", width=110, clamp=True)
    except Exception as e:
        st.error(f"Erro no pipeline de visão: {e}")
        return

    st.divider()
    titulo_secao("Inferência e Ranking Top-K (Bubble Sort)")
    vetor = (canvas_28 / 255.0).flatten().reshape(1, -1)
    ranking_raw = _inferir_com_modelo(fachada, vetor)
    if ranking_raw is None:
        st.info("Treine um modelo no **Painel de Benchmarks** para ver a inferência aqui.")
        return

    ranking = ordenar_probabilidades_por_bolha(ranking_raw)
    melhor_classe, melhor_prob = ranking[0]
    k1, k2 = st.columns(2)
    k1.markdown(kpi_tile(f"Dígito {melhor_classe}", "🎯 Predição"), unsafe_allow_html=True)
    k2.markdown(kpi_tile(f"{melhor_prob * 100:.1f}%", "Confiança"), unsafe_allow_html=True)

    from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza
    probs_array = np.array([p for _, p in sorted(ranking_raw, key=lambda x: x[0])])
    avaliacao = ValidadorFalsaCerteza().avaliar_predicao(probs_array, list(range(10)))
    if avaliacao["alerta_overconfidence"]:
        st.warning("⚠️ Alerta de Falsa Certeza: confiança alta em classe potencialmente desconhecida.")

    st.markdown("<br>", unsafe_allow_html=True)
    _grafico_topk(ranking)
    with st.expander("🔢 Ver ranking completo (Bubble Sort)"):
        for pos, (cls, prob) in enumerate(ranking):
            barra = "█" * int(prob * 30)
            st.text(f"#{pos + 1:2d}  Dígito {cls}  {prob * 100:6.2f}%  {barra}")


def renderizar(fachada) -> None:
    """Ponto de entrada do Painel 5 — recebe FachadaPipelineIA já inicializada."""
    aplicar_estilos()
    st.markdown("## ✍️ Laboratório de Visão Computacional")
    st.caption(
        "Desenhe um dígito ou faça upload de uma foto real. "
        "O pipeline processa e classifica em tempo real com Bubble Sort Top-K."
    )
    if not fachada.modelos:
        st.warning(
            "⚠️ Nenhum modelo treinado ainda. Acesse o **Painel de Benchmarks** "
            "e treine ao menos um modelo antes de usar o Laboratório.")

    modo = st.radio("Modo de entrada", ["✍️ Canvas (Desenho)", "📷 Upload de Imagem"], horizontal=True)

    if "Canvas" in modo:
        img_array = _renderizar_modo_canvas()
    else:
        img_array = _renderizar_modo_upload()

    if img_array is not None and img_array.sum() > 0:
        _renderizar_pipeline_e_inferencia(fachada, img_array)
