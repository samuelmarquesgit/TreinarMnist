"""Painel 5 — Laboratório de Visão Computacional: Canvas de desenho + Upload de imagem.

Fluxo de renderização
---------------------
1. ``renderizar(fachada)`` é chamado pelo ``app.py`` com a instância de
   ``FachadaPipelineIA`` já configurada.
2. O usuário escolhe o modo de entrada: canvas de desenho livre ou upload
   de imagem (PNG / JPG / JPEG).
3. A imagem é carregada **inteiramente em memória** via ``io.BytesIO`` —
   sem gravação em disco — e validada antes do processamento.
4. ``_pipeline_visual()`` aplica as 4 etapas canônicas do pré-processamento
   MNIST (grayscale → inversão → bounding box → canvas 28×28).
5. A inferência é realizada via ``FachadaPipelineIA.prever_probabilidades()``,
   delegando ao backend correto (sklearn / PyTorch) de forma transparente.
6. O ranking Top-K é ordenado com ``sorted()`` nativo (Timsort O(N log N)),
   sem nenhum algoritmo ingênuo manual.

Gestão de recursos
------------------
- Imagens são processadas via ``io.BytesIO`` — sem ``NamedTemporaryFile``.
- Quando um caminho físico for estritamente necessário (ex: validador externo
  via CLI), utiliza-se ``tempfile.NamedTemporaryFile(delete=False)`` dentro
  de um bloco ``try / finally`` com ``Path.unlink(missing_ok=True)``
  garantindo remoção mesmo em caso de exceção.

Nota de logging
---------------
Não chama ``logging.basicConfig()`` — configuração delegada ao ponto de
entrada da aplicação.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from src.frontend.estilos import aplicar_estilos, kpi_tile, titulo_secao

logger = logging.getLogger(__name__)

# ── Dependências opcionais ────────────────────────────────────────────────────

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:  # pragma: no cover
    _PLOTLY_OK = False

try:
    from PIL import Image, UnidentifiedImageError
    _PIL_OK = True
except ImportError:  # pragma: no cover
    _PIL_OK = False

_TEMA_PLOTLY: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "template": "plotly_dark",
}

# Tipos MIME aceitos no upload
_TIPOS_UPLOAD = ["jpg", "jpeg", "png"]


# ──────────────────────────────────────────────────────────────────────────────
# Ordenação (Timsort nativo — sem Bubble Sort)
# ──────────────────────────────────────────────────────────────────────────────


def ordenar_ranking(
    pares_classe_prob: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Ordena pares (classe, probabilidade) em ordem decrescente de confiança.

    Utiliza ``sorted()`` (Timsort, O(N log N)) — substitui o Bubble Sort
    ingênuo que tinha complexidade O(N²) e não era adequado para produção.

    Args:
        pares_classe_prob: Lista de tuplas ``(classe: int, probabilidade: float)``.

    Returns:
        Nova lista ordenada do maior para o menor valor de probabilidade.
    """
    return sorted(pares_classe_prob, key=lambda par: par[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Inferência via Fachada
# ──────────────────────────────────────────────────────────────────────────────


def _inferir_com_fachada(
    fachada: Any,
    vetor: NDArray[np.float32],
) -> tuple[str, list[tuple[int, float]]] | None:
    """Obtém distribuição de probabilidade do primeiro modelo treinado disponível.

    Delega para ``FachadaPipelineIA.prever_probabilidades()``, que gerencia
    os diferentes backends (sklearn, PyTorch/ViT) de forma transparente.

    Args:
        fachada: Instância de ``FachadaPipelineIA``.
        vetor: Array de shape ``(1, 784)`` com a imagem normalizada.

    Returns:
        Tupla ``(nome_modelo, [(classe, prob), ...])`` ou ``None`` se nenhum
        modelo estiver treinado ou se todos falharem.
    """
    modelos_treinados: list[str] = fachada.listar_modelos_treinados()
    if not modelos_treinados:
        return None

    for nome in modelos_treinados:
        try:
            probs: NDArray[np.float64] = fachada.prever_probabilidades(nome, vetor)
            probs_linha = probs[0]  # shape (n_classes,)
            pares = [(int(c), float(p)) for c, p in enumerate(probs_linha)]
            return nome, pares
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[LaboratorioVisao] Falha ao inferir com '%s': %s", nome, exc
            )
            continue

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline de visão computacional
# ──────────────────────────────────────────────────────────────────────────────


def _pipeline_visual(
    img_orig: NDArray[np.uint8],
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
]:
    """Aplica as 4 etapas do pipeline canônico de pré-processamento MNIST.

    Etapas:
        1. Conversão para escala de cinza.
        2. Inversão de intensidade (fundo preto, dígito branco).
        3. Recorte pelo bounding box do dígito.
        4. Redimensionamento 20×20 + padding para canvas 28×28 centralizado.

    Args:
        img_orig: Imagem de entrada em RGB ou escala de cinza, shape ``(H, W)``
            ou ``(H, W, 3)``.

    Returns:
        Tupla ``(gray, invertida, bbox_crop, canvas_28x28)``, todas com dtype
        ``uint8``.

    Raises:
        ImportError: Se OpenCV (``cv2``) não estiver instalado.
        ValueError: Se a imagem de entrada estiver vazia ou com shape inválido.
    """
    import cv2  # type: ignore

    if img_orig.size == 0:
        raise ValueError("Imagem de entrada vazia — shape inválido.")

    # Etapa 1: escala de cinza
    if img_orig.ndim == 3:
        gray: NDArray[np.uint8] = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)  # type: ignore[assignment]
    else:
        gray = img_orig.copy()

    # Etapa 2: inversão (fundo preto, dígito branco)
    invertida: NDArray[np.uint8] = (255 - gray).astype(np.uint8)

    # Etapa 3: bounding box
    _, bin_img = cv2.threshold(invertida, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(bin_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        bbox_crop: NDArray[np.uint8] = invertida[y : y + h, x : x + w]
    else:
        bbox_crop = invertida

    # Etapa 4: resize 20×20 → canvas 28×28 centralizado
    resized: NDArray[np.uint8] = cv2.resize(bbox_crop, (20, 20), interpolation=cv2.INTER_AREA)  # type: ignore[assignment]
    canvas: NDArray[np.uint8] = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized

    return gray, invertida, bbox_crop, canvas


# ──────────────────────────────────────────────────────────────────────────────
# Componentes de UI
# ──────────────────────────────────────────────────────────────────────────────


def _grafico_topk(
    ranking: list[tuple[int, float]],
    k: int = 10,
) -> None:
    """Renderiza gráfico de barras horizontal com o Top-K de probabilidades.

    Args:
        ranking: Lista de ``(classe, probabilidade)`` já ordenada de forma
            decrescente por ``ordenar_ranking()``.
        k: Número máximo de classes a exibir. Padrão: 10.
    """
    top = ranking[:k]
    rotulos = [f"Dígito {c}" for c, _ in top]
    valores = [round(p * 100, 2) for _, p in top]
    cores = [
        "#58a6ff" if i == 0 else "#3fb950" if i == 1 else "#8b949e"
        for i in range(len(top))
    ]

    if _PLOTLY_OK:
        fig = go.Figure(
            go.Bar(
                x=valores,
                y=rotulos,
                orientation="h",
                marker_color=cores,
                text=[f"{v:.2f}%" for v in valores],
                textposition="outside",
            )
        )
        fig.update_layout(
            **_TEMA_PLOTLY,
            height=320,
            margin={"t": 10, "b": 10, "l": 80},
            xaxis_title="Probabilidade (%)",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart({r: v for r, v in zip(rotulos, valores)})


def _renderizar_guardrail(
    ranking_bruto: list[tuple[int, float]],
) -> None:
    """Avalia e exibe alertas do guardrail de falsa certeza.

    Usa ``ValidadorFalsaCerteza`` do módulo de guardrails do projeto.
    Falha silenciosamente (com log) se o módulo não estiver disponível.

    Args:
        ranking_bruto: Lista de ``(classe, prob)`` não necessariamente ordenada.
    """
    try:
        from guardrails.validador_falsa_certeza import (
            ValidadorFalsaCerteza,
        )

        probs_array = np.array(
            [p for _, p in sorted(ranking_bruto, key=lambda x: x[0])],
            dtype=np.float64,
        )
        avaliacao = ValidadorFalsaCerteza().avaliar_predicao(
            probs_array, list(range(10))
        )
        # Suporta tanto ResultadoValidacao (NamedTuple) quanto dict legado
        alerta = (
            avaliacao.alerta_falsa_certeza
            if hasattr(avaliacao, "alerta_falsa_certeza")
            else avaliacao.get("alerta_overconfidence", False)
        )
        if alerta:
            st.warning(
                "⚠️ **Alerta de Falsa Certeza** — confiança extremamente alta "
                "em classe potencialmente fora de domínio (OOD). "
                "Interprete a predição com cautela.",
                icon="⚠️",
            )
    except ImportError:
        logger.debug("[LaboratorioVisao] guardrails.validador_falsa_certeza não disponível.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[LaboratorioVisao] Erro ao avaliar guardrail: %s", exc)


def _renderizar_pipeline_e_inferencia(
    fachada: Any,
    img_array: NDArray[np.uint8],
) -> None:
    """Renderiza o pipeline de 4 etapas e o painel de inferência Top-K.

    Args:
        fachada: Instância de ``FachadaPipelineIA``.
        img_array: Imagem de entrada em uint8, shape ``(H, W)`` ou ``(H, W, 3)``.
    """
    # ── Pipeline visual ───────────────────────────────────────────────────────
    st.divider()
    titulo_secao("Pipeline de Transformação (4 Etapas)")
    try:
        _gray, invertida, bbox_crop, canvas_28 = _pipeline_visual(img_array)
    except Exception as exc:  # noqa: BLE001
        logger.error("[LaboratorioVisao] Falha no pipeline de visão: %s", exc)
        st.error(f"Erro no pipeline de pré-processamento: {exc}", icon="🚨")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.image(img_array,   caption="① Original",            width=110, clamp=True)
    col2.image(invertida,   caption="② Grayscale/Invertida", width=110, clamp=True)
    col3.image(bbox_crop,   caption="③ Bounding Box",        width=110, clamp=True)
    col4.image(canvas_28,   caption="④ 28×28 Centralizado",  width=110, clamp=True)

    # ── Inferência ────────────────────────────────────────────────────────────
    st.divider()
    titulo_secao("Inferência e Ranking Top-K")

    vetor: NDArray[np.float32] = (canvas_28 / 255.0).flatten().reshape(1, -1).astype(np.float32)
    resultado = _inferir_com_fachada(fachada, vetor)

    if resultado is None:
        st.info(
            "Treine um modelo no **Painel de Benchmarks** para ver a inferência aqui.",
            icon="ℹ️",
        )
        return

    nome_modelo, ranking_bruto = resultado
    ranking = ordenar_ranking(ranking_bruto)

    melhor_classe, melhor_prob = ranking[0]
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_tile(f"Dígito {melhor_classe}", "🎯 Predição"), unsafe_allow_html=True)
    k2.markdown(kpi_tile(f"{melhor_prob * 100:.1f}%", "Confiança"), unsafe_allow_html=True)
    k3.markdown(kpi_tile(nome_modelo, "Modelo usado"), unsafe_allow_html=True)

    _renderizar_guardrail(ranking_bruto)

    st.markdown("<br>", unsafe_allow_html=True)
    _grafico_topk(ranking)

    with st.expander("🔢 Ver ranking completo"):
        for pos, (cls, prob) in enumerate(ranking):
            barra = "█" * int(prob * 30)
            st.text(f"#{pos + 1:2d}  Dígito {cls}  {prob * 100:6.2f}%  {barra}")


# ──────────────────────────────────────────────────────────────────────────────
# Modos de entrada
# ──────────────────────────────────────────────────────────────────────────────


def _renderizar_modo_canvas() -> NDArray[np.uint8] | None:
    """Renderiza o canvas de desenho livre e retorna a imagem capturada.

    Requer o pacote opcional ``streamlit-drawable-canvas``.

    Returns:
        Array ``(H, W, 3)`` uint8 com o conteúdo do canvas, ou ``None`` se
        o componente não estiver instalado ou o canvas estiver vazio.
    """
    try:
        from streamlit_drawable_canvas import st_canvas  # type: ignore
    except ImportError:
        st.error(
            "Componente `streamlit-drawable-canvas` não instalado. "
            "Execute: `pip install streamlit-drawable-canvas`",
            icon="🚨",
        )
        return None

    titulo_secao("Desenhe o dígito abaixo")
    col_canvas, col_config = st.columns([2, 1])

    with col_config:
        espessura: int = st.slider("Espessura do traço", 10, 40, 20)
        cor_traco: str = st.color_picker("Cor do traço", "#FFFFFF")

    with col_canvas:
        resultado = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=espessura,
            stroke_color=cor_traco,
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_digito",
        )

    if resultado.image_data is not None:
        return resultado.image_data[:, :, :3].astype(np.uint8)  # type: ignore[no-any-return]
    return None


def _carregar_imagem_de_bytes(
    dados: bytes,
    nome_arquivo: str,
) -> NDArray[np.uint8] | None:
    """Decodifica bytes de imagem em array NumPy inteiramente em memória.

    Não cria arquivos temporários em disco. Usa ``io.BytesIO`` para passar
    os bytes diretamente ao PIL, eliminando I/O desnecessário e risco de
    vazamento de descritores.

    Args:
        dados: Conteúdo binário bruto do arquivo de imagem.
        nome_arquivo: Nome original do arquivo (usado apenas em mensagens de log).

    Returns:
        Array ``(H, W, 3)`` uint8 em RGB, ou ``None`` se a decodificação falhar.
    """
    if not dados:
        logger.warning("[LaboratorioVisao] Upload vazio: '%s'.", nome_arquivo)
        st.warning("O arquivo enviado está vazio.", icon="⚠️")
        return None

    if not _PIL_OK:
        st.error("Pillow não está instalado. Execute: `pip install Pillow`", icon="🚨")
        return None

    try:
        buffer = io.BytesIO(dados)
        pil_img = Image.open(buffer).convert("RGB")
        pil_img.verify()          # Valida integridade sem decodificar pixels
        buffer.seek(0)            # Reinicia após verify()
        pil_img = Image.open(buffer).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except UnidentifiedImageError:
        logger.warning("[LaboratorioVisao] Formato não reconhecido: '%s'.", nome_arquivo)
        st.error(
            f"Não foi possível identificar o formato da imagem **{nome_arquivo}**. "
            "Envie um arquivo PNG ou JPG válido.",
            icon="🚨",
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("[LaboratorioVisao] Erro ao decodificar '%s': %s", nome_arquivo, exc)
        st.error(f"Imagem corrompida ou inválida ({exc}).", icon="🚨")
        return None


def _validar_com_guardrail_arquivo(
    dados: bytes,
    sufixo: str,
    nome_arquivo: str,
) -> bool:
    """Chama ``ValidadorImagemEntrada.validar_arquivo()`` via arquivo temporário gerenciado.

    Cria um ``NamedTemporaryFile`` apenas quando o validador externo exige
    um caminho físico em disco. Usa ``try / finally`` com
    ``Path.unlink(missing_ok=True)`` para garantir remoção mesmo em caso de
    exceção — sem vazamento de descritores ou acúmulo em disco.

    Args:
        dados: Bytes brutos da imagem.
        sufixo: Extensão do arquivo (ex: ``".png"``).
        nome_arquivo: Nome original (usado em mensagens de log/erro).

    Returns:
        ``True`` se a validação passou ou o módulo não está disponível (fail-open).
        ``False`` se o validador rejeitou a imagem (já exibe ``st.error``).
    """
    try:
        from guardrails.validador_imagem_entrada import (
            ValidadorImagemEntrada,
        )
    except ImportError:
        logger.debug("[LaboratorioVisao] ValidadorImagemEntrada não disponível — validação ignorada.")
        return True  # fail-open: módulo ausente não bloqueia o fluxo

    caminho_tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
            tmp.write(dados)
            caminho_tmp = Path(tmp.name)
        ValidadorImagemEntrada.validar_arquivo(str(caminho_tmp))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[LaboratorioVisao] Validação de imagem falhou para '%s': %s", nome_arquivo, exc)
        st.error(f"Imagem inválida: {exc}", icon="🚨")
        return False
    finally:
        if caminho_tmp is not None:
            caminho_tmp.unlink(missing_ok=True)


def _renderizar_modo_upload() -> NDArray[np.uint8] | None:
    """Renderiza o uploader de arquivo e retorna a imagem decodificada.

    O processamento é inteiramente em memória (``io.BytesIO``). Um arquivo
    temporário só é criado quando o guardrail externo exigir caminho físico,
    e é removido deterministicamente após o uso.

    Returns:
        Array ``(H, W, 3)`` uint8 em RGB, ou ``None`` se nenhum arquivo
        válido for enviado.
    """
    titulo_secao("Faça upload de uma foto do dígito")
    arquivo = st.file_uploader(
        "Formatos suportados: JPG, JPEG, PNG",
        type=_TIPOS_UPLOAD,
        help="Envie uma imagem clara do dígito manuscrito sobre fundo branco ou preto.",
    )

    if arquivo is None:
        return None

    dados: bytes = arquivo.getvalue()
    sufixo = Path(arquivo.name).suffix.lower() if arquivo.name else ".png"

    # Validação via guardrail externo (apenas se disponível)
    if not _validar_com_guardrail_arquivo(dados, sufixo, arquivo.name):
        return None

    # Decodificação inteiramente em memória
    img_array = _carregar_imagem_de_bytes(dados, arquivo.name)
    if img_array is None:
        return None

    st.image(img_array, caption=f"Imagem carregada: {arquivo.name}", width=200)
    return img_array


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────────────────────────────────────


def renderizar(fachada: Any) -> None:
    """Ponto de entrada do Painel 5 — recebe ``FachadaPipelineIA`` inicializada.

    Orquestra a seleção do modo de entrada (canvas ou upload), o pipeline
    de pré-processamento visual e a inferência com ranking Top-K ordenado
    pelo Timsort nativo do Python.

    Args:
        fachada: Instância de ``FachadaPipelineIA`` pronta para uso.
    """
    aplicar_estilos()
    st.markdown("## ✍️ Laboratório de Visão Computacional")
    st.caption(
        "Desenhe um dígito ou faça upload de uma foto real. "
        "O pipeline processa e classifica em tempo real com ranking Top-K otimizado."
    )

    if not fachada.listar_modelos_treinados():
        st.warning(
            "⚠️ Nenhum modelo treinado ainda. Acesse o **Painel de Benchmarks** "
            "e treine ao menos um modelo antes de usar o Laboratório.",
            icon="⚠️",
        )

    modo: str = st.radio(
        "Modo de entrada",
        ["✍️ Canvas (Desenho)", "📷 Upload de Imagem"],
        horizontal=True,
    )

    img_array: NDArray[np.uint8] | None = (
        _renderizar_modo_canvas() if "Canvas" in modo else _renderizar_modo_upload()
    )

    if img_array is not None and img_array.sum() > 0:
        _renderizar_pipeline_e_inferencia(fachada, img_array)
