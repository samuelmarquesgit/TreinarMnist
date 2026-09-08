"""Motor analítico de robustez OOD (Out-Of-Distribution) para a plataforma MNIST.

Implementa Class Masking: classes são ocultadas do treino para avaliar se o modelo
emite Falsa Certeza (Overconfidence) ao encontrar esses dígitos na inferência.

Nota de logging:
    Biblioteca interna — nunca chama ``logging.basicConfig()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.schemas import RelatorioOOD

import numpy as np
from numpy.typing import NDArray

from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza
from src.modelos.base_modelo import ModeloAbstratoIA

logger = logging.getLogger(__name__)

# Classes padrão ocultadas conforme especificação do projeto
_CLASSES_OOD_PADRAO: list[int] = [4, 7]


def _entropia_shannon(prob: NDArray[np.float64]) -> float:
    """Calcula a Entropia de Shannon para um vetor de probabilidades.

    Args:
        prob: Vetor de probabilidades de shape ``(C,)`` com valores em ``[0, 1]``.

    Returns:
        Entropia em nats (float não-negativo).
    """
    p = np.clip(prob, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


class AnalisadorRobustezOOD:
    """Motor analítico para simulação de dados Out-Of-Distribution (OOD).

    Mascaramos intencionalmente algumas classes durante o treino (ex: 4 e 7)
    para avaliar se o modelo emite falsa certeza (overconfidence) ao encontrar
    esses dígitos ocultos na fase de inferência.

    Attributes:
        validador: Instância de ``ValidadorFalsaCerteza`` configurada.
        classes_mascaradas: Lista de classes ocultadas; definida em
            ``preparar_dados_id()``.

    Example:
        >>> analisador = AnalisadorRobustezOOD()
        >>> X_id, y_id = analisador.preparar_dados_id(X, y)
        >>> X_ood, y_ood = analisador.isolar_dados_ood(X, y)
        >>> relatorio = analisador.relatorio_overconfidence(modelo, X_ood, y_ood)
    """

    def __init__(self, limiar_alerta: float = 0.85) -> None:
        self.validador = ValidadorFalsaCerteza(limiar_alerta_certeza=limiar_alerta)
        self.classes_mascaradas: list[int] = []

    def preparar_dados_id(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        classes_ocultas: list[int] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Remove as classes especificadas criando um conjunto In-Distribution (ID).

        Args:
            X: Matriz de features completa de shape ``(N, F)``.
            y: Vetor de rótulos de shape ``(N,)``.
            classes_ocultas: Classes a excluir do conjunto ID.
                Padrão: ``[4, 7]`` conforme especificação do projeto.

        Returns:
            Tupla ``(X_id, y_id)`` sem as instâncias das classes ocultas.
        """
        if classes_ocultas is None:
            classes_ocultas = list(_CLASSES_OOD_PADRAO)
        self.classes_mascaradas = classes_ocultas
        mascara_id = ~np.isin(y, classes_ocultas)
        X_id = X[mascara_id]
        y_id = y[mascara_id]
        logger.info(
            "Dados ID preparados. Classes mascaradas (OOD): %s | "
            "Amostras ID: %d | Amostras removidas: %d",
            classes_ocultas,
            len(X_id),
            len(X) - len(X_id),
        )
        return X_id, y_id

    def isolar_dados_ood(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Isola exclusivamente as classes ocultadas para teste de estresse.

        Args:
            X: Matriz de features completa.
            y: Vetor de rótulos completo.

        Returns:
            Tupla ``(X_ood, y_ood)`` contendo apenas instâncias OOD.

        Raises:
            ValueError: Se ``preparar_dados_id()`` não tiver sido chamado primeiro.
        """
        if not self.classes_mascaradas:
            raise ValueError(
                "Classes mascaradas não foram definidas. "
                "Execute preparar_dados_id() antes de isolar_dados_ood()."
            )
        mascara_ood = np.isin(y, self.classes_mascaradas)
        X_ood, y_ood = X[mascara_ood], y[mascara_ood]
        logger.info(
            "Dados OOD isolados: %d amostras das classes %s.",
            len(X_ood),
            self.classes_mascaradas,
        )
        return X_ood, y_ood

    def relatorio_overconfidence(
        self,
        modelo: ModeloAbstratoIA,
        X_ood: NDArray[np.float32],
        y_ood_real: NDArray[np.int32],
    ) -> RelatorioOOD:
        """Submete o modelo às instâncias OOD e mensura a taxa de Falsa Certeza.

        Compatível com a nova interface ``ResultadoValidacao`` (NamedTuple) do
        ``ValidadorFalsaCerteza``. Calcula entropia de Shannon diretamente
        sobre o vetor de probabilidades para independência de interface.

        Args:
            modelo: ``ModeloAbstratoIA`` treinado exclusivamente nos dados ID.
            X_ood: Instâncias OOD de shape ``(N, F)``.
            y_ood_real: Rótulos originais das instâncias OOD de shape ``(N,)``.

        Returns:
            RelatorioOOD: Objeto Pydantic com o relatório.

        Raises:
            TypeError: Se o modelo não implementar ``prever_probabilidades()``.
        """
        from src.schemas import RelatorioOOD

        if not hasattr(modelo, "prever_probabilidades"):
            raise TypeError(
                "O modelo deve implementar 'prever_probabilidades' "
                "para análise de entropia OOD."
            )

        probabilidades: NDArray[np.float64] = modelo.prever_probabilidades(X_ood)
        total_amostras = len(X_ood)
        alertas_overconfidence = 0
        entropia_soma = 0.0

        # O modelo conhece apenas as classes não mascaradas
        classes_conhecidas = [c for c in range(10) if c not in self.classes_mascaradas]

        for prob in probabilidades:
            # ── Interface nova: ResultadoValidacao(NamedTuple) ────────────────
            resultado = self.validador.avaliar_predicao(prob, classes_conhecidas)

            # Compatibilidade com NamedTuple (.alerta_falsa_certeza) e dict legado
            if hasattr(resultado, "alerta_falsa_certeza"):
                alerta = resultado.alerta_falsa_certeza
            else:
                alerta = resultado.get("alerta_overconfidence", False)  # type: ignore[union-attr]

            if alerta:
                alertas_overconfidence += 1

            # Entropia calculada diretamente — independente da interface do guardrail
            entropia_soma += _entropia_shannon(prob)

        taxa_overconfidence = (
            alertas_overconfidence / total_amostras if total_amostras > 0 else 0.0
        )
        entropia_media = entropia_soma / total_amostras if total_amostras > 0 else 0.0

        logger.warning(
            "Relatório OOD: %.2f%% de Falsa Certeza detectada em %d amostras.",
            taxa_overconfidence * 100,
            total_amostras,
        )

        return RelatorioOOD(
            total_amostras_ood=total_amostras,
            total_falsa_certeza=alertas_overconfidence,
            taxa_overconfidence=taxa_overconfidence,
            entropia_media=entropia_media,
            classes_ood=self.classes_mascaradas,
        )


# ── Função de conveniência para o painel Streamlit ────────────────────────────


def executar_experimento_ood(
    fachada: Any,
    classes_mascaradas: list[int] | None = None,
    n_amostras: int = 200,
) -> NDArray[np.float64]:
    """Executa experimento OOD real usando a ``FachadaPipelineIA``.

    Isola amostras das classes mascaradas do dataset bruto e retorna as
    probabilidades inferidas pelo primeiro modelo treinado disponível.
    Se nenhum modelo estiver treinado, treina ``RegressaoLogistica`` nos
    dados In-Distribution.

    Args:
        fachada: Instância de ``FachadaPipelineIA`` (tipagem Any para evitar
            importação circular; o contrato é duck-typing).
        classes_mascaradas: Classes a excluir do treino. Padrão: ``[4, 7]``.
        n_amostras: Número máximo de amostras OOD a retornar.

    Returns:
        Array de shape ``(n_amostras, 10)`` com probabilidades por classe.

    Raises:
        RuntimeError: Se os dados brutos não estiverem disponíveis na fachada.
    """
    if classes_mascaradas is None:
        classes_mascaradas = list(_CLASSES_OOD_PADRAO)

    if not fachada.dados_inicializados():
        fachada.inicializar_dados()

    if fachada.X is None or fachada.y is None:
        raise RuntimeError(
            "FachadaPipelineIA.X / .y não disponíveis. "
            "Chame fachada.inicializar_dados() antes do experimento OOD."
        )

    analisador = AnalisadorRobustezOOD()
    X_id, y_id = analisador.preparar_dados_id(fachada.X, fachada.y, classes_mascaradas)
    X_ood, _ = analisador.isolar_dados_ood(fachada.X, fachada.y)

    # Usa modelo já treinado ou treina RegressaoLogistica nos dados ID
    modelos_treinados = fachada.listar_modelos_treinados()
    if modelos_treinados:
        modelo = fachada.modelos[modelos_treinados[0]]
        logger.info(
            "[OOD] Usando modelo '%s' para inferência.", modelos_treinados[0]
        )
    else:
        logger.info(
            "[OOD] Nenhum modelo treinado encontrado. "
            "Treinando RegressaoLogistica nos dados ID..."
        )
        from src.modelos.fabrica_modelos import FabricaModelos
        from src.pre_processamento import pre_processar_dados

        X_id_tr, _, y_id_tr, _, scaler = pre_processar_dados(X_id, y_id)
        modelo = FabricaModelos.criar_modelo("RegressaoLogistica")
        modelo.treinar(X_id_tr, y_id_tr)

    # Normaliza amostras OOD com o scaler da fachada (se disponível)
    scaler = getattr(fachada, "scaler", None)
    if scaler is not None:
        X_ood_norm = scaler.transform(X_ood[:n_amostras]).astype(np.float32)
    else:
        X_ood_norm = X_ood[:n_amostras]

    res = modelo.prever_probabilidades(X_ood_norm)
    return res  # type: ignore[no-any-return]
