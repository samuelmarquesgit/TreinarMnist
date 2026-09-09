"""Fábrica de modelos de Machine Learning da plataforma MNIST.

Implementa o Padrão Factory Method (GoF) com registro centralizado e dinâmico
de algoritmos. Expõe uma API limpa para instanciação, listagem e verificação
dos modelos disponíveis, sem vazar detalhes de implementação para as camadas
superiores (Fachada, Frontend, MCP).

Nota de logging:
    Este módulo é uma biblioteca interna. Nunca chama ``logging.basicConfig()``
    — a configuração do handler é exclusiva do ponto de entrada da aplicação
    (``main.py``, ``app.py``). Usa apenas ``logger = logging.getLogger(__name__)``
    para emitir mensagens rastreáveis sem interferir no pipeline de logs do
    sistema pai.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.modelos.base_modelo import ModeloAbstratoIA
from src.utilitarios.excecoes import ModeloNaoEncontradoError

# Logger de módulo — sem basicConfig; configuração delegada ao ponto de entrada.
logger = logging.getLogger(__name__)

# Tipo para as funções construtoras do registro.
_Construtor = Callable[[], Any]


# ──────────────────────────────────────────────────────────────────────────────
# Funções auxiliares de conversão de scores → probabilidades
# ──────────────────────────────────────────────────────────────────────────────


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """Aplica Softmax estável numericamente sobre a última dimensão.

    Args:
        logits: Array de shape ``(N, C)`` com scores brutos.

    Returns:
        Array de shape ``(N, C)`` com valores em ``[0, 1]`` somando 1 por linha.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return (exp_vals / exp_vals.sum(axis=-1, keepdims=True)).astype(np.float64)  # type: ignore[no-any-return]


def _sigmoid(scores: NDArray[np.floating]) -> NDArray[np.float64]:
    """Aplica Sigmoid elemento a elemento (classificador binário).

    Args:
        scores: Array 1-D de shape ``(N,)`` com scores de decisão.

    Returns:
        Array de shape ``(N, 2)`` com colunas ``[P(classe_0), P(classe_1)]``.
    """
    p_pos = (1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))).reshape(-1, 1)
    return np.hstack([1.0 - p_pos, p_pos])


def _one_hot_a_partir_de_predict(
    estimador: Any,
    X: NDArray[np.floating],
    nome_log: str,
) -> NDArray[np.float64]:
    """Constrói distribuição One-Hot a partir de ``predict()`` como fallback.

    Utilizado quando o estimador não expõe ``predict_proba`` nem
    ``decision_function``. Emite aviso formal no log.

    Args:
        estimador: Estimador Scikit-Learn já ajustado.
        X: Amostras de entrada de shape ``(N, F)``.
        nome_log: Nome do modelo para mensagens de log.

    Returns:
        Array de shape ``(N, n_classes)`` com distribuição One-Hot (confiança 1.0
        na classe prevista, 0.0 nas demais).
    """
    logger.warning(
        "[%s] Estimador não possui predict_proba nem decision_function. "
        "Usando distribuição One-Hot a partir de predict() — "
        "probabilidades não são calibradas.",
        nome_log,
    )
    previsoes = estimador.predict(X)
    classes = getattr(estimador, "classes_", np.unique(previsoes))
    n_classes = len(classes)
    classe_para_idx = {c: i for i, c in enumerate(classes)}

    n_amostras = len(previsoes)
    probs = np.zeros((n_amostras, n_classes), dtype=np.float64)
    for i, pred in enumerate(previsoes):
        idx = classe_para_idx.get(pred, 0)
        probs[i, idx] = 1.0
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper Scikit-Learn
# ──────────────────────────────────────────────────────────────────────────────


class ModeloSklearn(ModeloAbstratoIA):
    """Wrapper que integra estimadores Scikit-Learn à interface ``ModeloAbstratoIA``.

    Encapsula qualquer estimador sklearn (ou Pipeline sklearn) garantindo
    cumprimento do contrato ``treinar / prever / prever_probabilidades`` com
    extração robusta de probabilidades em três níveis de fallback.

    Attributes:
        modelo: Estimador ou Pipeline sklearn encapsulado.
        nome_log: Identificador do modelo para rastreamento em logs.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> wrapper = ModeloSklearn(LogisticRegression(), nome_log="LR")
        >>> wrapper.treinar(X_treino, y_treino)
        >>> probs = wrapper.prever_probabilidades(X_teste)  # shape (N, 10)
    """

    def __init__(self, modelo: Any, nome_log: str = "Desconhecido") -> None:
        self.modelo: Any = modelo
        self.nome_log: str = nome_log

    # ── Interface obrigatória ──────────────────────────────────────────────────

    def treinar(
        self,
        X_treino: NDArray[np.floating],
        y_treino: NDArray[np.integer],
    ) -> None:
        """Ajusta o estimador encapsulado sobre os dados de treinamento.

        Args:
            X_treino: Matriz de features de shape ``(N, F)``.
            y_treino: Vetor de rótulos de shape ``(N,)``.
        """
        logger.info(
            "[%s] Iniciando treinamento com %d amostras…", self.nome_log, len(X_treino)
        )
        self.modelo.fit(X_treino, y_treino)
        logger.info("[%s] Treinamento concluído.", self.nome_log)

    def prever(
        self, X_teste: NDArray[np.floating]
    ) -> NDArray[np.integer]:
        """Realiza predição de classes sobre as amostras de entrada.

        Args:
            X_teste: Matriz de features de shape ``(N, F)``.

        Returns:
            Vetor de classes preditas de shape ``(N,)``.
        """
        logger.debug("[%s] Inferência em %d amostras…", self.nome_log, len(X_teste))
        return self.modelo.predict(X_teste)  # type: ignore[no-any-return]

    def prever_probabilidades(
        self, X_teste: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Retorna distribuição de probabilidade por classe para cada amostra.

        Inspeciona o estimador encapsulado em três níveis de fallback:

        1. **predict_proba** — utilizado diretamente se disponível (Árvores,
           Florestas, Regressão Logística, GaussianNB, MLP, SVC com
           ``probability=True``).
        2. **decision_function** — scores convertidos via Softmax (multi-classe)
           ou Sigmoid (binário) para estimadores como SVM linear ou SGD.
        3. **One-Hot via predict** — fallback final; emite aviso de log pois
           não produz probabilidades calibradas.

        Para estimadores encapsulados em ``Pipeline``, inspeciona o passo
        final (``steps[-1][1]``) para determinar qual caminho seguir.

        Args:
            X_teste: Matriz de features de shape ``(N, F)``.

        Returns:
            Array NumPy 2-D de shape ``(N, n_classes)`` com valores em
            ``[0.0, 1.0]`` somando 1.0 por linha.
        """
        logger.debug(
            "[%s] Extraindo probabilidades para %d amostras…", self.nome_log, len(X_teste)
        )

        # Identifica o estimador final (navega por Pipeline se necessário)
        estimador_final: Any = (
            self.modelo.steps[-1][1]
            if isinstance(self.modelo, Pipeline)
            else self.modelo
        )

        # ── Nível 1: predict_proba ────────────────────────────────────────────
        if hasattr(estimador_final, "predict_proba"):
            logger.debug("[%s] Usando predict_proba.", self.nome_log)
            probs = np.array(self.modelo.predict_proba(X_teste), dtype=np.float64)
            return probs

        # ── Nível 2: decision_function ────────────────────────────────────────
        if hasattr(estimador_final, "decision_function"):
            logger.debug(
                "[%s] predict_proba indisponível — usando decision_function + softmax/sigmoid.",
                self.nome_log,
            )
            scores = np.array(self.modelo.decision_function(X_teste), dtype=np.float64)
            if scores.ndim == 1:
                # Classificador binário → Sigmoid
                return _sigmoid(scores)
            # Multi-classe → Softmax
            return _softmax(scores)

        # ── Nível 3: One-Hot via predict ──────────────────────────────────────
        return _one_hot_a_partir_de_predict(self.modelo, X_teste, self.nome_log)


# ──────────────────────────────────────────────────────────────────────────────
# Fábrica de Modelos
# ──────────────────────────────────────────────────────────────────────────────


class FabricaModelos:
    """Fábrica centralizada de algoritmos de ML (GoF — Factory Method).

    Mantém um registro privado de construtores (``_REGISTRO_MODELOS``) e
    uma entrada especial para o ``VisionTransformer`` (importação lazy para
    evitar dependência obrigatória de PyTorch em toda a aplicação).

    O registro pode ser inspecionado via ``listar_disponiveis()`` por qualquer
    camada do sistema, garantindo sincronização automática entre a fábrica e
    os painéis de frontend.

    Attributes:
        _REGISTRO_MODELOS: Dicionário imutável de ``{chave: callable}`` com
            os construtores dos estimadores sklearn suportados.

    Example:
        >>> modelo = FabricaModelos.criar_modelo("FlorestaAleatoria")
        >>> print(FabricaModelos.listar_disponiveis())
        ['RegressaoLogistica', 'ArvoreDecisao', ..., 'VisionTransformer']
    """

    _REGISTRO_MODELOS: dict[str, _Construtor] = {  # noqa: RUF012
        "RegressaoLogistica": lambda: LogisticRegression(
            max_iter=500, solver="lbfgs", multi_class="auto", random_state=42
        ),
        "ArvoreDecisao": lambda: DecisionTreeClassifier(
            max_depth=20, random_state=42
        ),
        "FlorestaAleatoria": lambda: RandomForestClassifier(
            n_estimators=50, max_depth=20, n_jobs=-1, random_state=42
        ),
        "ImpulsionamentoGradiente": lambda: GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42
        ),
        "SVM": lambda: SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42
        ),
        "KNN": lambda: KNeighborsClassifier(
            n_neighbors=5, metric="euclidean", n_jobs=-1
        ),
        "NaiveBayes": lambda: GaussianNB(var_smoothing=1e-9),
        "PerceptronMulticamadas": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
    }

    # Chave especial para o ViT (importação lazy de PyTorch)
    _CHAVE_VIT: str = "VisionTransformer"

    @staticmethod
    def listar_disponiveis() -> list[str]:
        """Retorna a lista canônica e completa de chaves aceitas por ``criar_modelo()``.

        A lista é derivada diretamente do registro interno — nunca hardcoded
        em outra camada — garantindo que adições ao registro sejam refletidas
        automaticamente em todo o sistema.

        Returns:
            Lista de strings com as chaves válidas, em ordem de inserção,
            com ``"VisionTransformer"`` ao final.
        """
        return list(FabricaModelos._REGISTRO_MODELOS.keys()) + [FabricaModelos._CHAVE_VIT]

    @staticmethod
    def esta_registrado(nome_modelo: str) -> bool:
        """Verifica se uma chave está registrada na fábrica sem instanciar o modelo.

        Args:
            nome_modelo: Chave a verificar.

        Returns:
            ``True`` se ``nome_modelo`` é suportado, ``False`` caso contrário.
        """
        return nome_modelo in FabricaModelos._REGISTRO_MODELOS or nome_modelo == FabricaModelos._CHAVE_VIT

    @staticmethod
    def criar_modelo(nome_modelo: str) -> ModeloAbstratoIA:
        """Instancia e retorna o modelo solicitado encapsulado na interface base.

        Para modelos Scikit-Learn, retorna um ``ModeloSklearn`` configurado
        com os hiperparâmetros padrão do projeto. Para o ``VisionTransformer``,
        importa ``ModeloViT`` sob demanda (lazy import) para não forçar a
        dependência de PyTorch em todo o sistema.

        Args:
            nome_modelo: Chave canônica do modelo (ex: ``"FlorestaAleatoria"``,
                ``"VisionTransformer"``). Deve corresponder exatamente a uma
                entrada em ``_REGISTRO_MODELOS`` ou à chave especial do ViT.

        Returns:
            Instância de ``ModeloAbstratoIA`` pronta para receber ``treinar()``.

        Raises:
            ModeloNaoEncontradoError: Se ``nome_modelo`` não constar no registro.
                A mensagem lista todas as chaves válidas para facilitar diagnóstico.

        Example:
            >>> m = FabricaModelos.criar_modelo("SVM")
            >>> m.treinar(X_treino, y_treino)
        """
        # ── VisionTransformer (importação lazy) ───────────────────────────────
        if nome_modelo == FabricaModelos._CHAVE_VIT:
            logger.info("[Fábrica] Instanciando VisionTransformer (importação lazy PyTorch).")
            from src.modelos.vision_transformer import ModeloViT
            return ModeloViT(nome_log=nome_modelo)

        # ── Modelos Scikit-Learn ──────────────────────────────────────────────
        construtor: _Construtor | None = FabricaModelos._REGISTRO_MODELOS.get(nome_modelo)

        if construtor is None:
            chaves_validas = FabricaModelos.listar_disponiveis()
            logger.error(
                "[Fábrica] Tentativa de instanciar modelo não registrado: '%s'. "
                "Chaves válidas: %s",
                nome_modelo,
                chaves_validas,
            )
            raise ModeloNaoEncontradoError(
                f"Modelo '{nome_modelo}' não encontrado no registro da fábrica.\n"
                f"Modelos disponíveis: {chaves_validas}"
            )

        logger.info("[Fábrica] Instanciando '%s'.", nome_modelo)
        return ModeloSklearn(construtor(), nome_log=nome_modelo)
