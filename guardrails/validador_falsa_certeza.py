"""
Guardrail de Falsa Certeza (Overconfidence) — Plataforma MNIST.

Detecta predições em que o modelo expressa confiança quase determinística
(p_max >= limiar_confianca) combinada com evidências de overconfidence:
entropia colapsada, classe fora do domínio de treino, ou razão de
concentração extrema.

Compatível com modelos scikit-learn via ExtratorProbabilidades, que suporta
predict_proba, decision_function e sklearn.pipeline.Pipeline.

Correção da versão anterior:
    - Limiar elevado de 0.85 → 0.98 (predições legítimas de sklearn
      frequentemente superam 0.85, silenciando o alerta).
    - Três evidências independentes avaliadas (OR); antes apenas entropia.
    - API sklearn-nativa via avaliar_modelo_sklearn().
    - Retorno estruturado (ResultadoValidacao NamedTuple) + dict legado.

Referência matemática:
    H(p) = -sum(pi * log(pi))          Entropia de Shannon
    Delta = p_max - p_second_max        Margem de certeza
    rho   = p_max / (1 - p_max + eps)  Razão de concentração
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_ENTROPIA_MAXIMA_10_CLASSES = float(np.log(10))  # ≈ 2.302 nats


# ──────────────────────────────────────────────────────────────
# Tipo de retorno estruturado
# ──────────────────────────────────────────────────────────────

class ResultadoValidacao(NamedTuple):
    """
    Resultado estruturado da validação de falsa certeza.

    Attributes:
        alerta_falsa_certeza: True quando overconfidence é detectado.
        confianca_maxima:     p_max — probabilidade da classe dominante.
        limiar_utilizado:     Valor do limiar_confianca configurado.
        motivo_auditoria:     Diagnóstico em texto (uma linha).
        detalhes:             Métricas intermediárias:
            classe_prevista (int), entropia (float),
            entropia_normalizada (float), margem (float),
            razao_concentracao (float), classe_fora_dominio (bool),
            confiavel (bool).
    """

    alerta_falsa_certeza: bool
    confianca_maxima: float
    limiar_utilizado: float
    motivo_auditoria: str
    detalhes: Dict[str, Any]


# ──────────────────────────────────────────────────────────────
# Extrator de probabilidades sklearn-compatível
# ──────────────────────────────────────────────────────────────

class ExtratorProbabilidades:
    """
    Extrai probabilidades de estimadores scikit-learn de forma desacoplada.

    Ordem de preferência:
      1. predict_proba(X)      → probabilidades calibradas diretas.
      2. decision_function(X)  → Softmax (multi-classe) ou Sigmoid (binário).
      3. Zeros + aviso de log  → quando nenhum método está disponível.

    Funciona com sklearn.pipeline.Pipeline sem modificação.
    """

    @staticmethod
    def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Softmax linha a linha em (N, C) → (N, C) com soma 1."""
        l_est = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(l_est)
        return exp / np.sum(exp, axis=1, keepdims=True)  # type: ignore[no-any-return]

    @staticmethod
    def _sigmoid_binario(scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sigmoid em (N,) → (N, 2): [P(0), P(1)]."""
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p1, p1])

    @classmethod
    def extrair(
        cls,
        modelo: Any,
        X: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """
        Extrai probabilidades por classe de um estimador sklearn.

        Args:
            modelo: Estimador scikit-learn treinado (ou Pipeline).
            X:      Features de shape (N, n_features), dtype float32.

        Returns:
            Array (N, C) com probabilidades em [0, 1] somando 1 por linha.
            Zeros (N, 1) com warning se nenhum método estiver disponível.

        Raises:
            ValueError: Se X não for 2-D.
        """
        X_arr = np.array(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X deve ter shape (N, features), recebido {X_arr.shape}."
            )

        if hasattr(modelo, "predict_proba"):
            try:
                return np.array(modelo.predict_proba(X_arr), dtype=np.float64)
            except Exception as exc:
                logger.warning(
                    "[ExtratorProbabilidades] predict_proba falhou (%s). "
                    "Tentando decision_function.", exc
                )

        if hasattr(modelo, "decision_function"):
            try:
                scores = np.array(
                    modelo.decision_function(X_arr), dtype=np.float64
                )
                return (
                    cls._sigmoid_binario(scores)
                    if scores.ndim == 1
                    else cls._softmax(scores)
                )
            except Exception as exc:
                logger.warning(
                    "[ExtratorProbabilidades] decision_function falhou (%s). "
                    "Retornando zeros.", exc
                )

        logger.warning(
            "[ExtratorProbabilidades] '%s' não implementa predict_proba "
            "nem decision_function. Probabilidades indisponíveis.",
            type(modelo).__name__,
        )
        return np.zeros((X_arr.shape[0], 1), dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# Validador principal
# ──────────────────────────────────────────────────────────────

class ValidadorFalsaCerteza:
    """
    Guardrail de Falsa Certeza (overconfidence) para a Plataforma MNIST.

    Condição de disparo (dado p_max >= limiar_confianca):
        alerta = (H < limiar_entropia_min)
              OR (classe_prevista not in classes_conhecidas)
              OR (rho = p_max/(1-p_max+eps) >= limiar_razao)

    Args:
        limiar_confianca:    p_max mínimo para acionar avaliação (padrão 0.98).
        limiar_entropia_min: Entropia (nats) abaixo da qual dispara por
                             distribuição colapsada (padrão 0.05).
        limiar_razao:        Razão de odds acima da qual dispara por
                             concentração extrema (padrão 50.0).
    """

    def __init__(
        self,
        limiar_confianca: float = 0.98,
        limiar_entropia_min: float = 0.05,
        limiar_razao: float = 50.0,
    ) -> None:
        if not (0.0 < limiar_confianca <= 1.0):
            raise ValueError(
                f"limiar_confianca deve estar em (0, 1], recebido: {limiar_confianca}"
            )
        self.limiar_confianca = limiar_confianca
        self.limiar_entropia_min = limiar_entropia_min
        self.limiar_razao = limiar_razao

    # Mantém compatibilidade com código que usa o nome antigo do parâmetro
    @property
    def limiar_alerta_certeza(self) -> float:
        """Alias de compatibilidade para limiar_confianca."""
        return self.limiar_confianca

    @property
    def limiar_entropia_baixa(self) -> float:
        """Alias de compatibilidade para limiar_entropia_min."""
        return self.limiar_entropia_min

    # ── Métricas estatísticas ──────────────────────────────────

    @staticmethod
    def calcular_entropia_shannon(probabilidades: NDArray[np.float64]) -> float:
        """
        Calcula H(p) = -sum(pi * log(pi)) em nats.

        Args:
            probabilidades: Vetor 1-D de probabilidades em [0, 1], somando ≈ 1.

        Returns:
            Escalar >= 0. Zero para distribuição determinística.
        """
        p = np.clip(np.array(probabilidades).flatten(), 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def calcular_margem(probabilidades: NDArray[np.float64]) -> float:
        """
        Calcula Delta = p_max - p_second_max.

        Args:
            probabilidades: Vetor 1-D de probabilidades.

        Returns:
            Escalar em [0, 1].
        """
        p = np.sort(np.array(probabilidades).flatten())[::-1]
        return float(p[0] - p[1]) if len(p) >= 2 else float(p[0])

    @staticmethod
    def calcular_razao_concentracao(
        p_max: float, epsilon: float = 1e-9
    ) -> float:
        """
        Calcula rho = p_max / (1 - p_max + epsilon).

        Args:
            p_max:   Probabilidade máxima da predição.
            epsilon: Estabilizador numérico.

        Returns:
            Escalar >= 0. Valores >= 50 indicam concentração extrema.
        """
        return float(p_max / (1.0 - p_max + epsilon))

    # ── Avaliação central ──────────────────────────────────────

    def avaliar_predicao(
        self,
        probabilidades: NDArray[np.float64],
        classes_conhecidas: Optional[List[int]] = None,
    ) -> ResultadoValidacao:
        """
        Avalia se a predição caracteriza Falsa Certeza (overconfidence).

        Regra:
            Dado p_max >= limiar_confianca:
              alerta = entropia_colapsada OR classe_fora_dominio OR razao_extrema

        Args:
            probabilidades:    Vetor 1-D de shape (C,) com P por classe.
            classes_conhecidas: Classes vistas no treino (ex: [0,1,2,3,5,6,8,9]
                               para class masking de 4 e 7). None desativa OOD.

        Returns:
            ResultadoValidacao estruturado.

        Raises:
            ValueError: Se probabilidades for vazio.

        Example:
            >>> v = ValidadorFalsaCerteza(limiar_confianca=0.98)
            >>> p = np.zeros(10); p[3] = 0.99; p[7] = 0.01
            >>> r = v.avaliar_predicao(p, list(range(10)))
            >>> r.alerta_falsa_certeza
            True
        """
        probs = np.array(probabilidades, dtype=np.float64).flatten()
        if probs.size == 0:
            raise ValueError("probabilidades não pode ser vazio.")

        classe_prevista    = int(np.argmax(probs))
        confianca_maxima   = float(np.max(probs))
        entropia           = self.calcular_entropia_shannon(probs)
        entropia_norm      = (
            entropia / _ENTROPIA_MAXIMA_10_CLASSES if len(probs) == 10 else entropia
        )
        margem             = self.calcular_margem(probs)
        razao_concentracao = self.calcular_razao_concentracao(confianca_maxima)

        if classes_conhecidas is not None:
            classe_fora_dominio = classe_prevista not in [int(c) for c in classes_conhecidas]
        else:
            classe_fora_dominio = False

        confianca_extrema  = confianca_maxima >= self.limiar_confianca
        entropia_colapsada = entropia < self.limiar_entropia_min
        razao_extrema      = razao_concentracao >= self.limiar_razao

        alerta = confianca_extrema and (
            entropia_colapsada or classe_fora_dominio or razao_extrema
        )

        # Diagnóstico textual
        if not confianca_extrema:
            motivo = (
                f"Confiança {confianca_maxima:.4f} abaixo do limiar "
                f"{self.limiar_confianca:.4f}. Sem alerta."
            )
        elif alerta:
            ev = []
            if entropia_colapsada:
                ev.append(f"H={entropia:.4f}<{self.limiar_entropia_min}")
            if classe_fora_dominio:
                ev.append(f"classe {classe_prevista} OOD")
            if razao_extrema:
                ev.append(f"rho={razao_concentracao:.1f}>={self.limiar_razao}")
            motivo = (
                f"ALERTA — Falsa Certeza: "
                f"p_max={confianca_maxima:.4f} | {'; '.join(ev)}."
            )
        else:
            motivo = (
                f"p_max={confianca_maxima:.4f} >= limiar, "
                "sem evidência adicional de overconfidence."
            )

        if alerta:
            logger.warning(
                "[ValidadorFalsaCerteza] %s | classe=%d | H=%.4f | "
                "Delta=%.4f | rho=%.1f | OOD=%s",
                motivo, classe_prevista, entropia, margem,
                razao_concentracao, classe_fora_dominio,
            )

        return ResultadoValidacao(
            alerta_falsa_certeza=alerta,
            confianca_maxima=confianca_maxima,
            limiar_utilizado=self.limiar_confianca,
            motivo_auditoria=motivo,
            detalhes={
                "classe_prevista":      classe_prevista,
                "entropia":             entropia,
                "entropia_normalizada": entropia_norm,
                "margem":               margem,
                "razao_concentracao":   razao_concentracao,
                "classe_fora_dominio":  classe_fora_dominio,
                "confiavel": (
                    not alerta
                    and not classe_fora_dominio
                    and confianca_maxima >= 0.5
                ),
            },
        )

    def avaliar_modelo_sklearn(
        self,
        modelo: Any,
        X: NDArray[np.float32],
        classes_conhecidas: Optional[List[int]] = None,
    ) -> List[ResultadoValidacao]:
        """
        Avalia overconfidence em um lote via estimador sklearn.

        Args:
            modelo:            Estimador scikit-learn treinado (ou Pipeline).
            X:                 Features (N, n_features), dtype float32.
            classes_conhecidas: Classes do treino; None desativa OOD.

        Returns:
            Lista de N ResultadoValidacao, um por amostra.

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> clf = RandomForestClassifier().fit(X_tr, y_tr)
            >>> res = v.avaliar_modelo_sklearn(clf, X_test)
            >>> alertas = [r for r in res if r.alerta_falsa_certeza]
        """
        probs_lote = ExtratorProbabilidades.extrair(modelo, X)
        return [
            self.avaliar_predicao(probs_lote[i], classes_conhecidas)
            for i in range(probs_lote.shape[0])
        ]

    # ── Compatibilidade com código legado ──────────────────────

    def avaliar_predicao_dict(
        self,
        probabilidades: NDArray[np.float64],
        classes_conhecidas: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Wrapper de compatibilidade — retorna dict com chaves da versão anterior.

        Returns:
            Dict com: classe_prevista, confianca, entropia,
            classe_fora_dominio, alerta_overconfidence, confiavel.
        """
        r = self.avaliar_predicao(probabilidades, classes_conhecidas)
        return {
            "classe_prevista":       r.detalhes["classe_prevista"],
            "confianca":             r.confianca_maxima,
            "entropia":              r.detalhes["entropia"],
            "classe_fora_dominio":   r.detalhes["classe_fora_dominio"],
            "alerta_overconfidence": r.alerta_falsa_certeza,
            "confiavel":             r.detalhes["confiavel"],
        }


# ──────────────────────────────────────────────────────────────
# Testes de referência (importados pelo pytest automaticamente)
# ──────────────────────────────────────────────────────────────

def test_alerta_dispara_confianca_extrema():
    """p_max = 0.99 com rho extrema → alerta True."""
    import pytest as _pt  # noqa: F401
    v = ValidadorFalsaCerteza(limiar_confianca=0.98)
    probs = np.zeros(10, dtype=np.float64)
    probs[3] = 0.99
    probs[7] = 0.01
    r = v.avaliar_predicao(probs, classes_conhecidas=list(range(10)))
    assert r.alerta_falsa_certeza is True
    assert abs(r.confianca_maxima - 0.99) < 1e-6


def test_sem_alerta_confianca_moderada():
    """p_max = 0.70 (< 0.98) → sem alerta."""
    v = ValidadorFalsaCerteza(limiar_confianca=0.98)
    probs = np.full(10, 0.03, dtype=np.float64)
    probs[5] = 0.70
    r = v.avaliar_predicao(probs, classes_conhecidas=list(range(10)))
    assert r.alerta_falsa_certeza is False


def test_alerta_classe_fora_dominio():
    """Classe 4 prevista com p=0.99 mas 4 foi mascarada → OOD dispara alerta."""
    v = ValidadorFalsaCerteza(limiar_confianca=0.98)
    probs = np.zeros(10, dtype=np.float64)
    probs[4] = 0.99
    probs[9] = 0.01
    r = v.avaliar_predicao(probs, classes_conhecidas=[0, 1, 2, 3, 5, 6, 7, 8, 9])
    assert r.alerta_falsa_certeza is True
    assert r.detalhes["classe_fora_dominio"] is True


def test_sklearn_logistic_regression_dispara_alerta():
    """
    LogisticRegression (C=1000, dados bem separados) deve dar p_max >= 0.90
    em amostra central → alerta dispara com limiar_confianca=0.90.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return  # sklearn ausente — pula silenciosamente

    np.random.seed(0)
    X_tr = np.vstack(
        [np.random.randn(40, 20) + i * 6 for i in range(3)]
    ).astype(np.float32)
    y_tr = np.repeat([0, 1, 2], 40)
    clf = LogisticRegression(C=1000, max_iter=500, random_state=0).fit(X_tr, y_tr)

    x_central = np.zeros((1, 20), dtype=np.float32)  # centróide da classe 0
    v = ValidadorFalsaCerteza(limiar_confianca=0.90)
    resultados = v.avaliar_modelo_sklearn(clf, x_central, classes_conhecidas=[0, 1, 2])
    r = resultados[0]

    assert r.confianca_maxima > 0.90, (
        f"Esperado p_max > 0.90, obtido {r.confianca_maxima:.4f}."
    )
    assert r.alerta_falsa_certeza is True, (
        f"Alerta não disparou. Motivo: {r.motivo_auditoria}"
    )
