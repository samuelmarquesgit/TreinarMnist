"""Fachada centralizada do pipeline IA da plataforma MNIST.

Este módulo implementa o Padrão Fachada (GoF) como ponto único de entrada
consumido pela CLI (``main.py``), pelo servidor MCP (``mcp_servidor.py``)
e pelo frontend Streamlit (``app.py``).

Responsabilidades:
    - Carregar e pré-processar o dataset MNIST sob demanda.
    - Instanciar modelos via ``FabricaModelos`` (Factory Method).
    - Treinar, avaliar e armazenar modelos em memória de sessão.
    - Expor probabilidades multi-backend para os módulos de robustez/OOD.
    - Executar benchmarks completos com persistência determinística em disco.
    - Registrar experimentos no MLflow quando disponível.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.analise_estatistica import CalculadorEstatistico
from src.avaliacao_metricas import calcular_metricas
from src.carregador_dados import carregar_dados_mnist
from src.modelos.base_modelo import ModeloAbstratoIA
from src.modelos.fabrica_modelos import FabricaModelos
from src.pre_processamento import pre_processar_dados
from src.utilitarios.excecoes import ModeloNaoTreinadoError

# MLflow é opcional — a fachada funciona sem ele.
try:
    import mlflow  # type: ignore
    _MLFLOW_OK = True
except ImportError:
    _MLFLOW_OK = False

logger = logging.getLogger(__name__)

# Diretório padrão para persistência de benchmarks.
_DIR_BENCHMARKS: Path = Path("artifacts") / "benchmarks"

# Tipo auxiliar para matrizes NumPy genéricas.
_Array = NDArray[np.floating]


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """Aplica Softmax estável numericamente sobre a última dimensão.

    Args:
        logits: Array de shape ``(N, C)`` com scores brutos.

    Returns:
        Array de shape ``(N, C)`` com probabilidades somando 1 por linha.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)  # type: ignore[no-any-return]


def _sigmoid(scores: NDArray[np.floating]) -> NDArray[np.float64]:
    """Aplica Sigmoid elemento a elemento (caso binário).

    Args:
        scores: Array 1-D ou 2-D com decisão bruta.

    Returns:
        Array com valores em ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-scores))  # type: ignore[no-any-return]


class ResultadoBenchmark:
    """Contêiner de dados de um único resultado de benchmark.

    Attributes:
        modelo_id: Nome canônico do modelo (chave da fábrica).
        status: ``"sucesso"`` ou ``"falha"``.
        metricas: Dicionário de métricas numéricas; ``None`` em caso de falha.
        erro: Mensagem de erro; ``None`` em caso de sucesso.
        timestamp: Instante ISO 8601 da execução.
    """

    def __init__(
        self,
        modelo_id: str,
        status: str,
        metricas: dict[str, Any] | None = None,
        erro: str | None = None,
    ) -> None:
        self.modelo_id: str = modelo_id
        self.status: str = status
        self.metricas: dict[str, Any] | None = metricas
        self.erro: str | None = erro
        self.timestamp: str = datetime.now(tz=timezone.utc).isoformat()

    def para_dict(self) -> dict[str, Any]:
        """Serializa o resultado em dicionário pronto para JSON.

        Returns:
            Dicionário com todos os campos do resultado.
        """
        return {
            "modelo_id": self.modelo_id,
            "status": self.status,
            "timestamp": self.timestamp,
            "metricas": self.metricas,
            "erro": self.erro,
        }


class FachadaPipelineIA:
    """Fachada (GoF) que orquestra o pipeline completo de ML da plataforma MNIST.

    Encapsula carregamento de dados, treinamento, avaliação, inferência de
    probabilidades multi-backend e persistência de benchmarks, expondo uma
    API limpa para CLI, MCP e frontend.

    Attributes:
        X: Dataset completo de features (70 000 × 784), carregado sob demanda.
        y: Rótulos completos (70 000,), carregado sob demanda.
        X_treino: Partição de treinamento (features).
        X_teste: Partição de teste (features).
        y_treino: Rótulos de treinamento.
        y_teste: Rótulos de teste.
        scaler: Objeto de normalização ajustado sobre ``X_treino``.
        modelos: Registro em memória dos modelos instanciados e treinados.

    Example:
        >>> fachada = FachadaPipelineIA()
        >>> fachada.treinar_modelo("RegressaoLogistica")
        >>> metricas = fachada.avaliar_modelo("RegressaoLogistica")
        >>> probs = fachada.prever_probabilidades("RegressaoLogistica", X_ood)
    """

    def __init__(self) -> None:
        self.X: NDArray[np.float32] | None = None
        self.y: NDArray[np.int32] | None = None
        self.X_treino: NDArray[np.float32] | None = None
        self.X_teste: NDArray[np.float32] | None = None
        self.y_treino: NDArray[np.int32] | None = None
        self.y_teste: NDArray[np.int32] | None = None
        self.scaler: Any = None
        self.modelos: dict[str, ModeloAbstratoIA] = {}

        if _MLFLOW_OK:
            mlflow.set_experiment("Treinamento_MNIST")

    # ──────────────────────────────────────────────────────────────────────────
    # Dados
    # ──────────────────────────────────────────────────────────────────────────

    def inicializar_dados(self) -> None:
        """Carrega o MNIST e executa o pré-processamento completo.

        Popula ``X_treino``, ``X_teste``, ``y_treino``, ``y_teste`` e
        ``scaler``. Idempotente: ignora chamadas subsequentes se os dados
        já estiverem carregados.

        Raises:
            RuntimeError: Se todas as fontes de download falharem
                (propagado de ``carregar_dados_mnist``).
        """
        if self.X_treino is not None:
            logger.debug("[Fachada] Dados já inicializados — ignorando chamada.")
            return

        logger.info("[Fachada] Carregando dataset MNIST…")
        self.X, self.y = carregar_dados_mnist()

        logger.info("[Fachada] Pré-processando dados (split estratificado + normalização)…")
        (
            self.X_treino,
            self.X_teste,
            self.y_treino,
            self.y_teste,
            self.scaler,
        ) = pre_processar_dados(self.X, self.y)

        logger.info(
            "[Fachada] Dados prontos — treino=%d amostras, teste=%d amostras.",
            len(self.X_treino),
            len(self.X_teste),
        )

    def _garantir_dados(self) -> None:
        """Inicializa os dados caso ainda não tenham sido carregados.

        Método interno utilizado pelos demais métodos da fachada para
        garantir o pré-requisito de forma transparente.
        """
        if self.X_treino is None:
            self.inicializar_dados()

    # ──────────────────────────────────────────────────────────────────────────
    # Treino e avaliação
    # ──────────────────────────────────────────────────────────────────────────

    def treinar_modelo(self, nome_modelo: str) -> ModeloAbstratoIA:
        """Instancia e treina um modelo via Factory, armazenando-o em memória.

        Args:
            nome_modelo: Chave canônica registrada em ``FabricaModelos``
                (ex: ``"RegressaoLogistica"``, ``"VisionTransformer"``).

        Returns:
            Instância treinada de ``ModeloAbstratoIA``.

        Raises:
            ValueError: Se ``nome_modelo`` não existir no registro da fábrica.
        """
        self._garantir_dados()
        logger.info("[Fachada] Instanciando modelo '%s'…", nome_modelo)
        modelo = FabricaModelos.criar_modelo(nome_modelo)

        logger.info("[Fachada] Treinando '%s'…", nome_modelo)
        modelo.treinar(self.X_treino, self.y_treino)
        self.modelos[nome_modelo] = modelo
        logger.info("[Fachada] Modelo '%s' treinado e registrado em memória.", nome_modelo)
        return modelo

    def avaliar_modelo(self, nome_modelo: str) -> dict[str, Any]:
        """Avalia o modelo treinado sobre a partição de teste.

        Args:
            nome_modelo: Chave do modelo já treinado e registrado.

        Returns:
            Dicionário com ``acuracia``, ``precisao``, ``recall``, ``f1``
            e ``matriz_confusao`` (lista 10×10).

        Raises:
            ModeloNaoTreinadoError: Se o modelo não foi treinado previamente.
        """
        if nome_modelo not in self.modelos:
            raise ModeloNaoTreinadoError(
                f"Modelo '{nome_modelo}' não foi treinado. Chame treinar_modelo() primeiro."
            )
        self._garantir_dados()
        assert self.X_teste is not None
        assert self.y_teste is not None

        modelo = self.modelos[nome_modelo]
        logger.info("[Fachada] Avaliando '%s' sobre %d amostras de teste…", nome_modelo, len(self.X_teste))
        y_previsto: NDArray[np.int64] = modelo.prever(self.X_teste)
        metricas = calcular_metricas(self.y_teste, y_previsto)
        logger.info(
            "[Fachada] '%s' — acurácia=%.4f  F1=%.4f",
            nome_modelo, metricas["acuracia"], metricas["f1"],
        )
        return metricas

    # ──────────────────────────────────────────────────────────────────────────
    # Probabilidades multi-backend (OOD / Guardrails)
    # ──────────────────────────────────────────────────────────────────────────

    def prever_probabilidades(
        self,
        modelo_id: str,
        x: NDArray[np.floating] | Any,
    ) -> NDArray[np.float64]:
        """Retorna distribuição de probabilidade por classe para cada amostra.

        Lida de forma transparente com os três backends presentes no projeto:

        1. **Scikit-Learn com ``predict_proba``** — chamada direta; retorna
           shape ``(N, C)`` já normalizado.
        2. **Scikit-Learn com ``decision_function``** — scores convertidos
           via Softmax (multi-classe) ou Sigmoid (binário).
        3. **PyTorch / ViT** — inferência com ``torch.no_grad()`` +
           ``torch.softmax()`` sobre os logits; converte para NumPy no final.
        4. **Fallback** — delega para ``modelo.prever_probabilidades()``
           definido na interface base; erros são logados e relançados.

        Args:
            modelo_id: Chave do modelo já treinado.
            x: Array de features de shape ``(N, 784)`` ou ``(N, 1, 28, 28)``
               dependendo do backend.

        Returns:
            Array NumPy de shape ``(N, num_classes)`` com valores em ``[0, 1]``
            somando 1.0 por linha.

        Raises:
            ModeloNaoTreinadoError: Se o modelo não foi treinado.
            NotImplementedError: Se o modelo não suporta nenhum mecanismo de
                probabilidades após todas as tentativas.
        """
        if modelo_id not in self.modelos:
            raise ModeloNaoTreinadoError(
                f"Modelo '{modelo_id}' não foi treinado. Chame treinar_modelo() primeiro."
            )

        modelo = self.modelos[modelo_id]

        # ── Backend 1 & 2: Scikit-Learn via ModeloSklearn ────────────────────
        estimador_sklearn = getattr(modelo, "modelo", None)
        if estimador_sklearn is not None:
            # Navega por Pipeline para chegar no estimador final
            from sklearn.pipeline import Pipeline  # type: ignore
            estimador_final = (
                estimador_sklearn.steps[-1][1]
                if isinstance(estimador_sklearn, Pipeline)
                else estimador_sklearn
            )

            if hasattr(estimador_final, "predict_proba"):
                logger.debug("[Fachada] '%s' — usando predict_proba.", modelo_id)
                return np.array(estimador_sklearn.predict_proba(x), dtype=np.float64)

            if hasattr(estimador_final, "decision_function"):
                logger.debug("[Fachada] '%s' — usando decision_function → softmax.", modelo_id)
                scores = np.array(estimador_sklearn.decision_function(x), dtype=np.float64)
                if scores.ndim == 1:
                    # Classificador binário → Sigmoid
                    p_pos = _sigmoid(scores).reshape(-1, 1)
                    return np.hstack([1.0 - p_pos, p_pos])
                return _softmax(scores)

        # ── Backend 3: PyTorch / ViT ──────────────────────────────────────────
        modulo_torch = getattr(modelo, "model", None)
        device = getattr(modelo, "device", None)
        if modulo_torch is not None and device is not None:
            try:
                import torch  # type: ignore

                logger.debug("[Fachada] '%s' — usando torch.softmax com no_grad.", modelo_id)
                modulo_torch.eval()
                x_np = np.array(x, dtype=np.float32)
                # Aceita (N, 784) → reshape para (N, 1, 28, 28)
                if x_np.ndim == 2 and x_np.shape[1] == 784:
                    x_np = x_np.reshape(-1, 1, 28, 28)
                tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = modulo_torch(tensor)
                    probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy().astype(np.float64)
            except Exception as exc:
                logger.error(
                    "[Fachada] Falha no backend PyTorch para '%s': %s", modelo_id, exc
                )
                raise

        # ── Backend 4: Fallback — interface base ──────────────────────────────
        try:
            logger.debug("[Fachada] '%s' — usando prever_probabilidades() da interface base.", modelo_id)
            resultado = modelo.prever_probabilidades(x)
            return np.array(resultado, dtype=np.float64)
        except NotImplementedError:
            msg = (
                f"Modelo '{modelo_id}' não suporta inferência de probabilidades. "
                "Nenhum dos backends (predict_proba, decision_function, torch.softmax) "
                "está disponível."
            )
            logger.error("[Fachada] %s", msg)
            raise NotImplementedError(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Benchmark com persistência
    # ──────────────────────────────────────────────────────────────────────────

    def executar_benchmark(
        self,
        modelos_ids: list[str],
        dir_saida: str | Path = _DIR_BENCHMARKS,
    ) -> dict[str, ResultadoBenchmark]:
        """Executa ciclo completo de treino + avaliação e persiste os resultados.

        Para cada modelo em ``modelos_ids``:
          1. Treina sobre ``X_treino``.
          2. Avalia sobre ``X_teste`` e mede latência de inferência.
          3. Grava resultado em ``ResultadoBenchmark`` (sucesso ou falha).

        Ao final persiste um arquivo JSON estruturado em ``dir_saida`` com
        timestamp ISO 8601, métricas completas e status por modelo. Em caso
        de erro de I/O, os resultados permanecem acessíveis em memória.

        Args:
            modelos_ids: Lista de chaves canônicas da fábrica a serem avaliadas.
            dir_saida: Diretório onde o JSON de benchmark será salvo.
                Padrão: ``artifacts/benchmarks/``.

        Returns:
            Dicionário ``{modelo_id: ResultadoBenchmark}`` com todos os resultados,
            incluindo falhas individuais (não interrompe o loop).

        Raises:
            ValueError: Se ``modelos_ids`` for vazio.
        """
        if not modelos_ids:
            raise ValueError("A lista de modelos para benchmark não pode ser vazia.")

        self._garantir_dados()
        assert self.X_treino is not None
        assert self.X_teste is not None
        resultados: dict[str, ResultadoBenchmark] = {}
        ts_inicio = datetime.now(tz=timezone.utc).isoformat()
        logger.info("[Fachada] Iniciando benchmark de %d modelos…", len(modelos_ids))

        for nome in modelos_ids:
            logger.info("[Fachada] ── Benchmark: '%s'", nome)
            try:
                # Treinamento
                t_treino_inicio = time.perf_counter()
                self.treinar_modelo(nome)
                tempo_treino_s = round(time.perf_counter() - t_treino_inicio, 4)

                # Avaliação
                metricas_base = self.avaliar_modelo(nome)

                # Latência de inferência (medida separada sobre X_teste)
                n_amostras = len(self.X_teste)
                t_inf_inicio = time.perf_counter()
                self.modelos[nome].prever(self.X_teste)
                tempo_inf_s = time.perf_counter() - t_inf_inicio
                latencia_media_ms = round((tempo_inf_s / n_amostras) * 1_000, 4)
                throughput = round(n_amostras / tempo_inf_s, 1) if tempo_inf_s > 0 else float("inf")

                metricas_completas: dict[str, Any] = {
                    **metricas_base,
                    "tempo_treino_s": tempo_treino_s,
                    "latencia_media_ms": latencia_media_ms,
                    "throughput_amostras_s": throughput,
                    "n_amostras_treino": len(self.X_treino),
                    "n_amostras_teste": int(n_amostras),
                }
                # matriz_confusao é serializável mas pode ser grande — mantida
                resultados[nome] = ResultadoBenchmark(
                    modelo_id=nome,
                    status="sucesso",
                    metricas=metricas_completas,
                )
                logger.info(
                    "[Fachada] '%s' — sucesso | acurácia=%.4f | latência=%.3fms",
                    nome, metricas_base["acuracia"], latencia_media_ms,
                )

            except Exception as exc:  # noqa: BLE001
                logger.error("[Fachada] Falha no benchmark de '%s': %s", nome, exc)
                resultados[nome] = ResultadoBenchmark(
                    modelo_id=nome,
                    status="falha",
                    erro=str(exc),
                )

        # ── Persistência em disco ─────────────────────────────────────────────
        self._persistir_benchmark(resultados, ts_inicio, dir_saida)

        logger.info(
            "[Fachada] Benchmark concluído — %d sucesso(s), %d falha(s).",
            sum(1 for r in resultados.values() if r.status == "sucesso"),
            sum(1 for r in resultados.values() if r.status == "falha"),
        )
        return resultados

    def _persistir_benchmark(
        self,
        resultados: dict[str, ResultadoBenchmark],
        ts_inicio: str,
        dir_saida: str | Path,
    ) -> None:
        """Serializa os resultados de benchmark para JSON estruturado em disco.

        Cria ``dir_saida`` automaticamente se não existir. Em caso de falha
        de I/O, emite ``logger.error`` sem propagar a exceção, garantindo que
        os dados em memória permaneçam acessíveis ao chamador.

        Args:
            resultados: Dicionário de resultados retornado por ``executar_benchmark``.
            ts_inicio: Timestamp ISO 8601 do início do benchmark.
            dir_saida: Diretório destino para o arquivo JSON.
        """
        payload: dict[str, Any] = {
            "schema_versao": "1.0",
            "timestamp_inicio": ts_inicio,
            "timestamp_fim": datetime.now(tz=timezone.utc).isoformat(),
            "dataset": "mnist_784",
            "n_modelos": len(resultados),
            "modelos": {nome: r.para_dict() for nome, r in resultados.items()},
        }

        try:
            caminho_dir = Path(dir_saida)
            caminho_dir.mkdir(parents=True, exist_ok=True)

            # Nome do arquivo: benchmark_<YYYYMMDD_HHMMSS>.json
            sufixo = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            arquivo = caminho_dir / f"benchmark_{sufixo}.json"

            with arquivo.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

            logger.info("[Fachada] Benchmark persistido em '%s'.", arquivo)

        except OSError as exc:
            logger.error(
                "[Fachada] Falha de I/O ao persistir benchmark em '%s': %s — "
                "resultados preservados apenas em memória.",
                dir_saida, exc,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Experimento MLflow
    # ──────────────────────────────────────────────────────────────────────────

    def executar_experimento(self, nome_modelo: str) -> dict[str, Any]:
        """Executa o ciclo completo de treino + avaliação com rastreio MLflow.

        Quando o MLflow não está instalado, executa normalmente sem rastreio.

        Args:
            nome_modelo: Chave canônica do modelo a executar.

        Returns:
            Dicionário de métricas retornado por ``avaliar_modelo()``, com
            ``tempo_treino_s`` adicionado.

        Raises:
            ValueError: Se ``nome_modelo`` não existir na fábrica.
        """
        self._garantir_dados()

        t0 = time.perf_counter()
        self.treinar_modelo(nome_modelo)
        tempo_treino_s = time.perf_counter() - t0

        metricas = self.avaliar_modelo(nome_modelo)
        metricas["tempo_treino_s"] = round(tempo_treino_s, 4)

        if _MLFLOW_OK:
            with mlflow.start_run(run_name=f"Exp_{nome_modelo}"):
                mlflow.log_param("modelo", nome_modelo)
                mlflow.log_param("dataset", "mnist_784")
                mlflow.log_metrics(
                    {
                        "acuracia": metricas["acuracia"],
                        "precisao": metricas["precisao"],
                        "recall": metricas["recall"],
                        "f1": metricas["f1"],
                        "tempo_treino_s": tempo_treino_s,
                    }
                )
        else:
            logger.warning(
                "[Fachada] MLflow não instalado — experimento '%s' não rastreado.",
                nome_modelo,
            )

        return metricas

    # ──────────────────────────────────────────────────────────────────────────
    # Estatísticas descritivas
    # ──────────────────────────────────────────────────────────────────────────

    def obter_estatisticas_dados(
        self, tipo: str = "treino"
    ) -> dict[str, float]:
        """Calcula estatísticas descritivas sobre a partição especificada.

        Args:
            tipo: ``"treino"`` (padrão) ou ``"teste"``.

        Returns:
            Dicionário com métricas estatísticas (média, desvio, min, max etc.)
            conforme implementado em ``CalculadorEstatistico``.

        Raises:
            ValueError: Se os dados não foram inicializados ou ``tipo`` inválido.
        """
        self._garantir_dados()
        dados = self.X_treino if tipo == "treino" else self.X_teste
        if dados is None:
            raise ValueError(
                f"Partição '{tipo}' não disponível. Chame inicializar_dados() primeiro."
            )
        calc = CalculadorEstatistico()
        return calc.estatisticas_descritivas(dados)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitários de consulta
    # ──────────────────────────────────────────────────────────────────────────

    def listar_modelos_treinados(self) -> list[str]:
        """Retorna os nomes dos modelos atualmente treinados e em memória.

        Returns:
            Lista de chaves de modelo presentes em ``self.modelos``.
        """
        return list(self.modelos.keys())

    def dados_inicializados(self) -> bool:
        """Indica se o dataset já foi carregado e pré-processado.

        Returns:
            ``True`` se ``X_treino`` está populado, ``False`` caso contrário.
        """
        return self.X_treino is not None
