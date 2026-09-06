"""Fachada do pipeline de IA da plataforma MNIST (Padrão GoF — Facade).

Centraliza a orquestração de todas as etapas do pipeline (dados, treinamento,
avaliação, benchmark e estatísticas) para CLI, Servidor MCP e Frontend Web.

Nota de logging:
    Biblioteca interna — nunca chama ``logging.basicConfig()``.
    Usa apenas ``logger = logging.getLogger(__name__)`` para emitir mensagens
    rastreáveis sem interferir no pipeline de logs do sistema pai.

Nota MLflow:
    MLflow é uma dependência opcional. Quando não instalado, o rastreamento
    de experimentos é silenciosamente desativado via ``_MLFLOW_OK``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.avaliacao_metricas import calcular_metricas
from src.carregador_dados import carregar_dados_mnist
from src.modelos.base_modelo import ModeloAbstratoIA
from src.modelos.fabrica_modelos import FabricaModelos
from src.pre_processamento import pre_processar_dados
from src.utilitarios.excecoes import ModeloNaoTreinadoError
from src.analise_estatistica import CalculadorEstatistico

# MLflow é opcional — não deve impedir a inicialização do módulo
try:
    import mlflow
    _MLFLOW_OK: bool = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_OK = False

logger = logging.getLogger(__name__)

# Diretório padrão para persistência de benchmarks
_DIR_BENCHMARKS: str = os.path.join("artifacts", "benchmarks")


# ──────────────────────────────────────────────────────────────────────────────
# Estrutura de resultado de benchmark
# ──────────────────────────────────────────────────────────────────────────────


class ResultadoBenchmark:
    """Encapsula o resultado de um benchmark de modelo individual.

    Attributes:
        modelo_id: Chave canônica do modelo.
        status: ``"ok"`` ou ``"erro"``.
        metricas: Dicionário de métricas (acuracia, f1, etc.) quando ok.
        erro: Mensagem de erro quando status == "erro".
        latencia_ms: Latência média de inferência por amostra em milissegundos.
        throughput: Amostras processadas por segundo.
    """

    def __init__(
        self,
        modelo_id: str,
        status: str,
        metricas: Optional[Dict[str, Any]] = None,
        erro: Optional[str] = None,
        latencia_ms: float = 0.0,
        throughput: float = 0.0,
    ) -> None:
        self.modelo_id = modelo_id
        self.status = status
        self.metricas = metricas or {}
        self.erro = erro
        self.latencia_ms = latencia_ms
        self.throughput = throughput

    def para_dict(self) -> Dict[str, Any]:
        """Serializa o resultado para dicionário JSON-compatível."""
        return {
            "modelo_id": self.modelo_id,
            "status": self.status,
            "metricas": self.metricas,
            "erro": self.erro,
            "latencia_ms": round(self.latencia_ms, 4),
            "throughput_amostras_por_segundo": round(self.throughput, 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Fachada principal
# ──────────────────────────────────────────────────────────────────────────────


class FachadaPipelineIA:
    """Fachada central do pipeline MNIST (GoF — Facade).

    Expõe uma API unificada e simplificada para CLI, MCP e Frontend,
    sem vazar detalhes de implementação (sklearn, PyTorch, bancos de dados)
    para as camadas superiores.

    Attributes:
        X: Dataset completo de features (shape N×784).
        y: Rótulos correspondentes (shape N,).
        X_treino / X_teste: Partições pós-split normalizadas.
        y_treino / y_teste: Rótulos das partições.
        modelos: Registro de modelos treinados {nome: instância}.
        scaler: MinMaxScaler ajustado apenas no treino.

    Example:
        >>> fachada = FachadaPipelineIA()
        >>> fachada.inicializar_dados()
        >>> fachada.treinar_modelo("RegressaoLogistica")
        >>> metricas = fachada.avaliar_modelo("RegressaoLogistica")
        >>> print(metricas["acuracia"])
    """

    def __init__(self) -> None:
        self.X: Optional[NDArray[np.float32]] = None
        self.y: Optional[NDArray[np.int32]] = None
        self.X_treino: Optional[NDArray[np.float32]] = None
        self.X_teste: Optional[NDArray[np.float32]] = None
        self.y_treino: Optional[NDArray[np.int32]] = None
        self.y_teste: Optional[NDArray[np.int32]] = None
        self.modelos: Dict[str, ModeloAbstratoIA] = {}
        self.scaler: Any = None

        if _MLFLOW_OK:
            try:
                mlflow.set_experiment("Treinamento_MNIST")
            except Exception as exc:  # servidor MLflow inacessível
                logger.warning("MLflow: não foi possível definir experimento — %s", exc)

    # ── Dados ─────────────────────────────────────────────────────────────────

    def inicializar_dados(self) -> None:
        """Carrega o MNIST e aplica split estratificado + normalização MinMax."""
        self.X, self.y = carregar_dados_mnist()
        (
            self.X_treino,
            self.X_teste,
            self.y_treino,
            self.y_teste,
            self.scaler,
        ) = pre_processar_dados(self.X, self.y)
        logger.info(
            "Dados inicializados — treino: %d amostras | teste: %d amostras.",
            len(self.X_treino),
            len(self.X_teste),
        )

    def _garantir_dados(self) -> None:
        """Garante que os dados estejam carregados; inicializa se necessário."""
        if self.X_treino is None:
            self.inicializar_dados()

    def dados_inicializados(self) -> bool:
        """Retorna ``True`` se o dataset já foi carregado e particionado."""
        return self.X_treino is not None

    # ── Treinamento e avaliação ───────────────────────────────────────────────

    def treinar_modelo(self, nome_modelo: str) -> ModeloAbstratoIA:
        """Instancia e treina o modelo solicitado.

        Args:
            nome_modelo: Chave canônica aceita pela ``FabricaModelos``.

        Returns:
            Instância de ``ModeloAbstratoIA`` já ajustada.
        """
        self._garantir_dados()
        modelo = FabricaModelos.criar_modelo(nome_modelo)
        modelo.treinar(self.X_treino, self.y_treino)  # type: ignore[arg-type]
        self.modelos[nome_modelo] = modelo
        logger.info("Modelo '%s' treinado com sucesso.", nome_modelo)
        return modelo

    def avaliar_modelo(self, nome_modelo: str) -> Dict[str, Any]:
        """Avalia o modelo treinado sobre o conjunto de teste.

        Args:
            nome_modelo: Chave de um modelo já treinado via ``treinar_modelo()``.

        Returns:
            Dicionário com chaves: ``acuracia``, ``precisao``, ``recall``,
            ``f1``, ``matriz_confusao``.

        Raises:
            ModeloNaoTreinadoError: Se o modelo não tiver sido treinado.
        """
        if nome_modelo not in self.modelos:
            raise ModeloNaoTreinadoError(
                f"Modelo '{nome_modelo}' não foi treinado. "
                "Chame treinar_modelo() antes de avaliar_modelo()."
            )
        modelo = self.modelos[nome_modelo]
        y_previsto = modelo.prever(self.X_teste)  # type: ignore[arg-type]
        return calcular_metricas(self.y_teste, y_previsto)  # type: ignore[arg-type]

    def prever_probabilidades(
        self,
        nome_modelo: str,
        X_entrada: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Retorna distribuição de probabilidade por classe.

        Args:
            nome_modelo: Chave de um modelo já treinado.
            X_entrada: Matriz de features de shape ``(N, 784)``.

        Returns:
            Array de shape ``(N, n_classes)`` com probabilidades ``[0, 1]``.

        Raises:
            ModeloNaoTreinadoError: Se o modelo não tiver sido treinado.
        """
        if nome_modelo not in self.modelos:
            raise ModeloNaoTreinadoError(
                f"Modelo '{nome_modelo}' não foi treinado."
            )
        return self.modelos[nome_modelo].prever_probabilidades(X_entrada)

    # ── Experimento MLflow ────────────────────────────────────────────────────

    def executar_experimento(self, nome_modelo: str) -> Dict[str, Any]:
        """Treina, avalia e registra o ciclo completo no MLflow (opcional).

        As chaves de métricas no MLflow espelham exatamente as chaves pt-BR
        retornadas por ``calcular_metricas()``.

        Args:
            nome_modelo: Chave canônica do modelo.

        Returns:
            Dicionário de métricas com ``tempo_treino_segundos`` adicional.
        """
        self._garantir_dados()

        inicio = time.perf_counter()
        self.treinar_modelo(nome_modelo)
        tempo_treino = time.perf_counter() - inicio

        metricas = self.avaliar_modelo(nome_modelo)
        metricas["tempo_treino_segundos"] = round(tempo_treino, 4)

        if _MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f"Exp_{nome_modelo}"):
                    mlflow.log_param("modelo", nome_modelo)
                    mlflow.log_param("dataset", "mnist_784")
                    mlflow.log_metrics({
                        # Chaves pt-BR idênticas às retornadas por calcular_metricas()
                        "acuracia": metricas["acuracia"],
                        "precisao": metricas["precisao"],
                        "recall": metricas["recall"],
                        "f1": metricas["f1"],
                        "tempo_treino": tempo_treino,
                    })
            except Exception as exc:
                logger.warning(
                    "MLflow: falha ao registrar experimento de '%s' — %s",
                    nome_modelo,
                    exc,
                )
        else:
            logger.debug("MLflow indisponível — experimento não rastreado.")

        return metricas

    # ── Benchmark comparativo ─────────────────────────────────────────────────

    def executar_benchmark(
        self,
        modelos_ids: Optional[List[str]] = None,
        dir_saida: str = _DIR_BENCHMARKS,
    ) -> Dict[str, ResultadoBenchmark]:
        """Executa benchmark comparativo sobre uma lista de modelos.

        Para cada modelo: treina (se necessário), avalia métricas, mede
        latência média de inferência e throughput. Persiste os resultados em
        ``dir_saida/benchmark_<timestamp>.json``.

        Args:
            modelos_ids: Lista de chaves canônicas. Se ``None``, usa todos os
                modelos disponíveis na fábrica.
            dir_saida: Diretório de saída para o JSON de resultados.

        Returns:
            Dicionário ``{nome_modelo: ResultadoBenchmark}``.
        """
        self._garantir_dados()
        ids = modelos_ids or FabricaModelos.listar_disponiveis()
        resultados: Dict[str, ResultadoBenchmark] = {}
        ts_inicio = datetime.now(tz=timezone.utc).isoformat()

        logger.info("[Benchmark] Iniciando para %d modelo(s).", len(ids))

        for nome in ids:
            logger.info("[Benchmark] Processando '%s'…", nome)
            try:
                # Treina se ainda não estiver no registro
                if nome not in self.modelos:
                    self.treinar_modelo(nome)

                metricas = self.avaliar_modelo(nome)

                # Medição de latência sobre o conjunto de teste
                n_amostras = len(self.X_teste)  # type: ignore[arg-type]
                t0 = time.perf_counter()
                self.modelos[nome].prever(self.X_teste)  # type: ignore[arg-type]
                elapsed = time.perf_counter() - t0

                latencia_ms = (elapsed / n_amostras) * 1000 if n_amostras > 0 else 0.0
                throughput = n_amostras / elapsed if elapsed > 0 else 0.0

                resultados[nome] = ResultadoBenchmark(
                    modelo_id=nome,
                    status="ok",
                    metricas=metricas,
                    latencia_ms=latencia_ms,
                    throughput=throughput,
                )
                logger.info(
                    "[Benchmark] '%s' — acurácia=%.4f | latência=%.2f ms/amostra.",
                    nome,
                    metricas.get("acuracia", 0.0),
                    latencia_ms,
                )
            except Exception as exc:
                msg = str(exc)
                resultados[nome] = ResultadoBenchmark(
                    modelo_id=nome, status="erro", erro=msg
                )
                logger.warning("[Benchmark] '%s' falhou: %s", nome, msg)

        self._persistir_benchmark(resultados, ts_inicio, dir_saida)
        return resultados

    def _persistir_benchmark(
        self,
        resultados: Dict[str, ResultadoBenchmark],
        ts_inicio: str,
        dir_saida: str,
    ) -> None:
        """Persiste resultados do benchmark em arquivo JSON.

        Args:
            resultados: Mapa de resultados por modelo.
            ts_inicio: Timestamp ISO 8601 de início do benchmark.
            dir_saida: Diretório de saída.
        """
        try:
            os.makedirs(dir_saida, exist_ok=True)
            ts_arquivo = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            caminho = os.path.join(dir_saida, f"benchmark_{ts_arquivo}.json")
            payload = {
                "ts_inicio": ts_inicio,
                "ts_fim": datetime.now(tz=timezone.utc).isoformat(),
                "n_modelos": len(resultados),
                "resultados": {k: v.para_dict() for k, v in resultados.items()},
            }
            with open(caminho, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info("[Benchmark] Resultados persistidos em '%s'.", caminho)
        except OSError as exc:
            logger.error(
                "[Benchmark] Falha ao persistir resultados (dados preservados em memória): %s",
                exc,
            )

    # ── Utilitários ───────────────────────────────────────────────────────────

    def listar_modelos_treinados(self) -> List[str]:
        """Retorna os nomes dos modelos já treinados nesta sessão."""
        return list(self.modelos.keys())

    def obter_estatisticas_dados(self, tipo: str = "treino") -> Dict[str, float]:
        """Calcula estatísticas descritivas da partição solicitada.

        Args:
            tipo: ``"treino"`` (padrão) ou ``"teste"``.

        Returns:
            Dicionário com media, desvio_padrao, minimo, maximo, etc.

        Raises:
            ValueError: Se os dados não tiverem sido inicializados.
        """
        self._garantir_dados()
        dados = self.X_treino if tipo == "treino" else self.X_teste
        calc = CalculadorEstatistico()
        return calc.estatisticas_descritivas(dados)  # type: ignore[arg-type]
