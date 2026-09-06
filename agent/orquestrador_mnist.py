"""Agente autônomo responsável pela orquestração do ciclo completo de vida do pipeline MNIST.

Implementa o padrão Agente sequencial com controle de estado e tratamento de erros por etapa.
Delega toda a lógica de negócio à ``FachadaPipelineIA``, preservando o princípio de
responsabilidade única.

Nota de logging:
    Biblioteca interna — nunca chama ``logging.basicConfig()``.
    Usa apenas ``logger = logging.getLogger(__name__)`` para emitir mensagens
    rastreáveis sem interferir no pipeline de logs do sistema pai.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OrquestradorMNISTAgente:
    """Agente que executa e supervisiona as etapas do pipeline de machine learning.

    Coordena as cinco fases do projeto MNIST em sequência determinística,
    atualizando ``estado_execucao`` a cada etapa concluída e acumulando erros
    sem interromper as etapas subsequentes (política de resiliência parcial).

    Attributes:
        configuracoes: Parâmetros de configuração do agente (ex: modelos_selecionados).
        estado_execucao: Dicionário de estado mutable atualizado em tempo real.

    Example:
        >>> agente = OrquestradorMNISTAgente()
        >>> resultado = agente.executar_plano_completo()
        >>> print(resultado["status"])  # "SUCESSO", "PARCIAL" ou "FALHA"
    """

    def __init__(self, configuracoes: dict[str, Any] | None = None) -> None:
        self.configuracoes: dict[str, Any] = configuracoes or {}
        self.estado_execucao: dict[str, Any] = {
            "eda_concluido": False,
            "dados_preparados": False,
            "modelos_treinados": [],
            "avaliacao_concluida": False,
            "ood_concluido": False,
        }

    # ── Pipeline completo ─────────────────────────────────────────────────────

    def executar_plano_completo(self) -> dict[str, Any]:
        """Executa a sequência ponta a ponta do pipeline MNIST.

        Fases executadas em ordem:
        1. Inicialização de dados e EDA.
        2. Treinamento de todos os modelos disponíveis na fábrica.
        3. Avaliação individual de cada modelo treinado.
        4. Benchmark comparativo com persistência JSON.

        Erros em uma fase são registrados em ``erros`` sem interromper as demais.

        Returns:
            Dicionário com campos:
                - ``status``: "SUCESSO", "PARCIAL" ou "FALHA".
                - ``mensagem``: Resumo textual da execução.
                - ``estado``: Cópia final de ``estado_execucao``.
                - ``erros``: Lista de strings descrevendo falhas ocorridas.
                - ``metricas``: Dicionário ``{nome_modelo: Dict[str, float]}``
                  com as métricas de cada modelo avaliado com sucesso.
        """
        # Importações lazy para evitar dependência circular no nível de módulo
        from src.fachada import FachadaPipelineIA
        from src.modelos.fabrica_modelos import FabricaModelos

        erros: list[str] = []
        metricas: dict[str, Any] = {}
        fachada = FachadaPipelineIA()

        # Fase 1: Dados e EDA — falha aqui encerra o pipeline
        if not self._fase1_dados(fachada, erros):
            return self._montar_resultado(erros, metricas, "FALHA")

        # Fase 2: Treinamento
        modelos_selecionados: list[str] = self.configuracoes.get(
            "modelos_selecionados", FabricaModelos.listar_disponiveis()
        )
        self._fase2_treino(fachada, modelos_selecionados, erros)

        # Fase 3: Avaliação individual
        self._fase3_avaliacao(fachada, metricas, erros)

        # Fase 4: Benchmark comparativo
        self._fase4_benchmark(fachada, metricas, erros)

        # Status final
        modelos_avaliados = list(metricas.keys())
        if not erros:
            status = "SUCESSO"
        elif modelos_avaliados:
            status = "PARCIAL"
        else:
            status = "FALHA"

        return self._montar_resultado(erros, metricas, status)

    # ── Fases privadas ────────────────────────────────────────────────────────

    def _fase1_dados(self, fachada: Any, erros: list[str]) -> bool:
        """Fase 1: Inicialização dos dados e EDA. Retorna False em caso de falha."""
        logger.info("[Orquestrador] Fase 1 — Inicialização de dados e EDA.")
        try:
            fachada.inicializar_dados()
            self.estado_execucao["eda_concluido"] = True
            self.estado_execucao["dados_preparados"] = True
            logger.info("[Orquestrador] Dados inicializados com sucesso.")
            return True
        except Exception as exc:
            msg = f"Fase 1 (Dados/EDA): {exc}"
            erros.append(msg)
            logger.error("[Orquestrador] %s", msg)
            return False

    def _fase2_treino(self, fachada: Any, modelos: list[str], erros: list[str]) -> None:
        """Fase 2: Treinamento dos modelos selecionados."""
        logger.info("[Orquestrador] Fase 2 — Treinando %d modelo(s).", len(modelos))
        for nome_modelo in modelos:
            try:
                fachada.treinar_modelo(nome_modelo)
                self.estado_execucao["modelos_treinados"].append(nome_modelo)
                logger.info("[Orquestrador] Modelo '%s' treinado.", nome_modelo)
            except Exception as exc:
                msg = f"Fase 2 (Treino/{nome_modelo}): {exc}"
                erros.append(msg)
                logger.warning("[Orquestrador] %s", msg)

    def _fase3_avaliacao(self, fachada: Any, metricas: dict[str, Any], erros: list[str]) -> None:
        """Fase 3: Avaliação individual de cada modelo treinado."""
        logger.info("[Orquestrador] Fase 3 — Avaliando modelos treinados.")
        for nome_modelo in list(self.estado_execucao["modelos_treinados"]):
            try:
                resultado = fachada.avaliar_modelo(nome_modelo)
                metricas[nome_modelo] = resultado
                logger.info(
                    "[Orquestrador] '%s' avaliado — acurácia=%.4f.",
                    nome_modelo,
                    resultado.get("acuracia", float("nan")),
                )
            except Exception as exc:
                msg = f"Fase 3 (Avaliação/{nome_modelo}): {exc}"
                erros.append(msg)
                logger.warning("[Orquestrador] %s", msg)
        if metricas:
            self.estado_execucao["avaliacao_concluida"] = True

    def _fase4_benchmark(self, fachada: Any, metricas: dict[str, Any], erros: list[str]) -> None:
        """Fase 4: Benchmark comparativo entre os modelos avaliados."""
        logger.info("[Orquestrador] Fase 4 — Benchmark comparativo.")
        modelos_avaliados = list(metricas.keys())
        if not modelos_avaliados:
            erros.append("Fase 4 ignorada: nenhum modelo avaliado com sucesso.")
            return
        try:
            fachada.executar_benchmark(modelos_avaliados)
            logger.info(
                "[Orquestrador] Benchmark concluído para %d modelo(s).",
                len(modelos_avaliados),
            )
        except Exception as exc:
            msg = f"Fase 4 (Benchmark): {exc}"
            erros.append(msg)
            logger.warning("[Orquestrador] %s", msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _montar_resultado(
        self,
        erros: list[str],
        metricas: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        """Monta o dicionário de retorno padronizado.

        Args:
            erros: Lista de mensagens de erro acumuladas.
            metricas: Métricas por modelo avaliado com sucesso.
            status: ``"SUCESSO"``, ``"PARCIAL"`` ou ``"FALHA"``.

        Returns:
            Dicionário com chaves ``status``, ``mensagem``, ``estado``,
            ``erros`` e ``metricas``.
        """
        if status == "SUCESSO":
            mensagem = (
                f"Plano executado com sucesso: "
                f"{len(self.estado_execucao['modelos_treinados'])} modelo(s) treinado(s)."
            )
        elif status == "PARCIAL":
            mensagem = (
                f"Execução parcial: {len(erros)} erro(s) registrado(s). "
                f"{len(metricas)} modelo(s) avaliado(s)."
            )
        else:
            mensagem = f"Execução falhou: {len(erros)} erro(s) crítico(s)."

        logger.info("[Orquestrador] Resultado: %s — %s", status, mensagem)
        return {
            "status": status,
            "mensagem": mensagem,
            "estado": dict(self.estado_execucao),
            "erros": erros,
            "metricas": metricas,
        }
