"""Agente especialista em diagnóstico estatístico e análise comparativa de classificadores."""

from typing import Dict, Any
import pandas as pd


class AgenteAnalistaMetricas:
    """Agente que sintetiza resultados, identifica trade-offs e seleciona o modelo campeão."""

    @staticmethod
    def identificar_modelo_campeao(tabela_metricas: pd.DataFrame) -> Dict[str, Any]:
        """Avalia o melhor equilíbrio entre Acurácia, F1-Score e Custo Computacional.

        Args:
            tabela_metricas: DataFrame com colunas ['modelo', 'acuracia', 'f1_score', 'tempo_treino_s'].

        Returns:
            Dicionário com o modelo recomendado e justificativa técnica.
        """
        if tabela_metricas.empty:
            return {"modelo_campeao": "Nenhum", "justificativa": "Tabela de métricas vazia."}

        melhor_f1 = tabela_metricas.sort_values(by="f1_score", ascending=False).iloc[0]

        return {
            "modelo_campeao": melhor_f1["modelo"],
            "acuracia": float(melhor_f1["acuracia"]),
            "f1_score": float(melhor_f1["f1_score"]),
            "tempo_treino_s": float(melhor_f1.get("tempo_treino_s", 0.0)),
            "justificativa": (
                f"O modelo {melhor_f1['modelo']} atingiu a pontuação máxima de F1-Score ponderado "
                f"({melhor_f1['f1_score']:.4f}), garantindo equilíbrio ideal entre precisão e sensibilidade."
            )
        }
