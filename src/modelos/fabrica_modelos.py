import logging
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from src.modelos.base_modelo import ModeloAbstratoIA

# Configuração de log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModeloSklearn(ModeloAbstratoIA):
    """
    Wrapper padrão para integrar modelos do Scikit-Learn à arquitetura da Plataforma MNIST.
    Garante que todos os algoritmos respeitem a interface base de fit/predict com logs.
    """

    def __init__(self, modelo: Any, nome_log: str = "Desconhecido"):
        self.modelo = modelo
        self.nome_log = nome_log

    def treinar(self, X_treino: Any, y_treino: Any) -> None:
        logger.info(
            f"[{self.nome_log}] Iniciando treinamento com {len(X_treino)} amostras...")
        self.modelo.fit(X_treino, y_treino)
        logger.info(f"[{self.nome_log}] Treinamento concluido com sucesso.")

    def prever(self, X_teste: Any) -> Any:
        logger.info(
            f"[{self.nome_log}] Realizando inferencia para {len(X_teste)} amostras...")
        previsoes = self.modelo.predict(X_teste)
        logger.info(f"[{self.nome_log}] Inferencia concluida.")
        return previsoes

    def prever_probabilidades(self, X_teste: Any) -> Any:
        logger.info(
            f"[{self.nome_log}] Realizando previsão de probabilidades para {len(X_teste)} amostras...")
        if hasattr(self.modelo, "predict_proba"):
            probs = self.modelo.predict_proba(X_teste)
            logger.info(f"[{self.nome_log}] Previsão de probabilidades concluída.")
            return probs
        else:
            raise NotImplementedError(f"O modelo {self.nome_log} não suporta previsão de probabilidades nativamente.")


class FabricaModelos:
    """
    Implementa o padrao Factory Method para instanciar algoritmos de Machine Learning.
    Fornece um ponto centralizado para injecao de hiperparametros padrao.
    """

    _REGISTRO_MODELOS = {
        'RegressaoLogistica': lambda: LogisticRegression(max_iter=500, random_state=42),
        'ArvoreDecisao': lambda: DecisionTreeClassifier(random_state=42),
        'FlorestaAleatoria': lambda: RandomForestClassifier(n_estimators=50, random_state=42),
        'ImpulsionamentoGradiente': lambda: GradientBoostingClassifier(n_estimators=50, random_state=42),
        'SVM': lambda: SVC(kernel='rbf', random_state=42, probability=True),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': lambda: GaussianNB(),
        'PerceptronMulticamadas': lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    @staticmethod
    def criar_modelo(nome_modelo: str) -> ModeloAbstratoIA:
        """
        Instancia o modelo solicitado encapsulado na interface base.

        Args:
            nome_modelo (str): O nome do modelo (ex: 'FlorestaAleatoria').

        Returns:
            ModeloAbstratoIA: A instancia configurada e pronta para treino.

        Raises:
            ValueError: Caso o modelo nao esteja registrado na fabrica.
        """
        if nome_modelo == 'VisionTransformer':
            from src.modelos.vision_transformer import ModeloViT
            logger.info(f"Fabrica instanciando novo modelo: {nome_modelo}")
            return ModeloViT(nome_log=nome_modelo)

        construtor = FabricaModelos._REGISTRO_MODELOS.get(nome_modelo)

        if construtor is None:
            logger.error(
                f"Tentativa de instanciar modelo inexistente: {nome_modelo}")
            raise ValueError(
                f"Modelo '{nome_modelo}' desconhecido. "
                f"Modelos suportados: {list(FabricaModelos._REGISTRO_MODELOS.keys()) + ['VisionTransformer']}")

        logger.info(f"Fabrica instanciando novo modelo: {nome_modelo}")
        return ModeloSklearn(construtor(), nome_log=nome_modelo)
