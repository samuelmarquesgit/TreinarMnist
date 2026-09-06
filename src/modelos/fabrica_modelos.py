import logging
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import config_modelos
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
        'RegressaoLogistica': lambda: LogisticRegression(**config_modelos.regressao_logistica.model_dump()),
        'ArvoreDecisao': lambda: DecisionTreeClassifier(**config_modelos.arvore_decisao.model_dump()),
        'FlorestaAleatoria': lambda: RandomForestClassifier(**config_modelos.floresta_aleatoria.model_dump()),
        'ImpulsionamentoGradiente': lambda: GradientBoostingClassifier(
            **config_modelos.impulsionamento_gradiente.model_dump()),
        'SVM': lambda: SVC(**config_modelos.svm.model_dump()),
        'KNN': lambda: KNeighborsClassifier(**config_modelos.knn.model_dump()),
        'NaiveBayes': lambda: GaussianNB(**config_modelos.naive_bayes.model_dump()),
        'PerceptronMulticamadas': lambda: MLPClassifier(
            hidden_layer_sizes=tuple(config_modelos.perceptron_multicamadas.hidden_layer_sizes),
            max_iter=config_modelos.perceptron_multicamadas.max_iter,
            random_state=config_modelos.perceptron_multicamadas.random_state
        )
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
            return ModeloViT(
                nome_log=nome_modelo,
                epocas=config_modelos.vision_transformer.epocas,
                batch_size=config_modelos.vision_transformer.batch_size
            )

        construtor = FabricaModelos._REGISTRO_MODELOS.get(nome_modelo)

        if construtor is None:
            logger.error(
                f"Tentativa de instanciar modelo inexistente: {nome_modelo}")
            raise ValueError(
                f"Modelo '{nome_modelo}' desconhecido. "
                f"Modelos suportados: {list(FabricaModelos._REGISTRO_MODELOS.keys()) + ['VisionTransformer']}")

        logger.info(f"Fabrica instanciando novo modelo: {nome_modelo}")
        estimador_base = construtor()
        
        # Aplica Validação Cruzada (cv=5) e Calibração Isotônica (exceto Redes Neurais e Reg Logística pura)
        if nome_modelo not in ["PerceptronMulticamadas", "RegressaoLogistica"]:
            from sklearn.calibration import CalibratedClassifierCV
            logger.info(f"[{nome_modelo}] Aplicando K-Fold (cv=5) e Calibração Isotônica...")
            # method='isotonic' é rigoroso para multiclasse e requer amostras suficientes por fold
            # cv=5 garante validação cruzada real e não enviesada
            estimador_base = CalibratedClassifierCV(estimador_base, method='isotonic', cv=5)

        return ModeloSklearn(estimador_base, nome_log=nome_modelo)
