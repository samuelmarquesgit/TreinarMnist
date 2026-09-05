import logging
from typing import Dict, Any, Type
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from src.modelos.base_modelo import ModeloAbstratoIA

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logger.info(f"[{self.nome_log}] Iniciando treinamento com {len(X_treino)} amostras...")
        self.modelo.fit(X_treino, y_treino)
        logger.info(f"[{self.nome_log}] Treinamento concluido com sucesso.")
        
    def prever(self, X_teste: Any) -> Any:
        logger.info(f"[{self.nome_log}] Realizando inferencia para {len(X_teste)} amostras...")
        previsoes = self.modelo.predict(X_teste)
        logger.info(f"[{self.nome_log}] Inferencia concluida.")
        return previsoes

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
        'SVM': lambda: SVC(kernel='rbf', random_state=42),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': lambda: GaussianNB()
    }

    @staticmethod
    def criar_modelo(nome_modelo: str) -> ModeloSklearn:
        """
        Instancia o modelo solicitado encapsulado na interface base.

        Args:
            nome_modelo (str): O nome do modelo (ex: 'FlorestaAleatoria').

        Returns:
            ModeloSklearn: A instancia configurada e pronta para treino.

        Raises:
            ValueError: Caso o modelo nao esteja registrado na fabrica.
        """
        construtor = FabricaModelos._REGISTRO_MODELOS.get(nome_modelo)
        
        if construtor is None:
            logger.error(f"Tentativa de instanciar modelo inexistente: {nome_modelo}")
            raise ValueError(f"Modelo '{nome_modelo}' desconhecido. Modelos suportados: {list(FabricaModelos._REGISTRO_MODELOS.keys())}")
            
        logger.info(f"Fabrica instanciando novo modelo: {nome_modelo}")
        return ModeloSklearn(construtor(), nome_log=nome_modelo)
