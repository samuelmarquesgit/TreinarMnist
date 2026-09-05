from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from src.modelos.base_modelo import ModeloAbstratoIA

class ModeloSklearn(ModeloAbstratoIA):
    def __init__(self, modelo):
        self.modelo = modelo
        
    def treinar(self, X_treino, y_treino):
        self.modelo.fit(X_treino, y_treino)
        
    def prever(self, X_teste):
        return self.modelo.predict(X_teste)

class FabricaModelos:
    @staticmethod
    def criar_modelo(nome_modelo):
        if nome_modelo == 'RegressaoLogistica':
            return ModeloSklearn(LogisticRegression(max_iter=500, random_state=42))
        elif nome_modelo == 'ArvoreDecisao':
            return ModeloSklearn(DecisionTreeClassifier(random_state=42))
        elif nome_modelo == 'FlorestaAleatoria':
            return ModeloSklearn(RandomForestClassifier(n_estimators=50, random_state=42))
        elif nome_modelo == 'ImpulsionamentoGradiente':
            return ModeloSklearn(GradientBoostingClassifier(n_estimators=50, random_state=42))
        elif nome_modelo == 'SVM':
            return ModeloSklearn(SVC(kernel='rbf', random_state=42))
        elif nome_modelo == 'KNN':
            return ModeloSklearn(KNeighborsClassifier(n_neighbors=5))
        elif nome_modelo == 'NaiveBayes':
            return ModeloSklearn(GaussianNB())
        else:
            raise ValueError(f'Modelo {nome_modelo} desconhecido')
