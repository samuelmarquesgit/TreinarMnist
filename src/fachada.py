from src.carregador_dados import carregar_dados_mnist
from src.pre_processamento import pre_processar_dados
from src.modelos.fabrica_modelos import FabricaModelos
from src.avaliacao_metricas import calcular_metricas
from src.analise_estatistica import CalculadorEstatistico

class FachadaPipelineIA:
    def __init__(self):
        self.X, self.y = None, None
        self.modelos = {}
        self.scaler = None
    
    def inicializar_dados(self):
        self.X, self.y = carregar_dados_mnist()
        self.X_treino, self.X_teste, self.y_treino, self.y_teste, self.scaler = pre_processar_dados(self.X, self.y)
    
    def treinar_modelo(self, nome_modelo):
        if not self.X_treino is not None:
            self.inicializar_dados()
            
        modelo = FabricaModelos.criar_modelo(nome_modelo)
        modelo.treinar(self.X_treino, self.y_treino)
        self.modelos[nome_modelo] = modelo
        return modelo
    
    def avaliar_modelo(self, nome_modelo):
        if nome_modelo not in self.modelos:
            raise ValueError(f"Modelo {nome_modelo} não foi treinado.")
        
        modelo = self.modelos[nome_modelo]
        y_previsto = modelo.prever(self.X_teste)
        return calcular_metricas(self.y_teste, y_previsto)
        
    def obter_estatisticas_dados(self, tipo='treino'):
        calc = CalculadorEstatistico()
        dados = self.X_treino if tipo == 'treino' else self.X_teste
        return calc.estatisticas_descritivas(dados)
