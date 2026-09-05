import os
import json
from pymongo import MongoClient

class ConexaoMongoDB:
    def __init__(self):
        uri = os.getenv('MONGO_URI', None)
        self.usar_local = not uri
        if not self.usar_local:
            self.client = MongoClient(uri)
            self.db = self.client['treinarmnist']
            self.colecao = self.db['matrizes_confusao']
    
    def salvar_artefato(self, nome, dados):
        if self.usar_local:
            with open(f'reports/{nome}.json', 'w', encoding='utf-8') as f:
                json.dump(dados, f)
        else:
            self.colecao.insert_one({'nome': nome, 'dados': dados})
