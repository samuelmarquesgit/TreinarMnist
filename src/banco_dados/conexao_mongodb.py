import os
import json
import logging
from typing import Dict, Any
try:
    from pymongo import MongoClient
    _PYMONGO_OK = True
except Exception:
    MongoClient = None  # type: ignore
    _PYMONGO_OK = False

logger = logging.getLogger(__name__)


class ConexaoMongoDB:
    """
    Gerenciador de conexao e persistencia em formato documental.
    Especializado no armazenamento de metadados complexos como matrizes de confusao.
    """

    def __init__(self, uri: str = None) -> None:
        self.uri = uri or os.getenv('MONGO_URI', None)
        self.usar_local = not bool(self.uri)

        if not self.usar_local:
            try:
                self.client = MongoClient(
                    self.uri, serverSelectionTimeoutMS=5000)
                self.db = self.client['treinarmnist']
                self.colecao = self.db['matrizes_confusao']
                # Tenta forcar erro caso URI seja invalida e servidor
                # inacessivel
                self.client.server_info()
                logger.info("Conexao com MongoDB Cloud inicializada.")
            except Exception as e:
                logger.warning(
                    f"Falha ao conectar no MongoDB Cloud. Fazendo fallback local: {str(e)}")
                self.usar_local = True
        else:
            logger.info(
                "MONGO_URI nao definida. Utilizando armazenamento local Json.")

    def salvar_artefato(self, nome: str, dados: Dict[str, Any]) -> None:
        """
        Persiste os dados em colecao remota ou arquivo JSON local.

        Args:
            nome (str): Chave ou nome do arquivo para os dados.
            dados (Dict[str, Any]): Dicionario seralizavel em JSON.
        """
        if self.usar_local:
            os.makedirs('reports', exist_ok=True)
            caminho_arquivo = f'reports/{nome}.json'
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=4)
            logger.info(
                f"Artefato {nome} salvo em formato JSON local: {caminho_arquivo}")
        else:
            documento = {'nome': nome, 'dados': dados}
            self.colecao.insert_one(documento)
            logger.info(f"Artefato {nome} salvo no MongoDB Atlas.")
