"""Gerenciador de conexão e persistência documental MongoDB para a plataforma MNIST.

Nota de logging:
    Biblioteca interna — nunca chama ``logging.basicConfig()``.
    Usa apenas ``logger = logging.getLogger(__name__)`` para emitir mensagens
    rastreáveis sem interferir no pipeline de logs do sistema pai.
"""

import json
import logging
import os
from typing import Any

try:
    from pymongo import MongoClient
    _PYMONGO_OK = True
except Exception:
    MongoClient = None  # type: ignore
    _PYMONGO_OK = False

logger = logging.getLogger(__name__)


class ConexaoMongoDB:
    """Gerenciador de conexão e persistência em formato documental.

    Especializado no armazenamento e recuperação de metadados complexos,
    como matrizes de confusão, predições detalhadas e relatórios OOD.

    Suporta modo offline automático (fallback JSON local) quando o servidor
    MongoDB não está disponível.

    Attributes:
        uri: URI de conexão MongoDB.
        usar_local: ``True`` quando operando em modo offline (JSON local).

    Example:
        >>> conn = ConexaoMongoDB()
        >>> conn.salvar_artefato("matriz_lr", {"dados": [[1, 0], [0, 1]]})
        >>> doc = conn.buscar_artefato("matriz_lr")
        >>> lista = conn.listar_colecao(limite=10)
    """

    def __init__(self, uri: str | None = None) -> None:
        self.uri: str | None = uri or os.getenv('MONGO_URI', None)
        self.usar_local: bool = not bool(self.uri)

        if not self.usar_local:
            try:
                self.client: Any = MongoClient(
                    self.uri, serverSelectionTimeoutMS=5000
                )
                self.db = self.client['treinarmnist']
                self.colecao = self.db['matrizes_confusao']
                # Força erro imediato caso URI seja inválida / servidor inacessível
                self.client.server_info()
                logger.info("Conexão com MongoDB Cloud inicializada.")
            except Exception as e:
                logger.warning(
                    "Falha ao conectar no MongoDB Cloud. Fazendo fallback local: %s", e
                )
                self.usar_local = True
        else:
            logger.info("MONGO_URI não definida. Utilizando armazenamento local JSON.")

    # ── Escrita ───────────────────────────────────────────────────────────────

    def salvar_artefato(self, nome: str, dados: dict[str, Any]) -> None:
        """Persiste os dados em coleção remota ou arquivo JSON local.

        Args:
            nome: Chave ou nome do arquivo para os dados.
            dados: Dicionário serializável em JSON.
        """
        if self.usar_local:
            os.makedirs('reports', exist_ok=True)
            caminho_arquivo = f'reports/{nome}.json'
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=4, ensure_ascii=False)
            logger.info(
                "Artefato '%s' salvo em formato JSON local: %s", nome, caminho_arquivo
            )
        else:
            documento = {'nome': nome, 'dados': dados}
            self.colecao.insert_one(documento)
            logger.info("Artefato '%s' salvo no MongoDB.", nome)

    # ── Leitura ───────────────────────────────────────────────────────────────

    def buscar_artefato(self, nome: str) -> dict[str, Any] | None:
        """Recupera um artefato pelo nome.

        Busca no MongoDB remoto ou no arquivo JSON local, dependendo do modo
        de operação atual.

        Args:
            nome: Chave do artefato a ser recuperado (mesmo valor usado em
                ``salvar_artefato``).

        Returns:
            Dicionário com os dados do artefato, ou ``None`` se não encontrado.
        """
        if self.usar_local:
            caminho_arquivo = f'reports/{nome}.json'
            if not os.path.exists(caminho_arquivo):
                logger.warning(
                    "Artefato local não encontrado: '%s'.", caminho_arquivo
                )
                return None
            try:
                with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                    dados: dict[str, Any] = json.load(f)
                logger.info(
                    "Artefato '%s' carregado do JSON local.", nome
                )
                return dados
            except (json.JSONDecodeError, OSError) as e:
                logger.error(
                    "Erro ao ler artefato local '%s': %s", caminho_arquivo, e
                )
                return None
        else:
            documento = self.colecao.find_one({'nome': nome}, {'_id': 0, 'dados': 1})
            if documento is None:
                logger.warning(
                    "Artefato '%s' não encontrado no MongoDB.", nome
                )
                return None
            logger.info("Artefato '%s' recuperado do MongoDB.", nome)
            doc: dict[str, Any] | None = documento.get('dados')
            return doc

    def listar_colecao(
        self,
        limite: int = 50,
        filtro: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Lista documentos da coleção (MongoDB) ou arquivos JSON locais.

        Args:
            limite: Número máximo de documentos retornados. Padrão: 50.
            filtro: Filtro MongoDB opcional (ignorado no modo local).

        Returns:
            Lista de dicionários com campos ``nome`` e ``dados`` de cada
            artefato encontrado. Retorna lista vazia em caso de erro.
        """
        resultados: list[dict[str, Any]] = []

        if self.usar_local:
            pasta = 'reports'
            if not os.path.isdir(pasta):
                logger.info("Pasta local '%s' não existe — nenhum artefato encontrado.", pasta)
                return resultados
            try:
                arquivos = [
                    f for f in os.listdir(pasta)
                    if f.endswith('.json')
                ][:limite]
            except OSError as e:
                logger.error("Erro ao listar pasta '%s': %s", pasta, e)
                return resultados

            for nome_arquivo in arquivos:
                caminho = os.path.join(pasta, nome_arquivo)
                try:
                    with open(caminho, 'r', encoding='utf-8') as f:
                        dados = json.load(f)
                    resultados.append({
                        'nome': nome_arquivo.removesuffix('.json'),
                        'dados': dados,
                    })
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Ignorando artefato ilegível '%s': %s", caminho, e
                    )
            logger.info(
                "Listagem local: %d artefato(s) encontrado(s).", len(resultados)
            )
        else:
            try:
                cursor = self.colecao.find(
                    filtro or {},
                    {'_id': 0, 'nome': 1, 'dados': 1},
                ).limit(limite)
                resultados = list(cursor)
                logger.info(
                    "Listagem MongoDB: %d documento(s) retornado(s).", len(resultados)
                )
            except Exception as e:
                logger.error("Erro ao listar coleção MongoDB: %s", e)

        return resultados
