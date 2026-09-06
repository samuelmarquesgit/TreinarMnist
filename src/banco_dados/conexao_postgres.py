import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()


class Experimento(Base):
    """
    Mapeamento ORM (Object-Relational Mapping) da tabela de experimentos.
    Persiste resultados das avaliacoes estatisticas e de metricas de IA.
    """
    __tablename__ = 'experimentos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    modelo = Column(String, nullable=False)
    acuracia = Column(Float)
    tempo_treino = Column(Float)

    # datetime.utcnow() esta deprecado. Usamos timezone-aware nativo.
    data_execucao = Column(
        DateTime,
        default=lambda: datetime.now(
            timezone.utc))


class ConexaoPostgres:
    """
    Fornece o gerenciamento de sessoes do PostgreSQL via SQLAlchemy.
    Suporta fallback para SQLite local caso DATABASE_URL nao esteja disponivel.
    """

    def __init__(self, url: str = None) -> None:
        self.url = url or os.getenv(
            'DATABASE_URL', 'sqlite:///reports/banco_local.db')

        # Garante que a pasta reports exista para o sqlite local
        if self.url.startswith('sqlite:///reports/'):
            os.makedirs('reports', exist_ok=True)

        self.engine = create_engine(self.url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False)
        logger.info(f"Conexao com banco de dados inicializada: {self.url.split(chr(58))[0]}")

    @contextmanager
    def obter_sessao(self) -> Generator[Session, None, None]:
        """
        Gerenciador de contexto seguro para transacoes no banco.
        Yields:
            Session: Sessao ativa do SQLAlchemy.
        """
        sessao = self.SessionLocal()
        try:
            yield sessao
            sessao.commit()
        except Exception as e:
            sessao.rollback()
            logger.error(f"Erro em transacao de banco de dados: {str(e)}")
            raise
        finally:
            sessao.close()
