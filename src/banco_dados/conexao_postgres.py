import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base: Any = declarative_base()


class Experimento(Base):  # type: ignore[misc, valid-type]
    """
    Mapeamento ORM (Object-Relational Mapping) da tabela de experimentos.
    Persiste resultados das avaliacoes estatisticas e de metricas de IA.
    """

    __tablename__ = "experimentos"
    id = Column(Integer, primary_key=True, autoincrement=True)
    modelo = Column(String, nullable=False)
    acuracia = Column(Float)
    precisao = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    hiperparametros = Column(String, nullable=True)
    tempo_treino = Column(Float)

    # datetime.utcnow() esta deprecado. Usamos timezone-aware nativo.
    data_execucao = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ConexaoPostgres:
    """
    Fornece o gerenciamento de sessoes do PostgreSQL via SQLAlchemy.
    Suporta fallback para SQLite local caso DATABASE_URL nao esteja disponivel.
    """

    def __init__(self, url: str | None = None) -> None:
        self.url: str = (
            url
            or os.getenv("DATABASE_URL", "sqlite:///reports/banco_local.db")
            or "sqlite:///reports/banco_local.db"
        )

        # Garante que a pasta reports exista para o sqlite local
        if self.url.startswith("sqlite:///reports/"):  # type: ignore[union-attr]
            os.makedirs("reports", exist_ok=True)

        self.engine = create_engine(self.url, echo=False)  # type: ignore[arg-type]
        Base.metadata.create_all(self.engine)
        self._migrar_schema()
        # expire_on_commit=False → atributos NÃO expiram após commit,
        # evitando o erro "Instance is not bound to a Session" ao acessar
        # campos de objetos ORM fora da transação.
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        logger.info(f"Conexao com banco de dados inicializada: {self.url.split(chr(58))[0]}")  # type: ignore[union-attr]

    def _migrar_schema(self) -> None:
        """Garante que colunas recém-adicionadas existam na tabela experimentos."""
        try:
            inspector = inspect(self.engine)
            if "experimentos" in inspector.get_table_names():
                colunas_existentes = {col["name"] for col in inspector.get_columns("experimentos")}
                colunas_esperadas = {
                    "precisao": "FLOAT",
                    "recall": "FLOAT",
                    "f1": "FLOAT",
                    "hiperparametros": "VARCHAR",
                }
                with self.engine.connect() as conn:
                    for col, tipo in colunas_esperadas.items():
                        if col not in colunas_existentes:
                            conn.execute(text(f"ALTER TABLE experimentos ADD COLUMN {col} {tipo}"))
                    conn.commit()
        except Exception as e:
            logger.warning(f"Aviso ao verificar/migrar schema do banco: {e}")

    @contextmanager
    def obter_sessao(self) -> Generator[Session, None, None]:
        """
        Gerenciador de contexto seguro para transacoes no banco.

        Nota: a sessao é fechada no finally, portanto objetos ORM retornados
        dentro do bloco ``with`` ficam *detached* após o bloco. Use
        ``listar_experimentos()`` para obter dicts puros sem esse risco, ou
        acesse todos os atributos *dentro* do bloco ``with``.

        Yields:
            Session: Sessao ativa do SQLAlchemy.
        """
        sessao = self.SessionLocal()
        try:
            yield sessao
            sessao.commit()
        except Exception as e:
            sessao.rollback()
            logger.error(f"Erro em transacao de banco de dados: {e!s}")
            raise
        finally:
            sessao.close()

    def listar_experimentos(self, limite: int = 200) -> list[dict]:
        """Retorna todos os experimentos como lista de dicts puros (sem ORM detached).

        Converte cada linha para dict *dentro* da sessão ativa, antes de fechá-la,
        eliminando o risco de ``DetachedInstanceError`` / ``bhk3``.

        Args:
            limite: Número máximo de registros retornados.

        Returns:
            Lista de dicts com as colunas do modelo ``Experimento``.
        """
        with self.obter_sessao() as sessao:
            registros = (
                sessao.query(Experimento)
                .order_by(Experimento.data_execucao.desc())
                .limit(limite)
                .all()
            )
            # Serializa DENTRO da sessão — nunca após sessao.close()
            return [
                {
                    "id": r.id,
                    "modelo": r.modelo,
                    "acuracia": r.acuracia,
                    "precisao": r.precisao,
                    "recall": r.recall,
                    "f1": r.f1,
                    "hiperparametros": r.hiperparametros,
                    "tempo_treino": r.tempo_treino,
                    "data_execucao": r.data_execucao,
                }
                for r in registros
            ]
