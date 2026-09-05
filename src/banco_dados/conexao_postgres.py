import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class Experimento(Base):
    __tablename__ = 'experimentos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    modelo = Column(String, nullable=False)
    acuracia = Column(Float)
    tempo_treino = Column(Float)
    data_execucao = Column(DateTime, default=datetime.utcnow)

class ConexaoPostgres:
    def __init__(self):
        url = os.getenv('DATABASE_URL', 'sqlite:///reports/banco_local.db')
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def obter_sessao(self):
        return self.Session()
