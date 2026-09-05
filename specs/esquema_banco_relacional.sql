-- ============================================================================
-- ESQUEMA DO BANCO DE DADOS RELACIONAL (PostgreSQL 15)
-- Plataforma MNIST: Análise Preditiva e Robustez
-- ============================================================================

CREATE TABLE IF NOT EXISTS configuracoes_experimento (
    id_configuracao SERIAL PRIMARY KEY,
    chave VARCHAR(100) UNIQUE NOT NULL,
    valor_json JSONB NOT NULL,
    descricao TEXT,
    atualizado_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS execucoes_experimento (
    id_execucao VARCHAR(64) PRIMARY KEY,
    nome_modelo VARCHAR(100) NOT NULL,
    tipo_algoritmo VARCHAR(50) NOT NULL,
    hiperparametros JSONB NOT NULL,
    semente_utilizada INT NOT NULL DEFAULT 42,
    tempo_treino_segundos NUMERIC(10, 4) NOT NULL,
    tempo_inferencia_segundos NUMERIC(10, 4),
    status VARCHAR(20) NOT NULL DEFAULT 'CONCLUIDO',
    criado_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metricas_desempenho (
    id_metrica SERIAL PRIMARY KEY,
    id_execucao VARCHAR(64) NOT NULL REFERENCES execucoes_experimento(id_execucao) ON DELETE CASCADE,
    acuracia_global NUMERIC(6, 4) NOT NULL,
    precisao_ponderada NUMERIC(6, 4) NOT NULL,
    revocacao_ponderada NUMERIC(6, 4) NOT NULL,
    f1_score_ponderado NUMERIC(6, 4) NOT NULL,
    relatorio_classificacao JSONB,
    registrado_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logs_auditoria (
    id_log SERIAL PRIMARY KEY,
    modulo_origem VARCHAR(100) NOT NULL,
    nivel_severidade VARCHAR(20) NOT NULL,
    mensagem TEXT NOT NULL,
    metadados JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_execucoes_modelo ON execucoes_experimento(nome_modelo);
CREATE INDEX IF NOT EXISTS idx_metricas_execucao ON metricas_desempenho(id_execucao);
