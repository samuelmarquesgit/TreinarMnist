# Estagio 1: Build das dependencias
FROM python:3.10-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Estagio 2: Imagem final enxuta
FROM python:3.10-slim

# Cria usuario nao-root por seguranca (Melhor pratica SecOps)
RUN useradd -m -s /bin/bash mnist_user

WORKDIR /app

# Copia dependencias pre-compiladas
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Instala dependencias sem cache e remove as rodas
RUN pip install --no-cache /wheels/* \
    && rm -rf /wheels

# Copia o codigo do projeto e altera o dono dos arquivos
COPY --chown=mnist_user:mnist_user . .

# Garante a criacao dos diretorios necessarios
RUN mkdir -p reports data && chown -R mnist_user:mnist_user reports data

# Muda para o usuario nao-root
USER mnist_user

EXPOSE 8501
EXPOSE 8000

# Endpoint Web como padrao
CMD ["python", "main.py", "--modo", "web"]
