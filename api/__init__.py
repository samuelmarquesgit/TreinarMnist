"""API REST de inferência MNIST (Módulo 2): microsserviço FastAPI independente do restante da plataforma.

Camadas:
    core/      configuração, logging e exceções de domínio
    schemas/   contratos Pydantic de entrada, saída e erro
    services/  carregamento do artefato e inferência (sem nenhuma referência a HTTP)
    routers/   endpoints agrupados com APIRouter (só convertem e delegam)
    main.py    criação da aplicação, ciclo de vida (lifespan) e inclusão dos roteadores

A API só lê ``models/modelo_mnist_v1.joblib`` e ``models/metricas_modelo_v1.json``; nunca importa
``src/`` nem ``pipeline_offline/``.
"""
