"""Ponto de entrada da Plataforma Empresarial MNIST.

Modos:
    python main.py --modo cli    pipeline de treino e avaliação no terminal (padrão)
    python main.py --modo web    painel Streamlit (app.py)
    python main.py --modo mcp    servidor MCP para agentes (src/mcp_servidor.py)
    python main.py --modo api    API REST FastAPI do Módulo 2 (api/main.py) com Uvicorn
"""

import argparse
import subprocess
import sys

MODOS = ("cli", "web", "mcp", "api")
MODULO_MCP = "src.mcp_servidor"
APLICACAO_API = "api.main:app"


def montar_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Plataforma Empresarial MNIST")
    parser.add_argument("--modo", choices=MODOS, default="cli")
    parser.add_argument("--host", default="127.0.0.1", help="host da API (modo api)")
    parser.add_argument("--porta", type=int, default=8000, help="porta da API (modo api)")
    return parser


def executar_cli() -> None:
    """Treina e avalia a Regressão Logística pelo terminal (comportamento do Módulo 1)."""
    from src.fachada import FachadaPipelineIA

    print("Iniciando Pipeline via CLI...")
    fachada = FachadaPipelineIA()
    fachada.inicializar_dados()
    print("Dados carregados. Treinando Regressão Logística...")
    fachada.treinar_modelo("RegressaoLogistica")
    metricas = fachada.avaliar_modelo("RegressaoLogistica")
    print(f"Acurácia: {metricas['acuracia']}")

    estatisticas = fachada.obter_estatisticas_dados("treino")
    print(f"Estatísticas dos dados de treino: {estatisticas}")


def executar(argumentos: list[str] | None = None) -> int:
    """Executa o modo escolhido.

    Args:
        argumentos: Argumentos da linha de comando (``None`` usa ``sys.argv``).

    Returns:
        Código de saída do processo.
    """
    args = montar_parser().parse_args(argumentos)
    if args.modo == "web":
        return subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"]).returncode
    if args.modo == "mcp":
        return subprocess.run([sys.executable, "-m", MODULO_MCP]).returncode
    if args.modo == "api":
        import uvicorn

        uvicorn.run(APLICACAO_API, host=args.host, port=args.porta)
        return 0
    executar_cli()
    return 0


if __name__ == "__main__":
    sys.exit(executar())
