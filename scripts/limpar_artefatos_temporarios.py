"""Script utilitário para limpeza de artefatos temporários, caches e compilações."""

import os
import shutil


def limpar_diretorios():
    """Remove pastas e arquivos temporários de compilação e teste."""
    padroes_diretorios = [
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist"
    ]

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[LIMPEZA] Iniciando limpeza a partir de: {raiz}")

    for dirpath, dirnames, filenames in os.walk(raiz, topdown=False):
        for dirname in dirnames:
            if dirname in padroes_diretorios or dirname == "__pycache__":
                caminho_completo = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(caminho_completo)
                    print(f"  [REMOVIDO] {caminho_completo}")
                except Exception as e:
                    print(f"  [ERRO] Não foi possível remover {caminho_completo}: {e}")

    print("[CONCLUÍDO] Limpeza finalizada.")


if __name__ == "__main__":
    limpar_diretorios()
