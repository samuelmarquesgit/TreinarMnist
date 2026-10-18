"""Garante que as mensagens de console funcionam no terminal padrão do Windows (cp1252).

Regressão: ``scripts/gerar_colecao_postman.py`` quebrava com ``UnicodeEncodeError`` ao imprimir
"→" no PowerShell/cmd sem ``PYTHONIOENCODING=utf-8``. Logs com esses caracteres não quebram
(o stderr usa ``backslashreplace``), mas aparecem como ``\\u2192``.
"""

import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVOS = sorted(
    [
        *RAIZ.joinpath("api").rglob("*.py"),
        *RAIZ.joinpath("pipeline_offline").rglob("*.py"),
        *RAIZ.joinpath("scripts").glob("gerar_*.py"),
        RAIZ / "main.py",
    ]
)
METODOS_LOG = {"debug", "info", "warning", "error", "exception", "critical"}


def _mensagens_de_console(arquivo: Path) -> list[tuple[int, str]]:
    """Textos literais passados a print() e logger.<nível>() no arquivo."""
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    mensagens = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        funcao = no.func
        eh_print = isinstance(funcao, ast.Name) and funcao.id == "print"
        eh_log = isinstance(funcao, ast.Attribute) and funcao.attr in METODOS_LOG
        if not (eh_print or eh_log):
            continue
        for argumento in no.args:
            for parte in ast.walk(argumento):
                if isinstance(parte, ast.Constant) and isinstance(parte.value, str):
                    mensagens.append((parte.lineno, parte.value))
    return mensagens


@pytest.mark.parametrize("arquivo", ARQUIVOS, ids=lambda p: p.relative_to(RAIZ).as_posix())
def test_mensagens_de_console_sao_compativeis_com_cp1252(arquivo):
    problemas = []
    for linha, texto in _mensagens_de_console(arquivo):
        try:
            texto.encode("cp1252")
        except UnicodeEncodeError as erro:
            problemas.append(f"linha {linha}: {ascii(texto[erro.start : erro.end])}")
    assert not problemas, problemas


def test_varredura_encontra_as_mensagens():
    """Sanidade: a varredura de fato enxerga prints e logs (senão o teste acima seria inócuo)."""
    total = sum(len(_mensagens_de_console(a)) for a in ARQUIVOS)
    assert total > 20
