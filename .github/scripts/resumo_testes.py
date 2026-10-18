"""Gera um resumo em Markdown dos testes e da cobertura para o ``$GITHUB_STEP_SUMMARY``.

Uso (no workflow):
    python .github/scripts/resumo_testes.py junit.xml coverage.xml "Título" >> "$GITHUB_STEP_SUMMARY"

Só usa a biblioteca padrão, para rodar em qualquer job sem dependências extras. Se algum
arquivo não existir (ex.: os testes nem chegaram a rodar), informa isso em vez de falhar.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def resumo_junit(caminho: Path) -> list[str]:
    """Tabela com totais de testes a partir de um relatório JUnit."""
    if not caminho.exists():
        return [f"> Relatório `{caminho.name}` não encontrado: os testes não chegaram a rodar.", ""]
    raiz = ET.parse(caminho).getroot()
    suites = [raiz] if raiz.tag == "testsuite" else list(raiz.iter("testsuite"))
    total = sum(int(s.get("tests", 0)) for s in suites)
    falhas = sum(int(s.get("failures", 0)) for s in suites)
    erros = sum(int(s.get("errors", 0)) for s in suites)
    pulados = sum(int(s.get("skipped", 0)) for s in suites)
    tempo = sum(float(s.get("time", 0)) for s in suites)
    aprovados = total - falhas - erros - pulados
    status = "✅ todos aprovados" if falhas + erros == 0 else "❌ há falhas"
    linhas = [
        f"**{status}**",
        "",
        "| Total | Aprovados | Falhas | Erros | Pulados | Tempo |",
        "|---|---|---|---|---|---|",
        f"| {total} | {aprovados} | {falhas} | {erros} | {pulados} | {tempo:.1f} s |",
        "",
    ]
    falhos = [
        f"- `{caso.get('classname')}::{caso.get('name')}`"
        for caso in raiz.iter("testcase")
        if caso.find("failure") is not None or caso.find("error") is not None
    ]
    if falhos:
        linhas += [
            "<details><summary>Testes com falha</summary>",
            "",
            *falhos[:50],
            "",
            "</details>",
            "",
        ]
    return linhas


def resumo_cobertura(caminho: Path) -> list[str]:
    """Tabela com a cobertura de linhas por pacote a partir de um relatório Cobertura."""
    if not caminho.exists():
        return [f"> Relatório `{caminho.name}` não encontrado.", ""]
    raiz = ET.parse(caminho).getroot()
    linhas = [
        f"**Cobertura total de linhas: {float(raiz.get('line-rate', 0)) * 100:.1f}%**",
        "",
        "| Pacote | Cobertura |",
        "|---|---|",
    ]
    pacotes: dict[str, list[int]] = {}
    for classe in raiz.iter("class"):
        pacote = classe.get("filename", "").replace("\\", "/").split("/")[0] or "(raiz)"
        cobertas = sum(1 for ln in classe.iter("line") if int(ln.get("hits", 0)) > 0)
        total = sum(1 for _ in classe.iter("line"))
        acumulado = pacotes.setdefault(pacote, [0, 0])
        acumulado[0] += cobertas
        acumulado[1] += total
    for pacote, (cobertas, total) in sorted(pacotes.items()):
        percentual = cobertas / total * 100 if total else 100.0
        linhas.append(f"| `{pacote}` | {percentual:.1f}% |")
    return [*linhas, ""]


def main(argumentos: list[str]) -> None:
    """Imprime o resumo completo no stdout."""
    junit, cobertura, titulo = Path(argumentos[0]), Path(argumentos[1]), argumentos[2]
    print("\n".join([f"## 🧪 {titulo}", "", *resumo_junit(junit), *resumo_cobertura(cobertura)]))


if __name__ == "__main__":
    main(sys.argv[1:])
