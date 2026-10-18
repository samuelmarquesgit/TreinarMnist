"""Gera a coleção Postman completa da API e os environments em ``docs/postman/`` (TM-052).

Uso (a partir da raiz do projeto):

    python scripts/gerar_colecao_postman.py

A coleção é gerada a partir dos mesmos exemplos de ``docs/exemplos_api`` (TM-049), então nunca
fica dessincronizada da API. Cada requisição tem testes ``pm.test`` (status, contrato e
cabeçalhos); um script de coleção valida o contrato ``SaidaErro`` em toda resposta de erro.

Cenários e servidores:
    {{base_url}}            API normal, com MNIST_API_HABILITAR_DIAGNOSTICO=true (para o 500)
    {{base_url_degradado}}  segunda instância sem modelo (para o 503)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[1]
EXEMPLOS = RAIZ / "docs" / "exemplos_api"
DESTINO = RAIZ / "docs" / "postman"
SCHEMA = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"

DESCRICAO = """Coleção completa da **API MNIST — Classificação de Dígitos** (Projeto Avaliativo, Módulo 2).

Cobre todos os endpoints e todos os códigos HTTP: **200, 400, 404, 405, 422, 500 e 503**.

### Como rodar
1. Importe esta coleção e o environment `API_MNIST.local.postman_environment.json`.
2. Suba a API normal (com a rota de diagnóstico, para o 500):
   `MNIST_API_HABILITAR_DIAGNOSTICO=true uvicorn api.main:app --port 8000`
3. Suba uma segunda instância **sem modelo** (para o 503):
   `MNIST_API_CAMINHO_MODELO=models/inexistente.joblib uvicorn api.main:app --port 8001`
4. Para os uploads, defina a pasta de trabalho do Postman como a raiz do repositório
   (Settings → General → Working directory).
5. Rode a coleção (Runner) ou, no terminal:
   `npx newman run docs/postman/API_MNIST.postman_collection.json -e docs/postman/API_MNIST.local.postman_environment.json`
"""

TESTE_COLECAO = """// Executado em toda resposta: rastreabilidade e contrato único de erro
pm.test("[comum] resposta tem X-Request-ID", function () {
    pm.response.to.have.header("X-Request-ID");
});
if (pm.response.code >= 400) {
    pm.test("[comum] erro segue o contrato SaidaErro", function () {
        const e = pm.response.json();
        pm.expect(e.codigo).to.eql(pm.response.code);
        pm.expect(e.erro).to.match(/^[A-Z][A-Z0-9_]*$/);
        pm.expect(e.mensagem).to.be.a("string").and.not.empty;
        pm.expect(e.id_requisicao).to.eql(pm.response.headers.get("X-Request-ID"));
        pm.expect(pm.response.text()).to.not.include("Traceback");
    });
}"""


def _linhas(js: str) -> list[str]:
    return js.strip("\n").splitlines()


def evento(tipo: str, js: str) -> dict[str, Any]:
    return {"listen": tipo, "script": {"type": "text/javascript", "exec": _linhas(js)}}


def requisicao(
    nome: str,
    metodo: str,
    caminho: str,
    testes: str,
    *,
    base: str = "{{base_url}}",
    corpo: str | None = None,
    pre: str | None = None,
    arquivo: tuple[str, str] | None = None,
    sem_campo_arquivo: bool = False,
    cabecalhos: list[dict[str, str]] | None = None,
    descricao: str = "",
) -> dict[str, Any]:
    """Monta um item da coleção (requisição + scripts)."""
    partes = [p for p in caminho.strip("/").split("/") if p]
    req: dict[str, Any] = {
        "method": metodo,
        "header": list(cabecalhos or []),
        "url": {"raw": f"{base}/{'/'.join(partes)}", "host": [base], "path": partes},
        "description": descricao,
    }
    if corpo is not None:
        req["header"].append({"key": "Content-Type", "value": "application/json"})
        req["body"] = {"mode": "raw", "raw": corpo, "options": {"raw": {"language": "json"}}}
    if arquivo is not None:
        src, tipo = arquivo
        req["body"] = {
            "mode": "formdata",
            "formdata": [{"key": "arquivo", "type": "file", "src": src, "contentType": tipo}],
        }
    if sem_campo_arquivo:
        req["body"] = {
            "mode": "formdata",
            "formdata": [{"key": "outro", "value": "x", "type": "text"}],
        }
    eventos = [evento("test", testes)]
    if pre:
        eventos.insert(0, evento("prerequest", pre))
    return {"name": nome, "event": eventos, "request": req}


def teste_status(codigo: int, extra: str = "") -> str:
    return f'pm.test("status {codigo}", () => pm.response.to.have.status({codigo}));\n{extra}'


def teste_erro(codigo: int, erro: str, tipo: str | None = None, campo: str | None = None) -> str:
    js = teste_status(
        codigo,
        f'pm.test("erro {erro}", () => pm.expect(pm.response.json().erro).to.eql("{erro}"));\n',
    )
    if tipo:
        js += f'pm.test("detalhe do tipo {tipo}", () => pm.expect(pm.response.json().detalhes[0].tipo).to.eql("{tipo}"));\n'
    if campo:
        js += f'pm.test("campo {campo}", () => pm.expect(pm.response.json().detalhes[0].campo).to.eql("{campo}"));\n'
    return js


def corpo_a_partir_do_7(transformacao: str) -> str:
    """Pre-request que parte dos pixels do 7 e aplica uma alteração antes de enviar."""
    return f"""const pixels = JSON.parse(pm.collectionVariables.get("pixels_7"));
{transformacao}
pm.variables.set("corpo", JSON.stringify(corpo));"""


TESTE_PREDICAO = """const p = pm.response.json();
pm.test("10 probabilidades que somam 1", () => {
    pm.expect(p.probabilidades).to.have.lengthOf(10);
    pm.expect(p.probabilidades.reduce((a, b) => a + b, 0)).to.be.closeTo(1, 0.01);
});
pm.test("classe é o argmax e confiança é a maior probabilidade", () => {
    pm.expect(p.probabilidades.indexOf(Math.max(...p.probabilidades))).to.eql(p.classe_prevista);
    pm.expect(p.confianca).to.be.closeTo(Math.max(...p.probabilidades), 1e-6);
});
pm.test("cabeçalho de tempo de processamento", () => pm.response.to.have.header("X-Tempo-Processamento-ms"));
"""


def gerar_colecao() -> dict[str, Any]:
    pixels_7 = json.loads((EXEMPLOS / "predict_digito_7.json").read_text(encoding="utf-8"))[
        "pixels"
    ]
    lote = json.loads((EXEMPLOS / "predict_lote_0_3_9.json").read_text(encoding="utf-8"))

    monitoramento = [
        requisicao(
            "GET /health — 200 ok",
            "GET",
            "/health",
            teste_status(
                200,
                """const s = pm.response.json();
pm.test("status ok com modelo em memória", () => { pm.expect(s.status).to.eql("ok"); pm.expect(s.modelo_carregado).to.be.true; });
pm.test("versões semânticas", () => { pm.expect(s.versao_api).to.match(/^\\d+\\.\\d+\\.\\d+$/); pm.expect(s.versao_modelo).to.match(/^\\d+\\.\\d+\\.\\d+$/); });
""",
            ),
        ),
        requisicao(
            "GET /v1/modelo — 200 metadados e métricas",
            "GET",
            "/v1/modelo",
            teste_status(
                200,
                """const m = pm.response.json();
pm.test("métricas de teste presentes e acima de 97%", () => pm.expect(m.metricas_teste.acuracia).to.be.above(0.97));
pm.test("partições 49000/7000/14000", () => pm.expect(m.amostras).to.eql({treino: 49000, validacao: 7000, teste: 14000}));
""",
            ),
        ),
    ]

    sucesso = [
        requisicao(
            "POST /v1/predict — 200 dígito 7",
            "POST",
            "/v1/predict",
            teste_status(
                200,
                TESTE_PREDICAO
                + """pm.test("classe 7, confiável e sem alerta", () => {
    pm.expect(pm.response.json().classe_prevista).to.eql(7);
    pm.expect(pm.response.json().confiavel).to.be.true;
    pm.expect(pm.response.json().alerta_falsa_certeza).to.be.false;
});
""",
            ),
            corpo='{"pixels": {{pixels_7}}}',
        ),
        requisicao(
            "POST /v1/predict/lote — 200 dígitos 0, 3 e 9",
            "POST",
            "/v1/predict/lote",
            teste_status(
                200,
                """const l = pm.response.json();
pm.test("três predições na ordem enviada: 0, 3, 9", () => pm.expect(l.predicoes.map(p => p.classe_prevista)).to.eql([0, 3, 9]));
pm.test("quantidade = 3", () => pm.expect(l.quantidade).to.eql(3));
""",
            ),
            corpo="{{lote_0_3_9}}",
        ),
        requisicao(
            "POST /v1/predict/imagem — 200 imagem própria (6)",
            "POST",
            "/v1/predict/imagem",
            teste_status(
                200,
                TESTE_PREDICAO
                + 'pm.test("reconhece o 6", () => pm.expect(pm.response.json().classe_prevista).to.eql(6));\n',
            ),
            arquivo=("data/custom_digits/meu_numero.jpeg", "image/jpeg"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 200 sete manuscrito (7)",
            "POST",
            "/v1/predict/imagem",
            teste_status(
                200,
                'pm.test("reconhece o 7", () => pm.expect(pm.response.json().classe_prevista).to.eql(7));\n',
            ),
            arquivo=("data/custom_digits/bruno_7_com_barra.png", "image/png"),
        ),
        requisicao(
            "GET /health — X-Request-ID do cliente é ecoado",
            "GET",
            "/health",
            teste_status(
                200,
                'pm.test("id ecoado", () => pm.expect(pm.response.headers.get("X-Request-ID")).to.eql("postman-rastreio-1"));\n',
            ),
            cabecalhos=[{"key": "X-Request-ID", "value": "postman-rastreio-1"}],
        ),
    ]

    erros_422 = [
        requisicao(
            "POST /v1/predict — 422 texto no pixel",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "float_type", "pixels.0"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7('pixels[0] = "abc"; const corpo = {pixels};'),
        ),
        requisicao(
            "POST /v1/predict — 422 número como string",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "float_type", "pixels.0"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7('pixels[0] = "0.5"; const corpo = {pixels};'),
        ),
        requisicao(
            "POST /v1/predict — 422 pixel acima de 1",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "less_than_equal", "pixels.0"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7("pixels[0] = 1.5; const corpo = {pixels};"),
        ),
        requisicao(
            "POST /v1/predict — 422 pixel negativo",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "greater_than_equal", "pixels.0"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7("pixels[0] = -0.1; const corpo = {pixels};"),
        ),
        requisicao(
            "POST /v1/predict — 422 783 pixels",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "too_short", "pixels"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7("const corpo = {pixels: pixels.slice(0, 783)};"),
        ),
        requisicao(
            "POST /v1/predict — 422 campo extra",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "extra_forbidden", "classe"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7("const corpo = {pixels, classe: 7};"),
        ),
        requisicao(
            "POST /v1/predict — 422 corpo vazio",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "missing", "pixels"),
            corpo="{}",
        ),
        requisicao(
            "POST /v1/predict — 422 JSON malformado",
            "POST",
            "/v1/predict",
            teste_erro(422, "VALIDACAO_FALHOU", "json_invalid"),
            corpo='{"pixels": [0.1, 0.2',
        ),
        requisicao(
            "POST /v1/predict/lote — 422 lote vazio",
            "POST",
            "/v1/predict/lote",
            teste_erro(422, "VALIDACAO_FALHOU", "too_short", "amostras"),
            corpo='{"amostras": []}',
        ),
        requisicao(
            "POST /v1/predict/lote — 422 lote com 101 imagens",
            "POST",
            "/v1/predict/lote",
            teste_erro(422, "VALIDACAO_FALHOU", "too_long", "amostras"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7("const corpo = {amostras: Array(101).fill({pixels})};"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 422 sem o campo arquivo",
            "POST",
            "/v1/predict/imagem",
            teste_erro(422, "VALIDACAO_FALHOU", "missing", "arquivo"),
            sem_campo_arquivo=True,
        ),
    ]

    erros_400 = [
        requisicao(
            "POST /v1/predict — 400 imagem sem traço",
            "POST",
            "/v1/predict",
            teste_erro(400, "ENTRADA_INVALIDA", "imagem_vazia", "pixels"),
            corpo="{{corpo}}",
            pre=corpo_a_partir_do_7("const corpo = {pixels: pixels.map(() => 0)};"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 400 arquivo de texto",
            "POST",
            "/v1/predict/imagem",
            teste_erro(400, "ENTRADA_INVALIDA", "tipo_invalido"),
            arquivo=("docs/exemplos_api/upload_arquivo_texto.txt", "text/plain"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 400 texto disfarçado de PNG",
            "POST",
            "/v1/predict/imagem",
            teste_erro(400, "ENTRADA_INVALIDA", "formato_invalido"),
            arquivo=("docs/exemplos_api/upload_arquivo_texto.txt", "image/png"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 400 PNG corrompido",
            "POST",
            "/v1/predict/imagem",
            teste_erro(400, "ENTRADA_INVALIDA", "imagem_corrompida"),
            arquivo=("docs/exemplos_api/upload_imagem_corrompida.png", "image/png"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 400 imagem em branco",
            "POST",
            "/v1/predict/imagem",
            teste_erro(400, "ENTRADA_INVALIDA", "imagem_vazia"),
            arquivo=("docs/exemplos_api/upload_imagem_em_branco.png", "image/png"),
        ),
        requisicao(
            "POST /v1/predict/imagem — 400 resolução acima de 25 megapixels",
            "POST",
            "/v1/predict/imagem",
            teste_erro(400, "ENTRADA_INVALIDA", "imagem_grande_demais"),
            arquivo=("docs/exemplos_api/upload_imagem_gigante.png", "image/png"),
        ),
    ]

    erros_http = [
        requisicao(
            "GET /rota-inexistente — 404",
            "GET",
            "/rota-inexistente",
            teste_erro(404, "NAO_ENCONTRADO"),
        ),
        requisicao(
            "DELETE /health — 405", "DELETE", "/health", teste_erro(405, "METODO_NAO_PERMITIDO")
        ),
    ]

    erro_500 = [
        requisicao(
            "GET /v1/_diagnostico/erro-interno — 500 sem stack trace",
            "GET",
            "/v1/_diagnostico/erro-interno",
            teste_erro(500, "ERRO_INTERNO")
            + """pm.test("mensagem genérica, sem detalhes internos", () => {
    pm.expect(pm.response.json().detalhes).to.be.null;
    pm.expect(pm.response.text()).to.not.include("proposital");
});
""",
            descricao="Requer a API com MNIST_API_HABILITAR_DIAGNOSTICO=true.",
        ),
        requisicao(
            "GET /health — 200 após o 500 (API não caiu)",
            "GET",
            "/health",
            teste_status(
                200,
                'pm.test("continua ok", () => pm.expect(pm.response.json().status).to.eql("ok"));\n',
            ),
        ),
    ]

    degradado = [
        requisicao(
            "GET /health — 200 degradado (sem modelo)",
            "GET",
            "/health",
            teste_status(
                200,
                """pm.test("degradado e sem modelo", () => {
    pm.expect(pm.response.json().status).to.eql("degradado");
    pm.expect(pm.response.json().modelo_carregado).to.be.false;
});
""",
            ),
            base="{{base_url_degradado}}",
        ),
        requisicao(
            "POST /v1/predict — 503 modelo indisponível",
            "POST",
            "/v1/predict",
            teste_erro(503, "MODELO_INDISPONIVEL"),
            base="{{base_url_degradado}}",
            corpo='{"pixels": {{pixels_7}}}',
        ),
        requisicao(
            "GET /v1/modelo — 503 modelo indisponível",
            "GET",
            "/v1/modelo",
            teste_erro(503, "MODELO_INDISPONIVEL"),
            base="{{base_url_degradado}}",
        ),
    ]

    def pasta(nome: str, descricao: str, itens: list[dict[str, Any]]) -> dict[str, Any]:
        return {"name": nome, "description": descricao, "item": itens}

    return {
        "info": {
            "name": "API MNIST — Classificação de Dígitos (Módulo 2)",
            "description": DESCRICAO,
            "schema": SCHEMA,
        },
        "event": [evento("test", TESTE_COLECAO)],
        "variable": [
            {
                "key": "pixels_7",
                "value": json.dumps(pixels_7),
                "description": "784 pixels de um 7 real do conjunto de teste",
            },
            {
                "key": "lote_0_3_9",
                "value": json.dumps(lote),
                "description": "Lote com os dígitos reais 0, 3 e 9",
            },
        ],
        "item": [
            pasta("1. Monitoramento", "Saúde do serviço e metadados do modelo.", monitoramento),
            pasta(
                "2. Inferência — sucesso (200)",
                "Predições por pixels, lote e upload de imagem.",
                sucesso,
            ),
            pasta(
                "3. Erros 422 — contrato Pydantic",
                "Dados fora do contrato, barrados antes do modelo.",
                erros_422,
            ),
            pasta(
                "4. Erros 400 — entrada impossível de classificar",
                "Formato válido, conteúdo inválido.",
                erros_400,
            ),
            pasta(
                "5. Erros 404 e 405 — roteamento",
                "Rota ou método inexistente no formato padronizado.",
                erros_http,
            ),
            pasta(
                "6. Erro 500 — falha inesperada",
                "Exige MNIST_API_HABILITAR_DIAGNOSTICO=true na API.",
                erro_500,
            ),
            pasta(
                "7. Cenário degradado — 503",
                "Usa {{base_url_degradado}}: instância da API sem modelo.",
                degradado,
            ),
        ],
    }


def gerar_environment(nome: str, base: str, degradado: str) -> dict[str, Any]:
    return {
        "name": nome,
        "values": [
            {"key": "base_url", "value": base, "type": "default", "enabled": True},
            {"key": "base_url_degradado", "value": degradado, "type": "default", "enabled": True},
        ],
        "_postman_variable_scope": "environment",
    }


def salvar(nome: str, conteudo: dict[str, Any]) -> Path:
    caminho = DESTINO / nome
    caminho.write_text(
        json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return caminho


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    colecao = gerar_colecao()
    salvar("API_MNIST.postman_collection.json", colecao)
    salvar(
        "API_MNIST.local.postman_environment.json",
        gerar_environment("API MNIST — local", "http://localhost:8000", "http://localhost:8001"),
    )
    total = sum(len(p["item"]) for p in colecao["item"])
    print(
        f"Coleção gerada: {total} requisições em {len(colecao['item'])} pastas -> {DESTINO.relative_to(RAIZ).as_posix()}/"
    )


if __name__ == "__main__":
    main()
