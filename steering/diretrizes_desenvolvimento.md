# 🧭 Diretrizes de Desenvolvimento e Padrões de Código

Este documento orienta desenvolvedores e agentes de IA sobre as práticas arquiteturais e de codificação obrigatórias neste repositório.

---

## 1. Regras de Nomenclatura e Idioma
* **100% Português do Brasil (`pt-BR`):**
  - Módulos: `snake_case` em português (ex: `carregador_dados.py`, `pre_processamento.py`).
  - Classes: `PascalCase` em português (ex: `FlorestaAleatoriaClassificador`, `ModeloAbstratoIA`).
  - Funções e Métodos: `snake_case` em português (ex: `treinar()`, `divisao_estratificada_treino_val_teste()`).
  - Variáveis e Constantes: `snake_case` e `UPPER_SNAKE_CASE` em português (ex: `taxa_aprendizado`, `SEMENTE_ALEATORIA`).

---

## 2. Padrões Arquiteturais e GoF
1. **Clean Architecture / Separação de Camadas:**
   - Camada de Domínio e Modelos (`src/modelos/`): Isola a matemática dos classificadores.
   - Camada de Persistência (`src/banco_dados/`): Desacopla PostgreSQL e MongoDB através do *Repository Pattern*.
   - Camada de Aplicação (`src/fachada.py`): Unifica orquestrações complexas.
   - Camada de Apresentação (`main.py` e `mcp_servidor.py`): Expõe interfaces CLI e MCP.
2. **Strategy Pattern:** Todos os modelos implementam `ModeloAbstratoIA`.
3. **Factory Method:** Instanciação centralizada em `FabricaModelos`.
4. **Fallback Gracioso:** Os repositórios devem sempre funcionar offline se o Docker estiver desligado, salvando em arquivos locais (`reports/`).

---

## 3. Tipagem e Documentação
* Type Hints obrigatórios em todas as assinaturas de funções (`typing`).
* Docstrings completas no padrão Google em todas as classes e funções públicas.
* Testes unitários obrigatórios para novas funções adicionadas (`tests/`).
