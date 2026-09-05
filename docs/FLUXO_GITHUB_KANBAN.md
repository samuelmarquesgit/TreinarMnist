# Fluxo de Trabalho Semântico: GitHub, Kanban e CI/CD (TreinarMnist)

Este repositório (`samuelmarquesgit/TreinarMnist`) adota um fluxo de trabalho rigorosamente semântico e baseado no **Git Flow**, integrado a ferramentas avançadas de **CI/CD** e gerenciamento de projetos ágeis (**Kanban**).

## 🚀 O Fluxo Principal (Branching & PRs)

Todas as funcionalidades e correções seguem o fluxo:
**`feature/*` $\to$ Pull Request $\to$ `develop` $\to$ Pull Request $\to$ `main`**

1. **`main`:** Código de produção final. Nenhum commit direto é permitido. As submissões chegam exclusivamente através de PRs vindos de `develop`, aprovados pelo **CI-Gate**.
2. **`develop`:** Branch de integração contínua. É a base de onde nascem as *features* e onde são testadas globalmente (SonarCloud, Trivy).
3. **Branches Semânticas (`feat/`, `fix/`, `docs/`, `refactor/`):** Branches de trabalho. Criadas exclusivamente a partir do `develop`.

---

## 🛠️ Orquestrador GitHub (Scripts)

Para facilitar a aderência ao padrão, criamos o script `scripts/orquestrador_github.py`. Ele utiliza o `gh` (GitHub CLI) sob o capô para automatizar a burocracia do fluxo semântico e vincular as atividades ao seu **Projeto Kanban (ID 6)**.

### Passo 1: Criar uma Issue Semântica e Vincular ao Kanban
Para iniciar uma tarefa, crie a Issue pelo orquestrador. Ele aplicará as labels e vinculará o card à coluna *Todo* do projeto Kanban.

```bash
python scripts/orquestrador_github.py issue --tipo "feat" --titulo "Implementar Random Forest" --desc "Cria modelo de floresta aleatória no backend."
```
*(Anote o número da issue gerado, ex: 25)*

### Passo 2: Criar a Branch Semântica
Com base na Issue criada (ex: `#25`), o orquestrador atualiza a branch local `develop` e gera a nova branch de trabalho corretamente nomeada:

```bash
python scripts/orquestrador_github.py branch --issue 25 --tipo "feat" --desc "random forest"
```
*(Isso cria a branch: `feat/issue-25-random-forest`)*

### Passo 3: Commits Semânticos
Durante o desenvolvimento, faça commits seguindo o padrão Conventional Commits em português:
- `feat: implementa algoritmo random forest e fabrica de modelos`
- `fix: corrige divisao por zero no avaliador de metricas`
- `docs: atualiza documentacao da pipeline`

### Passo 4: Abrir Pull Request (PR)
Quando o desenvolvimento terminar, use o orquestrador para abrir o PR para `develop`. Ele configurará o texto informando que o merge fechará a issue (`Closes #25`), o que moverá automaticamente o card no Kanban.

```bash
python scripts/orquestrador_github.py pr --issue 25 --branch "feat/issue-25-random-forest"
```

## 🛡️ Pipeline CI/CD (GitHub Actions)

A configuração está em `.github/workflows/ci.yml`. Ao abrir o Pull Request ou fazer push, o **GitHub Actions** acionará os seguintes _Jobs_:

1. **Qualidade e Testes:**
   - **Flake8:** Linting rigoroso do código.
   - **Pytest:** Suíte de testes com cobertura de código (Code Coverage).
   - **Safety / Pip Audit:** Auditoria para garantir que `requirements.txt` não possui pacotes com CVEs conhecidos (Vulnerabilidades).

2. **Security Scans:**
   - **TruffleHog:** Escaneamento de Secrets vazados (senhas, tokens) em toda a árvore do git.
   - **Trivy Image Scan:** Constrói uma imagem Docker (`Dockerfile`) da aplicação e realiza a varredura por vulnerabilidades de OS e bibliotecas.

3. **Quality Gate (SonarCloud):**
   - Realiza análise estática avançada contra *Code Smells*, Bugs e duplicação de código. Requer que o repositório tenha os secrets `GITHUB_TOKEN` e `SONAR_TOKEN` configurados.

4. **CI-Gate:**
   - Job consolidador que aprova ou bloqueia o merge caso algum dos passos críticos acima falhe.
