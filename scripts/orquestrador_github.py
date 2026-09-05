import os
import subprocess
import argparse
import sys

# ID do Projeto (Kanban) do Usuário fornecido na URL: https://github.com/users/samuelmarquesgit/projects/6
PROJECT_ID = "6"
OWNER = "samuelmarquesgit"

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar: {cmd}\n{e.stderr}")
        sys.exit(1)

def criar_issue(titulo, tipo, descricao):
    print(f"🚀 Criando Issue Semântica: [{tipo}] {titulo}...")
    label = tipo.lower()
    cmd = f'gh issue create --title "{tipo}: {titulo}" --body "{descricao}" --label "{label}"'
    output = run_cmd(cmd)
    print(f"✅ Issue criada: {output}")
    
    # Extrai a URL para pegar o ID numérico da issue e vincular ao projeto
    # O output típico é a URL da issue
    issue_url = output
    issue_number = issue_url.split("/")[-1]
    
    # Linka a issue ao Kanban
    vincular_ao_kanban(issue_url)
    
    return issue_number

def vincular_ao_kanban(item_url):
    print(f"📋 Vinculando {item_url} ao Projeto Kanban {PROJECT_ID}...")
    cmd = f'gh project item-create {PROJECT_ID} --owner {OWNER} --url {item_url}'
    try:
        run_cmd(cmd)
        print(f"✅ Vinculado ao Kanban!")
    except Exception:
        print("⚠️ Aviso: Não foi possível vincular automaticamente ao Kanban. Verifique suas permissões de 'project' no GitHub Token.")

def criar_branch(issue_number, tipo, nome_descritivo):
    branch_name = f"{tipo}/issue-{issue_number}-{nome_descritivo.replace(' ', '-')}"
    print(f"🌿 Criando branch semântica: {branch_name} a partir da develop...")
    run_cmd("git checkout develop")
    run_cmd("git pull origin develop")
    run_cmd(f"git checkout -b {branch_name}")
    print(f"✅ Branch {branch_name} criada e selecionada!")

def abrir_pr(issue_number, branch_name):
    print("🔄 Fazendo push e abrindo PR para a develop...")
    run_cmd(f"git push -u origin {branch_name}")
    cmd = f'gh pr create --base develop --head {branch_name} --title "Merge {branch_name}" --body "Closes #{issue_number}"'
    output = run_cmd(cmd)
    print(f"✅ Pull Request criado com sucesso: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestrador Semântico do GitHub e Kanban (TreinarMnist)")
    parser.add_argument("acao", choices=["issue", "branch", "pr"], help="Ação a ser executada")
    parser.add_argument("--titulo", help="Título da Issue", default="Nova Tarefa")
    parser.add_argument("--tipo", help="Tipo semântico (feat, fix, docs, refactor)", default="feat")
    parser.add_argument("--desc", help="Descrição da Issue ou Branch", default="")
    parser.add_argument("--issue", help="Número da Issue relacionada")
    parser.add_argument("--branch", help="Nome da branch para o PR")
    
    args = parser.parse_args()
    
    if args.acao == "issue":
        criar_issue(args.titulo, args.tipo, args.desc)
    elif args.acao == "branch":
        if not args.issue:
            print("Erro: É necessário fornecer o --issue para criar a branch.")
            sys.exit(1)
        criar_branch(args.issue, args.tipo, args.desc)
    elif args.acao == "pr":
        if not args.issue or not args.branch:
            print("Erro: É necessário fornecer --issue e --branch para abrir o PR.")
            sys.exit(1)
        abrir_pr(args.issue, args.branch)
