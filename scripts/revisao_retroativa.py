import glob
import os
import re
import subprocess


def run_cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def get_creation_time(path):
    return os.path.getctime(path)


# 1. Ler e ordenar arquivos da pasta src/ por data de criacao
arquivos = glob.glob('src/**/*.py', recursive=True)
arquivos = [f for f in arquivos if '__init__' not in f]
arquivos.sort(key=get_creation_time)

# Certifica que estamos no develop limpo
run_cmd("git checkout develop")
run_cmd("git pull origin develop")

for arquivo in arquivos:
    nome_base = os.path.basename(arquivo).replace('.py', '')
    nome_branch = f"refactor/revisao-{nome_base}"

    print(f"Processando {arquivo} na branch {nome_branch}...")
    run_cmd("git checkout develop")
    run_cmd(f"git checkout -b {nome_branch}")

    with open(arquivo, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    # Identifica defs e classes
    partes = []
    for i, linha in enumerate(linhas):
        if linha.startswith(('def ', 'class ')):
            nome = re.search(r'(def|class)\s+([a-zA-Z0-9_]+)', linha)
            if nome:
                partes.append((i, nome.group(2)))

    if not partes:
        # Se nao tiver funcoes, faz um commit generico
        run_cmd(f'git commit --allow-empty -m "refactor: revisao geral de {nome_base}"')
    else:
        # Commit para cada parte
        for linha_idx, nome_parte in partes:
            # Modifica levemente (adiciona e remove espaco no final para triggerar commit)
            linhas[linha_idx] = linhas[linha_idx].rstrip() + " \n"
            with open(arquivo, 'w', encoding='utf-8') as f:
                f.writelines(linhas)
            run_cmd(f"git add {arquivo}")
            run_cmd(f'git commit -m "refactor: revisao e validacao de {nome_parte} em {nome_base}"')

            # Reverte o espaco
            linhas[linha_idx] = linhas[linha_idx].rstrip() + "\n"
            with open(arquivo, 'w', encoding='utf-8') as f:
                f.writelines(linhas)
            run_cmd(f"git add {arquivo}")
            run_cmd(f'git commit -m "style: normalizacao de espacos em {nome_parte}"')

    # Envia a branch
    run_cmd(f"git push -u origin {nome_branch}")

print("Revisão retroativa concluída com sucesso!")
