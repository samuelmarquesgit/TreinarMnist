import argparse
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plataforma Empresarial MNIST')
    parser.add_argument('--modo', choices=['cli', 'web', 'mcp'], default='cli')
    args = parser.parse_args()

    if args.modo == 'web':
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif args.modo == 'mcp':
        subprocess.run([sys.executable, "-m", "src.mcp.servidor"])
    else:
        from src.fachada import FachadaPipelineIA
        print('Iniciando Pipeline via CLI...')
        fachada = FachadaPipelineIA()
        fachada.inicializar_dados()
        print('Dados carregados. Treinando Regressão Logística...')
        fachada.treinar_modelo('RegressaoLogistica')
        metricas = fachada.avaliar_modelo('RegressaoLogistica')
        print(f'Acurácia: {metricas["acuracia"]}')

        estatisticas = fachada.obter_estatisticas_dados('treino')
        print(f'Estatísticas dos dados de treino: {estatisticas}')
