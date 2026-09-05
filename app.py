import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página Streamlit
st.set_page_config(page_title='Plataforma Empresarial MNIST', layout='wide')

try:
    from src.fachada import FachadaPipelineIA
except ImportError:
    st.error("Erro ao importar a fachada do projeto. Execute o script da raiz do repositório.")
    st.stop()

@st.cache_resource
def carregar_fachada():
    f = FachadaPipelineIA()
    with st.spinner("Baixando/Carregando dados MNIST... Isso pode demorar na primeira vez."):
        f.inicializar_dados()
    return f

st.title('🧠 Plataforma Empresarial MNIST')
st.markdown("Bem-vindo ao **TreinarMnist**. Selecione uma funcionalidade no menu lateral.")

fachada = carregar_fachada()

# Navegação lateral (Menu)
menu = st.sidebar.selectbox('Menu Principal', [
    '📊 Análise Exploratória',
    '📈 Análise Estatística',
    '🏆 Modelos & Benchmarks',
    '🧪 Testes de Robustez OOD',
    '✍️ Laboratório de Visão'
])

if menu == '📊 Análise Exploratória':
    st.header('Análise Exploratória (EDA)')
    st.info(f"Dados carregados. Total de amostras de treino: {fachada.X_treino.shape[0]}")
    if st.button("Mostrar Distribuição de Classes"):
        classes, contagens = np.unique(fachada.y_treino, return_counts=True)
        df_classes = pd.DataFrame({"Dígito": classes, "Quantidade": contagens})
        st.bar_chart(df_classes, x="Dígito", y="Quantidade")

elif menu == '📈 Análise Estatística':
    st.header('Análise Estatística Interativa')
    tipo_dado = st.radio("Selecione os dados para análise:", ['Treino', 'Teste'])
    if st.button("Calcular Estatísticas Descritivas"):
        est = fachada.obter_estatisticas_dados(tipo_dado.lower())
        st.json(est)

elif menu == '🏆 Modelos & Benchmarks':
    st.header('Treinamento e Benchmarks')
    modelo_selecionado = st.selectbox("Selecione o Modelo", ['RegressaoLogistica', 'ArvoreDecisao', 'FlorestaAleatoria', 'SVM'])
    if st.button("Treinar e Avaliar"):
        with st.spinner(f"Treinando {modelo_selecionado}..."):
            fachada.treinar_modelo(modelo_selecionado)
            metricas = fachada.avaliar_modelo(modelo_selecionado)
            st.success(f"Acurácia: {metricas['acuracia']:.4f}")
            st.write("Outras métricas:", metricas)

elif menu == '🧪 Testes de Robustez OOD':
    st.header('Testes Out-of-Distribution (OOD)')
    st.warning("Módulo de saturação Softmax em desenvolvimento...")

elif menu == '✍️ Laboratório de Visão':
    st.header('Laboratório de Visão Computacional')
    uploaded_file = st.file_uploader("Envie uma imagem do dígito (28x28)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagem carregada.', use_column_width=False)
        st.info("O processamento da imagem está pronto na API de visão, aguardando integração completa.")
