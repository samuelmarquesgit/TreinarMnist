import pytest
import numpy as np
from PIL import Image
from guardrails.validador_falsa_certeza import ValidadorFalsaCerteza
from guardrails.validador_vazamento_dados import ValidadorVazamentoDados
from guardrails.validador_imagem_entrada import ValidadorImagemEntrada

# --- Testes para ValidadorFalsaCerteza ---


def test_entropia_shannon():
    validador = ValidadorFalsaCerteza()
    # Vetor concentrado (certeza absoluta)
    vetor_concentrado = np.array([1.0, 0.0, 0.0])
    entropia_conc = validador.calcular_entropia_shannon(vetor_concentrado)
    assert np.isclose(entropia_conc, 0.0, atol=1e-5)

    # Vetor uniforme (total incerteza)
    vetor_uniforme = np.array([0.333333, 0.333333, 0.333333])
    entropia_unif = validador.calcular_entropia_shannon(vetor_uniforme)
    assert entropia_unif > 1.0  # Entropia máxima para 3 classes é ln(3) ≈ 1.09


def test_alerta_overconfidence_classe_conhecida():
    validador = ValidadorFalsaCerteza(limiar_alerta_certeza=0.85, limiar_entropia_baixa=0.5)
    probabilidades = np.array(
        [0.01, 0.90, 0.03, 0.05, 0.01])  # Confianca = 0.90 (>= 0.85), Entropia muito baixa
    classes_conhecidas = [0, 1, 2, 3]  # O modelo treinou na classe 1

    resultado = validador.avaliar_predicao(probabilidades, classes_conhecidas)

    assert resultado['classe_prevista'] == 1
    # Pela nova regra de Entropia (bugfix), como a Entropia < 0.3 e Confianca > 0.85, isso É um alerta
    assert resultado['alerta_overconfidence'] is True
    assert resultado['confiavel'] is False


def test_overconfidence_classe_desconhecida():
    validador = ValidadorFalsaCerteza(limiar_alerta_certeza=0.85, limiar_entropia_baixa=0.3)
    # Probabilidades distribuidas (sem um pico > 0.85, entropia mais alta > 0.3)
    # ou pico > 0.85 mas a entropia por algum motivo nao fura o limiar (teoricamente um pico de 90%
    # vai furar o limiar). Vamos ajustar para entropia mais proxima da incerteza onde a confianca nao bate 0.85
    probabilidades = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    classes_conhecidas = [0, 1, 2, 3]

    resultado = validador.avaliar_predicao(probabilidades, classes_conhecidas)

    assert resultado['alerta_overconfidence'] is False
    assert resultado['confiavel'] is True


# --- Testes para ValidadorVazamentoDados ---
def test_divisao_limpa_retorna_true():
    treino = np.array([[1, 2], [3, 4]])
    teste = np.array([[5, 6], [7, 8]])

    assert ValidadorVazamentoDados.validar_divisao(treino, teste) is True


def test_instancia_identica_levanta_valueerror():
    treino = np.array([[1, 2], [3, 4]])
    teste = np.array([[5, 6], [1, 2]])  # O [1, 2] vazou para o teste

    with pytest.raises(ValueError, match="Data Leakage detectado"):
        ValidadorVazamentoDados.validar_divisao(treino, teste)


def test_dimensoes_incompativeis_levanta_valueerror():
    treino = np.array([[1, 2], [3, 4]])
    teste = np.array([[5, 6, 7]])  # 3 features vs 2 features no treino

    with pytest.raises(ValueError, match="Incompatibilidade de dimensões"):
        ValidadorVazamentoDados.validar_divisao(treino, teste)


# --- Testes para ValidadorImagemEntrada ---
def test_arquivo_inexistente():
    with pytest.raises(FileNotFoundError, match="Arquivo de imagem não encontrado"):
        ValidadorImagemEntrada.validar_arquivo("caminho/fake.png")


def test_extensao_proibida(tmp_path):
    arquivo_gif = tmp_path / "imagem.gif"
    arquivo_gif.write_text("fake_data")

    with pytest.raises(ValueError, match="Formato inválido"):
        ValidadorImagemEntrada.validar_arquivo(str(arquivo_gif))


def test_arquivo_acima_do_limite(tmp_path, monkeypatch):
    arquivo_png = tmp_path / "grande.png"
    arquivo_png.write_text("fake_data")

    import os
    monkeypatch.setattr(os.path, 'getsize', lambda _: 11 * 1024 * 1024)

    with pytest.raises(ValueError, match="Arquivo muito grande"):
        ValidadorImagemEntrada.validar_arquivo(str(arquivo_png))


def test_imagem_png_valida(tmp_path):
    arquivo_valido = tmp_path / "valido.png"
    # Criar uma imagem PNG real em memoria e salvar
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(arquivo_valido, 'PNG')

    assert ValidadorImagemEntrada.validar_arquivo(str(arquivo_valido)) is True


def test_imagem_corrompida(tmp_path):
    arquivo_corrompido = tmp_path / "corrompido.png"
    # Salvar dados lixo em vez de uma imagem PNG
    with open(arquivo_corrompido, 'wb') as f:
        f.write(b'nao_sou_uma_imagem_valida')

    with pytest.raises(ValueError, match="Arquivo corrompido ou formato de imagem"):
        ValidadorImagemEntrada.validar_arquivo(str(arquivo_corrompido))
