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

def test_alerta_overconfidence_classe_desconhecida():
    validador = ValidadorFalsaCerteza(limiar_alerta_certeza=0.85)
    probabilidades = np.array([0.01, 0.01, 0.03, 0.05, 0.90])  # Prediz a classe 4 com 90%
    classes_conhecidas = [0, 1, 2, 3]  # O modelo não treinou na classe 4
    
    resultado = validador.avaliar_predicao(probabilidades, classes_conhecidas)
    
    assert resultado['classe_prevista'] == 4
    assert resultado['alerta_overconfidence'] is True
    assert resultado['confiavel'] is False

def test_overconfidence_classe_conhecida():
    validador = ValidadorFalsaCerteza(limiar_alerta_certeza=0.85)
    probabilidades = np.array([0.01, 0.90, 0.03, 0.05, 0.01])  # Prediz a classe 1 com 90%
    classes_conhecidas = [0, 1, 2, 3]  # O modelo treinou na classe 1
    
    resultado = validador.avaliar_predicao(probabilidades, classes_conhecidas)
    
    assert resultado['classe_prevista'] == 1
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
    
    with pytest.raises(ValueError, match="Alerta de Data Leakage detectado"):
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
    
    # Mockando getsize para simular um arquivo > 10MB
    monkeypatch.setattr(os.path, 'getsize', lambda _: 11 * 1024 * 1024)
    
    with pytest.raises(ValueError, match="Arquivo muito grande"):
        ValidadorImagemEntrada.validar_arquivo(str(arquivo_png))

def test_imagem_png_valida(tmp_path):
    arquivo_png = tmp_path / "valido.png"
    img = Image.new('RGB', (10, 10), color='white')
    img.save(arquivo_png)
    
    assert ValidadorImagemEntrada.validar_arquivo(str(arquivo_png)) is True
