"""Testes da configuração da API (TM-030)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from api.core.config import RAIZ_PROJETO, Configuracoes, obter_configuracoes


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch):
    """Isola os testes de variáveis MNIST_API_* e do .env da máquina."""
    for nome in list(__import__("os").environ):
        if nome.startswith("MNIST_API_"):
            monkeypatch.delenv(nome)
    monkeypatch.setitem(Configuracoes.model_config, "env_file", None)
    obter_configuracoes.cache_clear()
    yield
    obter_configuracoes.cache_clear()


def test_padroes_funcionam_sem_nenhuma_variavel():
    config = Configuracoes()
    assert config.versao_api == "2.0.0"
    assert config.limiar_confianca == 0.85
    assert config.tamanho_max_lote == 100
    assert config.tamanho_max_upload_bytes == 2 * 1024 * 1024
    assert config.max_pixels_imagem == 25_000_000
    assert config.habilitar_diagnostico is False


def test_caminhos_padrao_sao_absolutos_dentro_do_projeto_e_existem():
    config = Configuracoes()
    for caminho in (config.caminho_modelo, config.caminho_checksum, config.caminho_metricas):
        assert caminho.is_absolute()
        assert caminho.is_relative_to(RAIZ_PROJETO)
        assert caminho.exists(), caminho


def test_raiz_do_projeto_e_a_pasta_que_contem_api():
    assert (RAIZ_PROJETO / "api" / "main.py").exists()


def test_variavel_de_ambiente_sobrescreve_padrao(monkeypatch):
    monkeypatch.setenv("MNIST_API_LIMIAR_CONFIANCA", "0.9")
    monkeypatch.setenv("MNIST_API_HABILITAR_DIAGNOSTICO", "true")
    config = Configuracoes()
    assert config.limiar_confianca == 0.9
    assert config.habilitar_diagnostico is True


def test_caminho_relativo_do_ambiente_e_resolvido_pela_raiz(monkeypatch):
    monkeypatch.setenv("MNIST_API_CAMINHO_MODELO", "models/outro.joblib")
    assert Configuracoes().caminho_modelo == (RAIZ_PROJETO / "models" / "outro.joblib").resolve()


def test_caminho_absoluto_do_ambiente_e_mantido(monkeypatch, tmp_path):
    destino = tmp_path / "modelo.joblib"
    monkeypatch.setenv("MNIST_API_CAMINHO_MODELO", str(destino))
    assert Configuracoes().caminho_modelo == destino


@pytest.mark.parametrize(
    ("variavel", "valor"),
    [
        ("MNIST_API_LIMIAR_CONFIANCA", "1.5"),
        ("MNIST_API_LIMIAR_CONFIANCA", "0"),
        ("MNIST_API_TAMANHO_MAX_LOTE", "0"),
        ("MNIST_API_VERSAO_API", "v2"),
        ("MNIST_API_NIVEL_LOG", "VERBOSO"),
        ("MNIST_API_TAMANHO_MAX_UPLOAD_BYTES", "texto"),
    ],
)
def test_valores_invalidos_sao_recusados_na_inicializacao(monkeypatch, variavel, valor):
    monkeypatch.setenv(variavel, valor)
    with pytest.raises(ValidationError):
        Configuracoes()


def test_variaveis_sem_prefixo_sao_ignoradas(monkeypatch):
    monkeypatch.setenv("NIVEL_LOG", "DEBUG")
    monkeypatch.setenv("POSTGRES_HOST", "banco")
    assert Configuracoes().nivel_log == "INFO"


def test_obter_configuracoes_reaproveita_a_mesma_instancia():
    assert obter_configuracoes() is obter_configuracoes()
    assert isinstance(obter_configuracoes().caminho_modelo, Path)
