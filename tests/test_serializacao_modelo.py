"""Testes da serialização do artefato do modelo (TM-028)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from pipeline_offline import treinar_e_serializar as tes
from pipeline_offline.preparar_dados import dividir_treino_validacao_teste

ARTEFATO_REAL = tes.DIR_MODELOS / tes.NOME_ARTEFATO


@pytest.fixture(scope="module")
def particoes():
    """600 amostras separáveis: cada classe acende um bloco diferente de pixels."""
    gerador = np.random.default_rng(2)
    y = np.repeat(np.arange(10, dtype=np.int32), 60)
    X = gerador.random((600, 784), dtype=np.float32) * 0.1
    for classe in range(10):
        X[y == classe, classe * 70 : classe * 70 + 70] += 0.9
    return dividir_treino_validacao_teste(np.clip(X, 0, 1), y)


@pytest.fixture(scope="module")
def busca(particoes):
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(tes.PARAMETROS_FIXOS, "max_iter", 30)
        return tes.buscar_hiperparametros(
            particoes, {"hidden_layer_sizes": [(16,)], "alpha": [1e-3]}
        )


@pytest.fixture
def metadados(busca, particoes):
    return tes.montar_metadados(busca, particoes)


def test_metadados_tem_todas_as_chaves_e_sao_json(metadados):
    assert set(tes.CHAVES_METADADOS) <= set(metadados)
    assert metadados["classes"] == list(range(10))
    assert metadados["n_features"] == 784
    assert metadados["hiperparametros"]["hidden_layer_sizes"] == [16]
    assert metadados["amostras"] == {"treino": 420, "validacao": 60, "teste": 120}
    json.dumps(metadados)  # não pode lançar


def test_calcular_sha256_confere_com_hashlib(tmp_path):
    arquivo = tmp_path / "x.bin"
    arquivo.write_bytes(b"mnist" * 1000)
    assert tes.calcular_sha256(arquivo) == hashlib.sha256(b"mnist" * 1000).hexdigest()


def test_serializa_artefato_e_checksum(busca, metadados, tmp_path):
    caminho, caminho_sha = tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    assert caminho == tmp_path / tes.NOME_ARTEFATO
    hash_registrado, nome = caminho_sha.read_text(encoding="utf-8").split()
    assert nome == tes.NOME_ARTEFATO
    assert hash_registrado == tes.calcular_sha256(caminho)


def test_checksum_usa_formato_sha256sum_com_lf(busca, metadados, tmp_path):
    """Formato `<hash>  <arquivo>\\n` com LF puro, aceito por `sha256sum -c` em qualquer SO."""
    _, caminho_sha = tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    conteudo = caminho_sha.read_bytes()
    assert conteudo.endswith(b".joblib\n")
    assert b"\r" not in conteudo
    assert len(conteudo.split(b"  ")[0]) == 64


def test_artefato_recarregado_preve_igual_ao_original(busca, metadados, particoes, tmp_path):
    caminho, _ = tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    artefato = joblib.load(caminho)
    assert set(artefato) == {"pipeline", "metadados"}
    assert artefato["metadados"] == metadados
    np.testing.assert_allclose(
        artefato["pipeline"].predict_proba(particoes.X_validacao),
        busca.pipeline.predict_proba(particoes.X_validacao),
    )


def test_carrega_em_processo_isolado_sem_codigo_do_projeto(busca, metadados, tmp_path):
    """Prova que a API poderá carregar o artefato sem src/ nem pipeline_offline/."""
    caminho, _ = tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    codigo = (
        "import sys, joblib, numpy as np\n"
        f"a = joblib.load(r'{caminho}')\n"
        "proibidos = [m for m in sys.modules if m.split('.')[0] in ('src', 'pipeline_offline')]\n"
        "assert not proibidos, proibidos\n"
        "p = a['pipeline'].predict_proba(np.zeros((1, 784), dtype=np.float32))\n"
        "print(p.shape, a['metadados']['nome_modelo'])\n"
    )
    saida = subprocess.run(
        [sys.executable, "-I", "-c", codigo],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "(1, 10) PerceptronMulticamadas" in saida.stdout


def test_rejeita_metadados_incompletos(busca, metadados, tmp_path):
    incompletos = {k: v for k, v in metadados.items() if k != "versao_sklearn"}
    with pytest.raises(ValueError, match="versao_sklearn"):
        tes.serializar_artefato(busca.pipeline, incompletos, diretorio=tmp_path)
    assert not (tmp_path / tes.NOME_ARTEFATO).exists()


def test_rejeita_artefato_acima_do_limite_e_remove_arquivo(busca, metadados, tmp_path, monkeypatch):
    monkeypatch.setattr(tes, "TAMANHO_MAXIMO_BYTES", 10)
    with pytest.raises(ValueError, match="excede o limite"):
        tes.serializar_artefato(busca.pipeline, metadados, diretorio=tmp_path)
    assert not (tmp_path / tes.NOME_ARTEFATO).exists()


def test_main_rapido_treina_so_a_vencedora_e_serializa(busca, particoes, tmp_path, monkeypatch):
    grades_usadas = []

    def falso_treinar(semente=42, grade=None):
        grades_usadas.append(grade)
        return particoes, busca

    monkeypatch.setattr(tes, "treinar_modelo_campeao", falso_treinar)
    monkeypatch.setattr(tes, "RAIZ_PROJETO", tmp_path)
    monkeypatch.setattr(tes, "DIR_MODELOS", tmp_path / "models")
    tes.main(["--rapido"])
    assert grades_usadas == [tes.GRADE_VENCEDORA_V1]
    assert (tmp_path / "models" / tes.NOME_ARTEFATO).exists()
    assert (tmp_path / "models" / "modelo_mnist_v1.sha256").exists()


# ── Artefato real versionado em models/ (roda no CI, já que o arquivo está no Git) ──


@pytest.mark.skipif(not ARTEFATO_REAL.exists(), reason="artefato real ainda não gerado")
def test_artefato_real_integro_e_dentro_do_limite():
    assert ARTEFATO_REAL.stat().st_size < tes.TAMANHO_MAXIMO_BYTES
    hash_registrado, nome = ARTEFATO_REAL.with_suffix(".sha256").read_text(encoding="utf-8").split()
    assert nome == tes.NOME_ARTEFATO
    assert hash_registrado == tes.calcular_sha256(ARTEFATO_REAL)


@pytest.mark.skipif(not ARTEFATO_REAL.exists(), reason="artefato real ainda não gerado")
def test_artefato_real_tem_metadados_e_preve_10_classes():
    artefato = joblib.load(ARTEFATO_REAL)
    metadados = artefato["metadados"]
    assert set(tes.CHAVES_METADADOS) <= set(metadados)
    assert metadados["versao_modelo"] == tes.VERSAO_MODELO
    assert metadados["acuracia_validacao"] > 0.97
    probabilidades = artefato["pipeline"].predict_proba(np.zeros((2, 784), dtype=np.float32))
    assert probabilidades.shape == (2, 10)
    np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0, rtol=1e-6)


def test_gitignore_libera_o_artefato():
    resultado = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", f"models/{tes.NOME_ARTEFATO}"],
        cwd=Path(tes.RAIZ_PROJETO),
        capture_output=True,
    )
    assert resultado.returncode == 1  # 1 = não ignorado
