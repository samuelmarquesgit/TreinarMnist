"""Teste proposital para validar a proteção de branch (nunca deve ser mergeado)."""


def test_falha_proposital_para_validar_protecao() -> None:
    assert 1 == 2, "Falha proposital: o CI Gate deve bloquear o merge deste PR"
