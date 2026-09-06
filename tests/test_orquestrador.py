"""Testes unitários para o OrquestradorMNISTAgente com mocks da Fachada."""

from unittest.mock import MagicMock, patch

from agent.orquestrador_mnist import OrquestradorMNISTAgente


# ── Helpers ────────────────────────────────────────────────────────────────


def _mock_fachada_ok():
    """Fachada simulada que completa todas as fases com sucesso."""
    fachada = MagicMock()
    fachada.avaliar_modelo.return_value = {"acuracia": 0.95, "f1": 0.94}
    return fachada


# ── Testes de inicialização ────────────────────────────────────────────────


def test_inicializacao_estado_padrao():
    """Estado inicial deve ter todos os flags False e lista vazia."""
    agente = OrquestradorMNISTAgente()
    assert agente.estado_execucao["eda_concluido"] is False
    assert agente.estado_execucao["dados_preparados"] is False
    assert agente.estado_execucao["modelos_treinados"] == []
    assert agente.estado_execucao["avaliacao_concluida"] is False
    assert agente.estado_execucao["ood_concluido"] is False


def test_inicializacao_com_configuracoes():
    """Configurações passadas no construtor devem ser preservadas."""
    cfg = {"modelos_selecionados": ["RegressaoLogistica"]}
    agente = OrquestradorMNISTAgente(configuracoes=cfg)
    assert agente.configuracoes == cfg


def test_inicializacao_sem_configuracoes_usa_dict_vazio():
    """Sem configurações o atributo deve ser dict vazio."""
    agente = OrquestradorMNISTAgente()
    assert agente.configuracoes == {}


# ── Testes do pipeline completo ────────────────────────────────────────────


@patch("src.modelos.fabrica_modelos.FabricaModelos")
@patch("src.fachada.FachadaPipelineIA")
def test_executar_plano_completo_sucesso(mock_fachada_cls, mock_fabrica_cls):
    """Pipeline sem erros deve retornar status SUCESSO."""
    fachada = _mock_fachada_ok()
    mock_fachada_cls.return_value = fachada
    mock_fabrica_cls.listar_disponiveis.return_value = ["RegressaoLogistica"]

    agente = OrquestradorMNISTAgente()
    resultado = agente.executar_plano_completo()

    assert resultado["status"] == "SUCESSO"
    assert resultado["erros"] == []
    assert "RegressaoLogistica" in resultado["metricas"]
    fachada.inicializar_dados.assert_called_once()


@patch("src.modelos.fabrica_modelos.FabricaModelos")
@patch("src.fachada.FachadaPipelineIA")
def test_executar_plano_completo_fase1_falha_retorna_falha(mock_fachada_cls, mock_fabrica_cls):
    """Falha na Fase 1 deve encerrar o pipeline com status FALHA."""
    fachada = MagicMock()
    fachada.inicializar_dados.side_effect = RuntimeError("Dados corrompidos")
    mock_fachada_cls.return_value = fachada
    mock_fabrica_cls.listar_disponiveis.return_value = ["RegressaoLogistica"]

    agente = OrquestradorMNISTAgente()
    resultado = agente.executar_plano_completo()

    assert resultado["status"] == "FALHA"
    assert len(resultado["erros"]) == 1
    assert "Fase 1" in resultado["erros"][0]


@patch("src.modelos.fabrica_modelos.FabricaModelos")
@patch("src.fachada.FachadaPipelineIA")
def test_executar_plano_completo_fase2_falha_continua_parcial(mock_fachada_cls, mock_fabrica_cls):
    """Falha no treino de um modelo não deve impedir a avaliação dos demais."""
    fachada = MagicMock()
    mock_fachada_cls.return_value = fachada

    # Primeiro modelo falha, segundo não
    fachada.treinar_modelo.side_effect = [ValueError("Erro no treino"), None]
    fachada.avaliar_modelo.return_value = {"acuracia": 0.90, "f1": 0.89}

    mock_fabrica_cls.listar_disponiveis.return_value = ["ModeloRuim", "RegressaoLogistica"]

    agente = OrquestradorMNISTAgente()
    resultado = agente.executar_plano_completo()

    # Deve ter erros mas também métricas do modelo que funcionou
    assert len(resultado["erros"]) >= 1
    assert any("Fase 2" in e for e in resultado["erros"])


@patch("src.modelos.fabrica_modelos.FabricaModelos")
@patch("src.fachada.FachadaPipelineIA")
def test_executar_plano_completo_todos_modelos_falham_avaliacao(mock_fachada_cls, mock_fabrica_cls):
    """Se avaliação de todos os modelos falhar, status deve ser FALHA."""
    fachada = MagicMock()
    mock_fachada_cls.return_value = fachada
    fachada.avaliar_modelo.side_effect = RuntimeError("Erro na avaliação")
    mock_fabrica_cls.listar_disponiveis.return_value = ["ModeloX"]

    agente = OrquestradorMNISTAgente()
    resultado = agente.executar_plano_completo()

    assert resultado["status"] == "FALHA"


@patch("src.modelos.fabrica_modelos.FabricaModelos")
@patch("src.fachada.FachadaPipelineIA")
def test_executar_plano_completo_usa_modelos_das_configuracoes(mock_fachada_cls, mock_fabrica_cls):
    """Deve usar modelos_selecionados das configurações em vez da fábrica."""
    fachada = _mock_fachada_ok()
    mock_fachada_cls.return_value = fachada
    mock_fabrica_cls.listar_disponiveis.return_value = ["NaoDeveUsarEste"]

    agente = OrquestradorMNISTAgente(
        configuracoes={"modelos_selecionados": ["KNN"]}
    )
    agente.executar_plano_completo()

    # Python avalia o argumento padrão do .get() sempre (eager evaluation),
    # mas o resultado de listar_disponiveis não é usado quando a chave existe.
    # O que importa é que somente "KNN" foi treinado.
    fachada.treinar_modelo.assert_called_once_with("KNN")
    chamadas = [call.args[0] for call in fachada.treinar_modelo.call_args_list]
    assert "NaoDeveUsarEste" not in chamadas


@patch("src.modelos.fabrica_modelos.FabricaModelos")
@patch("src.fachada.FachadaPipelineIA")
def test_executar_plano_completo_sem_modelos_avaliados_benchmark_ignorado(
    mock_fachada_cls, mock_fabrica_cls
):
    """Fase 4 deve ser ignorada se nenhum modelo foi avaliado com sucesso."""
    fachada = MagicMock()
    mock_fachada_cls.return_value = fachada
    fachada.treinar_modelo.side_effect = RuntimeError("treino falhou")
    mock_fabrica_cls.listar_disponiveis.return_value = ["ModeloX"]

    agente = OrquestradorMNISTAgente()
    resultado = agente.executar_plano_completo()

    # Benchmark deve ter sido ignorado (erro acumulado)
    assert any("Fase 4" in e for e in resultado["erros"])
    fachada.executar_benchmark.assert_not_called()


# ── Testes dos helpers privados ────────────────────────────────────────────


def test_montar_resultado_sucesso():
    """Status SUCESSO deve gerar mensagem com contagem de modelos treinados."""
    agente = OrquestradorMNISTAgente()
    agente.estado_execucao["modelos_treinados"] = ["A", "B"]
    resultado = agente._montar_resultado([], {"A": {}, "B": {}}, "SUCESSO")
    assert resultado["status"] == "SUCESSO"
    assert "2" in resultado["mensagem"]
    assert resultado["erros"] == []


def test_montar_resultado_parcial():
    """Status PARCIAL deve mencionar quantidade de erros e modelos avaliados."""
    agente = OrquestradorMNISTAgente()
    resultado = agente._montar_resultado(
        ["erro1", "erro2"], {"ModeloOk": {}}, "PARCIAL"
    )
    assert resultado["status"] == "PARCIAL"
    assert "2" in resultado["mensagem"]  # erros
    assert "1" in resultado["mensagem"]  # modelos avaliados


def test_montar_resultado_falha():
    """Status FALHA deve mencionar quantidade de erros críticos."""
    agente = OrquestradorMNISTAgente()
    resultado = agente._montar_resultado(["fatal"], {}, "FALHA")
    assert resultado["status"] == "FALHA"
    assert "1" in resultado["mensagem"]


def test_fase1_atualiza_estado_em_sucesso():
    """_fase1_dados bem-sucedido deve marcar eda e dados como True."""
    agente = OrquestradorMNISTAgente()
    fachada = MagicMock()
    erros: list = []

    ok = agente._fase1_dados(fachada, erros)

    assert ok is True
    assert agente.estado_execucao["eda_concluido"] is True
    assert agente.estado_execucao["dados_preparados"] is True
    assert erros == []


def test_fase1_captura_excecao_e_retorna_false():
    """_fase1_dados com exceção deve capturar erro e retornar False."""
    agente = OrquestradorMNISTAgente()
    fachada = MagicMock()
    fachada.inicializar_dados.side_effect = IOError("disco cheio")
    erros: list = []

    ok = agente._fase1_dados(fachada, erros)

    assert ok is False
    assert len(erros) == 1
    assert "Fase 1" in erros[0]


def test_fase2_adiciona_modelos_treinados():
    """_fase2_treino deve adicionar modelos ao estado após treino."""
    agente = OrquestradorMNISTAgente()
    fachada = MagicMock()
    erros: list = []

    agente._fase2_treino(fachada, ["ModeloA", "ModeloB"], erros)

    assert "ModeloA" in agente.estado_execucao["modelos_treinados"]
    assert "ModeloB" in agente.estado_execucao["modelos_treinados"]
    assert erros == []


def test_fase3_marca_avaliacao_concluida_se_houver_metricas():
    """_fase3_avaliacao deve marcar avaliacao_concluida se houver resultados."""
    agente = OrquestradorMNISTAgente()
    agente.estado_execucao["modelos_treinados"] = ["ModeloX"]
    fachada = MagicMock()
    fachada.avaliar_modelo.return_value = {"acuracia": 0.9}
    metricas: dict = {}
    erros: list = []

    agente._fase3_avaliacao(fachada, metricas, erros)

    assert agente.estado_execucao["avaliacao_concluida"] is True
    assert "ModeloX" in metricas


def test_fase4_sem_modelos_adiciona_erro():
    """_fase4_benchmark sem modelos avaliados deve adicionar erro de ignorado."""
    agente = OrquestradorMNISTAgente()
    fachada = MagicMock()
    erros: list = []

    agente._fase4_benchmark(fachada, {}, erros)

    assert len(erros) == 1
    assert "Fase 4" in erros[0]
    fachada.executar_benchmark.assert_not_called()
