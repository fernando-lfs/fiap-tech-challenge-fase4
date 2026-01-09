import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np
import json

# Adiciona o diretório raiz ao path para importar a API corretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app
from src import config


# --- Fixtures (Preparação) ---cls
@pytest.fixture(scope="module")
def client():
    """
    Inicializa o cliente de teste COM o ciclo de vida (lifespan).
    Isso garante que o evento de 'startup' seja disparado, carregando
    o modelo e os artefatos antes da execução dos testes.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_input():
    """
    Gera um input estatisticamente válido (sem Drift).
    Lê o baseline_stats.json para criar dados próximos à média histórica,
    evitando falsos positivos nos testes de predição padrão.
    """
    if os.path.exists(config.STATS_PATH):
        with open(config.STATS_PATH, "r") as f:
            stats = json.load(f)
        # Gera uma lista com valores próximos à média histórica
        mean_val = stats["mean"]
        # Adiciona um pequeno ruído aleatório para não ser uma linha reta perfeita
        return [float(mean_val + np.random.uniform(-0.5, 0.5)) for _ in range(60)]
    else:
        # Fallback se não houver stats (apenas para não quebrar, mas vai dar drift)
        return [float(i) for i in np.random.rand(60) * 10 + 5]


@pytest.fixture(scope="module")
def drift_input():
    """
    Gera um input com valores extremos (Outliers).
    Projetado especificamente para disparar o alerta de Data Drift na API.
    """
    return [float(1000000.0) for _ in range(60)]


# --- Testes de Integração ---


def test_health_check(client):
    """
    Verifica se a API está online e respondendo.
    Valida o Liveness Probe e se os componentes de ML foram carregados.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Se o modelo carregou, status deve ser healthy
    if os.path.exists(config.MODEL_PATH):
        assert data["status"] == "healthy"


def test_prediction_success(client, sample_input):
    """
    Testa o fluxo feliz de predição (Happy Path).
    Verifica se a API aceita inputs válidos e retorna um float,
    sem disparar alertas de drift indevidos.
    """
    if not os.path.exists(config.MODEL_PATH):
        pytest.skip("Modelo não treinado. Pule este teste.")

    payload = {"last_prices": sample_input}
    response = client.post("/predict", json=payload)

    # Debug: Se falhar, mostra o erro retornado pela API
    if response.status_code != 200:
        print(f"Erro API: {response.json()}")

    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)

    # Agora deve ser False, pois estamos enviando dados dentro da média histórica
    if data["drift_warning"] is True:
        print(f"Drift inesperado: {data['drift_details']}")
    assert data["drift_warning"] is False


def test_sample_data_integration_flow(client):
    """
    Testa o fluxo integrado de usabilidade:
    1. Obter dados reais via /sample-data.
    2. Usar esses mesmos dados para realizar uma predição em /predict.
    """
    # 1. Obter dados de amostra
    response_sample = client.get("/sample-data")

    # Se não houver dados processados, pulamos o teste (ambiente limpo)
    if response_sample.status_code == 404:
        pytest.skip("Dados de teste não encontrados (execute o preprocessamento).")

    assert response_sample.status_code == 200
    sample_data = response_sample.json()

    assert "last_prices" in sample_data
    real_prices = sample_data["last_prices"]
    assert len(real_prices) == 60

    # 2. Usar esses dados para prever
    payload = {"last_prices": real_prices}
    response_predict = client.post("/predict", json=payload)

    assert response_predict.status_code == 200
    assert "predicted_price" in response_predict.json()


def test_prediction_drift_detection(client, drift_input):
    """
    Testa se a API detecta corretamente anomalias nos dados (Drift).
    Valida se o campo 'drift_warning' retorna True para inputs extremos.
    """
    if not os.path.exists(config.STATS_PATH):
        pytest.skip("Baseline stats não encontrado. Pule este teste.")

    payload = {"last_prices": drift_input}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["drift_warning"] is True
    assert len(data["drift_details"]) > 0


def test_prediction_invalid_shape(client):
    """
    Testa a validação de contrato (Schema).
    A API deve rejeitar listas que não tenham exatamente 60 elementos.
    """
    # Envia apenas 5 valores em vez de 60
    payload = {"last_prices": [10.0, 11.0, 12.0, 13.0, 14.0]}
    response = client.post("/predict", json=payload)

    # Pydantic retorna 422 Unprocessable Entity quando min_length não é atendido.
    assert response.status_code == 422


def test_prediction_type_error(client):
    """
    Testa a validação de tipos.
    A API deve rejeitar payloads contendo strings onde se espera floats.
    """
    # Lista com string misturada
    payload = {"last_prices": [10.0] * 59 + ["texto_invalido"]}
    response = client.post("/predict", json=payload)

    # FastAPI/Pydantic retorna 422 Unprocessable Entity para erros de tipagem
    assert response.status_code == 422


def test_prediction_zeros(client):
    """
    Testa a robustez matemática do modelo.
    Envia um vetor de zeros para garantir que a normalização/inferência
    não quebre (ex: divisão por zero não tratada), mesmo que o dado seja financeiramente implausível.
    """
    if not os.path.exists(config.MODEL_PATH):
        pytest.skip("Modelo não treinado.")

    payload = {"last_prices": [0.0] * 60}
    response = client.post("/predict", json=payload)

    # Deve responder 200 OK (matematicamente possível)
    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_train_validation_error(client):
    """
    Testa a validação de regras de negócio no endpoint de treino.
    Hiperparâmetros inválidos (ex: learning rate negativo) devem ser rejeitados (400 Bad Request).
    """
    # Learning rate negativo deve ser rejeitado
    payload = {"hyperparameters": {"learning_rate": -0.01}}
    response = client.post("/train", json=payload)

    assert response.status_code == 400
    assert "learning_rate deve ser maior que 0" in response.json()["detail"]


def test_train_trigger_success(client):
    """
    Testa o disparo do treinamento em background (Happy Path).
    Verifica se a API aceita a requisição (202 Accepted) e inicia a task.
    """
    payload = {"hyperparameters": {"num_epochs": 1}}
    response = client.post("/train", json=payload)

    # Se já houver treino rodando (de testes anteriores), pode dar 409
    if response.status_code == 409:
        pytest.skip("Treino já em andamento, pulando teste de trigger.")

    assert response.status_code == 202
    assert "iniciado em background" in response.json()["message"]


def test_model_info_structure(client):
    """
    Verifica se o endpoint de monitoramento retorna a estrutura JSON correta,
    incluindo versão, parâmetros e métricas.
    """
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()

    assert "version" in data
    assert "current_params" in data
    # metrics pode ser None ou dict, mas a chave deve existir
    assert "metrics" in data


def test_config_endpoint(client):
    """
    Testa se o endpoint de configuração retorna os parâmetros atuais corretamente.
    """
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "current_params" in data
