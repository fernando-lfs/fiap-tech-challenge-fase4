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


# --- Fixtures (Preparação) ---
@pytest.fixture(scope="module")
def client():
    """
    Inicializa o cliente de teste COM o ciclo de vida (lifespan).
    Isso garante que load_artifacts() seja chamado.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_input():
    """
    Gera um input válido baseado na média histórica para evitar Drift.
    Lê o baseline_stats.json para saber qual é um valor 'normal'.
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
    """Gera um input com valores extremos para forçar o Drift."""
    return [float(1000000.0) for _ in range(60)]


# --- Testes de Integração ---


def test_health_check(client):
    """Verifica se a API está online e respondendo."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Se o modelo carregou, status deve ser healthy
    if os.path.exists(config.MODEL_PATH):
        assert data["status"] == "healthy"


def test_prediction_success(client, sample_input):
    """
    Testa o fluxo feliz de predição.
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


def test_prediction_drift_detection(client, drift_input):
    """
    Testa se a API detecta corretamente anomalias nos dados (Drift).
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
    """Testa se a API rejeita inputs com tamanho incorreto."""
    # Envia apenas 5 valores em vez de 60
    payload = {"last_prices": [10.0, 11.0, 12.0, 13.0, 14.0]}
    response = client.post("/predict", json=payload)

    # CORREÇÃO: O FastAPI retorna 422 (Unprocessable Entity) quando a validação
    # do Pydantic falha (min_length=60), antes mesmo de entrar na rota.
    assert response.status_code == 422
    
    # Opcional: Verificar se a mensagem de erro menciona o tamanho
    assert "last_prices" in str(response.json())


def test_config_endpoint(client):
    """Testa se conseguimos ler a configuração."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "current_params" in data