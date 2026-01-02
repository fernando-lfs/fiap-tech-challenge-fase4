import torch
import joblib
import numpy as np
import psutil
import time
import sys
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

# Adiciona o diretório raiz ao path para garantir importação de 'src' e 'api'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel

# Importa logger e metadados centralizados do __init__.py
from api import logger, __app__, __version__

# --- Configurações da API ---
app = FastAPI(
    title=__app__,  # Usa o nome definido no __init__.py
    description="API para previsão de preços de ações usando Deep Learning (PyTorch)",
    version=__version__,  # Usa a versão definida no __init__.py
)

# Caminhos dos arquivos
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "models/scaler.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variáveis globais para armazenar modelo e scaler carregados
model = None
scaler = None


# --- Middleware de Monitoramento de Performance ---
@app.middleware("http")
async def monitor_performance(request: Request, call_next):
    """
    Intercepta todas as requisições para medir o tempo de resposta (latência).
    Essencial para identificar gargalos em produção.
    """
    start_time = time.time()

    # Processa a requisição
    response = await call_next(request)

    # Calcula o tempo total gasto
    process_time = time.time() - start_time

    # Registra no log centralizado
    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )

    return response


# --- Carga de Artefatos ---
@app.on_event("startup")
def load_artifacts():
    """
    Carrega o modelo e o scaler ao iniciar a API.
    Isso evita carregar arquivos do disco a cada requisição (Performance).
    """
    global model, scaler

    try:
        # 1. Carregar o Scaler
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler carregado com sucesso de {SCALER_PATH}")

        # 2. Carregar a Arquitetura e os Pesos do Modelo
        # Deve-se instanciar a classe com os mesmos parâmetros do treino
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Importante: Coloca o modelo em modo de avaliação
        logger.info(f"Modelo LSTM carregado com sucesso de {MODEL_PATH}")

    except Exception as e:
        logger.error(f"Falha crítica ao carregar artefatos: {e}")
        # Em produção, pode ser interessante impedir o start se o modelo falhar
        raise e


# --- Definição dos Dados de Entrada ---
class PredictionRequest(BaseModel):
    # O usuário deve enviar uma lista de preços de fechamento recentes
    last_prices: List[float]


# --- Endpoints ---


@app.get("/")
def root():
    """Endpoint raiz para verificação simples."""
    return {
        "app": __app__,
        "version": __version__,
        "message": "Tech Challenge LSTM API está online. Acesse /docs para documentação.",
    }


@app.get("/health")
def health_check():
    """
    Monitoramento de Recursos.
    Retorna o estado de saúde da aplicação e consumo de infraestrutura.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()

        return {
            "status": "healthy",
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resources": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_info.percent,
                "memory_available_mb": memory_info.available // (1024 * 1024),
            },
            "components": {
                "model_loaded": model is not None,
                "scaler_loaded": scaler is not None,
            },
        }
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return {"status": "unhealthy", "detail": str(e)}


@app.post("/predict")
def predict_next_day(request: PredictionRequest):
    """
    Recebe uma lista de preços históricos e retorna a previsão do próximo dia.
    """
    if not model or not scaler:
        raise HTTPException(
            status_code=500, detail="Modelo não carregado corretamente."
        )

    input_data = request.last_prices

    # Validação básica
    if len(input_data) < 10:
        raise HTTPException(
            status_code=400,
            detail="Forneça pelo menos 10 dias de preços históricos para uma previsão precisa.",
        )

    try:
        # 1. Pré-processamento: Normalizar os dados de entrada
        input_array = np.array(input_data).reshape(-1, 1)
        input_scaled = scaler.transform(input_array)

        # 2. Preparação para o PyTorch
        sequence = (
            torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        # 3. Inferência
        with torch.no_grad():  # Desabilita cálculo de gradiente para economizar memória
            prediction_scaled = model(sequence)

        # 4. Pós-processamento: Desnormalizar o resultado
        prediction_scaled_np = prediction_scaled.cpu().numpy()
        prediction_value = scaler.inverse_transform(prediction_scaled_np)

        result_value = float(prediction_value[0][0])

        logger.info(
            f"Predição realizada. Input size: {len(input_data)} | Output: {result_value:.2f}"
        )

        return {
            "input_days": len(input_data),
            "predicted_price": result_value,
        }

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
