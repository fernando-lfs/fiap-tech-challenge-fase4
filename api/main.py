import torch
import joblib
import numpy as np
import psutil
import time
import sys
import os
import importlib
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional

# Adiciona o diretório raiz ao path para garantir importação de 'src' e 'api'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel
from api import logger, __app__, __version__

# Importação dinâmica do script de treino
# (Necessário porque o arquivo começa com número "03_train", o que quebra o import padrão)
training_script = importlib.import_module("scripts.03_train")

# --- Configurações da API ---
app = FastAPI(
    title=__app__,
    description="API para previsão de preços de ações com suporte a MLOps",
    version=__version__,
)

# Caminhos dos arquivos
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "models/scaler.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variáveis globais
model = None
scaler = None
training_active = False  # Flag para controle de concorrência simples


# --- Modelos Pydantic ---
class PredictionRequest(BaseModel):
    last_prices: List[float]


class TrainRequest(BaseModel):
    # Dicionário opcional de hiperparâmetros
    hyperparameters: Optional[Dict[str, float]] = None


class ConfigResponse(BaseModel):
    current_params: Dict[str, float]


# --- Middleware de Monitoramento ---
@app.middleware("http")
async def monitor_performance(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )
    return response


# --- Lógica Reutilizável de Carregamento ---
def load_model_logic():
    """
    Função centralizada para carregar/recarregar o modelo e o scaler.
    Utilizada na inicialização e no endpoint de reload.
    """
    global model, scaler

    try:
        # 1. Carregar o Scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Scaler carregado: {SCALER_PATH}")
        else:
            logger.warning(f"Scaler não encontrado em {SCALER_PATH}")

        # 2. Carregar o Modelo
        if os.path.exists(MODEL_PATH):
            # Obtém os parâmetros atuais para instanciar a arquitetura correta
            params = training_script.CURRENT_PARAMS

            model = LSTMModel(
                input_size=1,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
            )
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            logger.info(f"Modelo LSTM carregado: {MODEL_PATH}")
        else:
            logger.warning(f"Modelo não encontrado em {MODEL_PATH}")

    except Exception as e:
        logger.error(f"Falha ao carregar artefatos: {e}")
        raise e


# --- Tarefa de Background ---
def background_train_task(params: dict):
    """Executa o treino em background e gerencia a flag de estado."""
    global training_active
    try:
        training_active = True
        logger.info("Iniciando tarefa de treinamento em background...")
        # Chama a função train do script importado dinamicamente
        training_script.train(override_params=params)
        logger.info("Tarefa de treinamento concluída.")
    except Exception as e:
        logger.error(f"Erro no treinamento em background: {e}")
    finally:
        training_active = False


# --- Eventos de Ciclo de Vida ---
@app.on_event("startup")
def startup_event():
    load_model_logic()


# --- Endpoints ---


@app.get("/")
def root():
    return {
        "app": __app__,
        "version": __version__,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """
    Endpoint 2: Monitoramento.
    Retorna saúde da aplicação, uso de recursos e status de treino.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()

        return {
            "status": "healthy" if model else "degraded",
            "training_active": training_active,
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
    """Endpoint 3: Inferência."""
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")

    input_data = request.last_prices
    if len(input_data) < 10:
        raise HTTPException(
            status_code=400, detail="Forneça ao menos 10 dias de dados."
        )

    try:
        input_array = np.array(input_data).reshape(-1, 1)
        input_scaled = scaler.transform(input_array)
        sequence = (
            torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        with torch.no_grad():
            prediction_scaled = model(sequence)

        prediction_val = scaler.inverse_transform(prediction_scaled.cpu().numpy())
        result_value = float(prediction_val[0][0])

        return {"input_days": len(input_data), "predicted_price": result_value}

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Endpoint 4: Gatilho de Treinamento.
    Roda em background para não bloquear a API.
    """
    if training_active:
        raise HTTPException(status_code=409, detail="Já existe um treino em andamento.")

    params = request.hyperparameters or {}

    # Adiciona a tarefa à fila de execução do FastAPI
    background_tasks.add_task(background_train_task, params)

    return {
        "message": "Treinamento iniciado em background.",
        "params_recebidos": params,
        "status": "processing",
    }


@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Endpoint 5 (Leitura): Visualiza hiperparâmetros atuais."""
    return {"current_params": training_script.CURRENT_PARAMS}


@app.post("/config")
def update_config(request: TrainRequest):
    """Endpoint 5 (Escrita): Atualiza hiperparâmetros padrão."""
    if request.hyperparameters:
        training_script.CURRENT_PARAMS.update(request.hyperparameters)
    return {
        "message": "Parâmetros globais atualizados.",
        "current_params": training_script.CURRENT_PARAMS,
    }


@app.post("/model/reload")
def reload_model():
    """
    Endpoint Extra: Hot Reload.
    Atualiza a API com o novo modelo treinado sem reiniciar o container.
    """
    try:
        load_model_logic()
        return {"message": "Modelo e scaler recarregados com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recarregar: {e}")
