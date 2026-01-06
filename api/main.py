import torch
import joblib
import numpy as np
import psutil
import time
import sys
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel
from src import config
from api import logger, __app__, __version__
# Importação direta do script de treino para acesso aos parâmetros e função de treino
from scripts import "03_train" as training_script

# Variáveis globais de estado
ml_components = {
    "model": None,
    "scaler": None,
    "baseline_stats": None,
    "training_active": False
}

# --- Lógica de Carregamento (Lifespan) ---
def load_artifacts():
    """Carrega os artefatos de ML na memória."""
    try:
        # 1. Carregar Scaler
        if os.path.exists(config.SCALER_PATH):
            ml_components["scaler"] = joblib.load(config.SCALER_PATH)
            logger.info("Scaler carregado com sucesso.")
        
        # 2. Carregar Estatísticas de Drift
        if os.path.exists(config.STATS_PATH):
            with open(config.STATS_PATH, "r") as f:
                ml_components["baseline_stats"] = json.load(f)
            logger.info("Baseline estatístico carregado.")
        else:
            logger.warning("Baseline stats não encontrado. Monitoramento de Drift inativo.")

        # 3. Carregar Modelo
        if os.path.exists(config.MODEL_PATH):
            params = training_script.CURRENT_PARAMS
            model = LSTMModel(
                input_size=1,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
            )
            # Usa config.DEVICE para consistência
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
            model.to(config.DEVICE)
            model.eval()
            ml_components["model"] = model
            logger.info("Modelo LSTM carregado com sucesso.")
        else:
            logger.warning("Arquivo de modelo não encontrado. API em modo degradado.")

    except Exception as e:
        logger.error(f"Erro crítico no carregamento de artefatos: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executado na inicialização
    logger.info("Inicializando componentes de ML...")
    load_artifacts()
    yield
    # Executado no desligamento (limpeza se necessário)
    logger.info("Desligando API...")

app = FastAPI(
    title=__app__,
    description="API LSTM com Monitoramento de Data Drift",
    version=__version__,
    lifespan=lifespan
)

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    last_prices: List[float]

class TrainRequest(BaseModel):
    hyperparameters: Optional[Dict[str, float]] = None

class ConfigResponse(BaseModel):
    current_params: Dict[str, float]

# --- Middleware ---
@app.middleware("http")
async def monitor_performance(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Latency: {process_time:.4f}s"
    )
    return response

# --- Lógica de Negócio ---

def detect_drift(input_data: List[float]) -> Dict:
    stats = ml_components["baseline_stats"]
    if not stats:
        return {"drift": False, "reason": "Baseline not loaded"}

    input_arr = np.array(input_data)
    drift_reasons = []

    # Margem de tolerância de 10%
    margin = 0.10
    limit_max = stats["max"] * (1 + margin)
    limit_min = stats["min"] * (1 - margin)

    if np.max(input_arr) > limit_max:
        drift_reasons.append(f"Input Max ({np.max(input_arr):.2f}) > Histórico ({limit_max:.2f})")

    if np.min(input_arr) < limit_min:
        drift_reasons.append(f"Input Min ({np.min(input_arr):.2f}) < Histórico ({limit_min:.2f})")

    input_std = np.std(input_arr)
    if input_std > (stats["std"] * 3):
        drift_reasons.append("Alta volatilidade detectada (3x superior ao treino).")

    is_drift = len(drift_reasons) > 0
    if is_drift:
        logger.warning(f"DATA DRIFT DETECTADO: {drift_reasons}")

    return {"drift": is_drift, "reasons": drift_reasons}

def background_train_task(params: dict):
    ml_components["training_active"] = True
    try:
        logger.info("Iniciando treino em background...")
        training_script.train(override_params=params)
        # Recarrega o modelo após o treino
        load_artifacts()
    except Exception as e:
        logger.error(f"Erro treino background: {e}")
    finally:
        ml_components["training_active"] = False

# --- Endpoints ---

@app.get("/")
def root():
    return {"app": __app__, "version": __version__, "status": "online"}

@app.get("/health")
def health_check():
    try:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        return {
            "status": "healthy" if ml_components["model"] else "degraded",
            "drift_monitoring": ml_components["baseline_stats"] is not None,
            "training_active": ml_components["training_active"],
            "resources": {"cpu": cpu, "memory": mem.percent},
        }
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}

@app.post("/predict")
def predict_next_day(request: PredictionRequest):
    model = ml_components["model"]
    scaler = ml_components["scaler"]

    if not model or not scaler:
        raise HTTPException(
            status_code=503, detail="Serviço indisponível (Modelo ou Scaler não carregados)."
        )

    input_data = request.last_prices
    expected_length = int(training_script.CURRENT_PARAMS["seq_length"])
    
    if len(input_data) != expected_length:
        raise HTTPException(
            status_code=400,
            detail=f"Esperado {expected_length} preços. Recebido: {len(input_data)}.",
        )

    drift_info = detect_drift(input_data)

    try:
        input_array = np.array(input_data).reshape(-1, 1)
        input_scaled = scaler.transform(input_array)

        sequence = (
            torch.tensor(input_scaled, dtype=torch.float32)
            .unsqueeze(0)
            .to(config.DEVICE)
        )

        with torch.no_grad():
            prediction_scaled = model(sequence)

        prediction_val = scaler.inverse_transform(prediction_scaled.cpu().numpy())
        result = float(prediction_val[0][0])

        return {
            "predicted_price": round(result, 2),
            "drift_warning": drift_info["drift"],
            "drift_details": drift_info["reasons"],
        }

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

@app.post("/train")
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    if ml_components["training_active"]:
        raise HTTPException(status_code=409, detail="Treino já está em andamento.")
    
    params = request.hyperparameters or {}
    background_tasks.add_task(background_train_task, params)
    return {"status": "processing", "message": "Treino iniciado em background."}

@app.get("/config", response_model=ConfigResponse)
def get_config():
    return {"current_params": training_script.CURRENT_PARAMS}

@app.post("/config")
def update_config(request: TrainRequest):
    if request.hyperparameters:
        training_script.CURRENT_PARAMS.update(request.hyperparameters)
    return {"message": "Configuração atualizada.", "current_params": training_script.CURRENT_PARAMS}

@app.post("/model/reload")
def reload_model():
    load_artifacts()
    return {"message": "Artefatos recarregados."}