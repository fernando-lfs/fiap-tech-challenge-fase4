import torch
import joblib
import numpy as np
import psutil
import time
import sys
import os
import importlib
import json
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel
from src import config  # Importação das configurações centralizadas
from api import logger, __app__, __version__

training_script = importlib.import_module("scripts.03_train")

app = FastAPI(
    title=__app__,
    description="API LSTM com Monitoramento de Data Drift",
    version=__version__,
)

# Caminhos (Agora vêm do config.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variáveis globais
model = None
scaler = None
baseline_stats = None  # Armazena estatísticas de treino
training_active = False


# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    last_prices: List[float]


class TrainRequest(BaseModel):
    hyperparameters: Optional[Dict[str, float]] = None


class ConfigResponse(BaseModel):
    current_params: Dict[str, float]


class DriftReport(BaseModel):
    is_drift: bool
    drift_details: Dict[str, str]


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
def load_model_logic():
    global model, scaler, baseline_stats
    try:
        # Usa caminhos do config
        if os.path.exists(config.SCALER_PATH):
            scaler = joblib.load(config.SCALER_PATH)
            logger.info("Scaler carregado.")

        if os.path.exists(config.STATS_PATH):
            with open(config.STATS_PATH, "r") as f:
                baseline_stats = json.load(f)
            logger.info("Baseline estatístico carregado para monitoramento de Drift.")
        else:
            logger.warning(
                "Baseline stats não encontrado. Monitoramento de Drift inativo."
            )

        if os.path.exists(config.MODEL_PATH):
            params = training_script.CURRENT_PARAMS
            model = LSTMModel(
                input_size=1,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
            )
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            logger.info("Modelo carregado.")
    except Exception as e:
        logger.error(f"Erro no carregamento: {e}")
        raise e


def detect_drift(input_data: List[float]) -> Dict:
    """
    Verifica se os dados de entrada desviam significativamente do baseline de treino.
    Regras simples de Drift:
    1. Out-of-Bounds: Valores maiores que o máx ou menores que o min do treino.
    2. Volatility Shift: Desvio padrão da entrada muito superior ao do treino.
    """
    if not baseline_stats:
        return {"drift": False, "reason": "Baseline not loaded"}

    input_arr = np.array(input_data)
    drift_reasons = []

    # Checagem 1: Extremos (Novos máximos ou mínimos históricos)
    # Adicionamos uma margem de 10% para não alertar ruídos pequenos
    margin = 0.10
    limit_max = baseline_stats["max"] * (1 + margin)
    limit_min = baseline_stats["min"] * (1 - margin)

    if np.max(input_arr) > limit_max:
        drift_reasons.append(
            f"Input Max ({np.max(input_arr):.2f}) excede baseline histórico."
        )

    if np.min(input_arr) < limit_min:
        drift_reasons.append(
            f"Input Min ({np.min(input_arr):.2f}) abaixo do baseline histórico."
        )

    # Checagem 2: Volatilidade (O mercado está muito mais arisco que no treino?)
    input_std = np.std(input_arr)
    # Se a volatilidade atual for 2x maior que a média histórica, alerta.
    if input_std > (baseline_stats["std"] * 3):
        drift_reasons.append("Alta volatilidade detectada (3x superior ao treino).")

    is_drift = len(drift_reasons) > 0
    if is_drift:
        logger.warning(f"DATA DRIFT DETECTADO: {drift_reasons}")

    return {"drift": is_drift, "reasons": drift_reasons}


def background_train_task(params: dict):
    global training_active
    try:
        training_active = True
        training_script.train(override_params=params)
    except Exception as e:
        logger.error(f"Erro treino background: {e}")
    finally:
        training_active = False


@app.on_event("startup")
def startup_event():
    load_model_logic()


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
            "status": "healthy" if model else "degraded",
            "drift_monitoring": baseline_stats is not None,
            "resources": {"cpu": cpu, "memory": mem.percent},
        }
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}


@app.post("/predict")
def predict_next_day(request: PredictionRequest):
    """
    Realiza a predição do preço para o próximo dia com base no histórico enviado.
    Realiza validação de janela temporal e monitoramento de Data Drift.
    """
    if not model or not scaler:
        raise HTTPException(
            status_code=503, detail="Serviço indisponível (artefatos não carregados)."
        )

    input_data = request.last_prices

    # --- OTIMIZAÇÃO: Validação de Janela Temporal ---
    # Recupera o seq_length esperado diretamente dos parâmetros de treino
    expected_length = int(training_script.CURRENT_PARAMS["seq_length"])
    current_length = len(input_data)

    if current_length != expected_length:
        logger.error(
            f"Tamanho de entrada inválido: {current_length}. Esperado: {expected_length}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"O modelo exige exatamente {expected_length} preços históricos para prever o próximo dia. Você enviou {current_length}.",
        )

    # 1. Monitoramento de Drift (Identifica se os dados saíram do padrão de treino)
    drift_info = detect_drift(input_data)

    try:
        # 2. Pipeline de Predição: Escalonamento -> Tensor -> Inferência -> Inversa
        input_array = np.array(input_data).reshape(-1, 1)
        input_scaled = scaler.transform(input_array)

        # Converte para tensor e adiciona a dimensão de Batch (1, Seq, Feature)
        sequence = (
            torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        with torch.no_grad():  # Desabilita gradientes para economizar memória em produção
            prediction_scaled = model(sequence)

        # Converte a saída normalizada de volta para o valor em Reais (BRL)
        prediction_val = scaler.inverse_transform(prediction_scaled.cpu().numpy())
        result = float(prediction_val[0][0])

        # Retorna o resultado predito juntamente com o status de saúde dos dados (Drift)
        return {
            "predicted_price": round(result, 2),
            "window_used": expected_length,
            "drift_warning": drift_info["drift"],
            "drift_details": drift_info["reasons"],
        }

    except Exception as e:
        logger.error(f"Erro interno no pipeline de predição: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar a predição.")


@app.post("/train")
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    if training_active:
        raise HTTPException(status_code=409, detail="Treino em andamento.")
    params = request.hyperparameters or {}
    background_tasks.add_task(background_train_task, params)
    return {"status": "processing", "message": "Treino iniciado."}


@app.get("/config", response_model=ConfigResponse)
def get_config():
    return {"current_params": training_script.CURRENT_PARAMS}


@app.post("/config")
def update_config(request: TrainRequest):
    if request.hyperparameters:
        training_script.CURRENT_PARAMS.update(request.hyperparameters)
    return {"message": "Atualizado.", "current_params": training_script.CURRENT_PARAMS}


@app.post("/model/reload")
def reload_model():
    load_model_logic()
    return {"message": "Recarregado."}
