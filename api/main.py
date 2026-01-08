import torch
import joblib
import numpy as np
import pandas as pd  # Adicionado para corrigir o warning do Scaler
import psutil
import time
import sys
import os
import json
import importlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel
from src import config
from api import logger, __app__, __version__

# --- Importação Dinâmica do Script de Treino ---
training_script = importlib.import_module("scripts.03_train")

# --- Variáveis Globais de Estado ---
ml_components = {
    "model": None,
    "scaler": None,
    "baseline_stats": None,
    "training_active": False,
}

# --- Metadados para Documentação (Tags) ---
tags_metadata = [
    {
        "name": "Inference",
        "description": "Endpoints para predição de preços e geração de dados de exemplo.",
    },
    {
        "name": "Training & Tuning",
        "description": "Funcionalidades de retreino do modelo e ajuste de hiperparâmetros.",
    },
    {
        "name": "Monitoring",
        "description": "Health checks, métricas de performance e informações do sistema.",
    },
    {
        "name": "Management",
        "description": "Gerenciamento de configurações e recarga de artefatos (Hot Reload).",
    },
]


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
            logger.warning(
                "Baseline stats não encontrado. Monitoramento de Drift inativo."
            )

        # 3. Carregar Modelo
        if os.path.exists(config.MODEL_PATH):
            params = training_script.CURRENT_PARAMS
            model = LSTMModel(
                input_size=1,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
            )
            # Usa config.DEVICE para consistência
            model.load_state_dict(
                torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            )
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
    description="""
    ## 🚀 API de Previsão de Ações (LSTM) - Tech Challenge
    
    Esta API fornece serviços de Machine Learning para previsão de preços de fechamento de ações (CMIG4).
    
    ### Funcionalidades Principais:
    * **Predição:** Estima o preço do dia seguinte (D+1) com base em uma janela histórica.
    * **Monitoramento:** Detecta *Data Drift* (mudanças no padrão dos dados) em tempo real.
    * **MLOps:** Permite retreino e tuning de hiperparâmetros em background.
    """,
    version=__version__,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)


# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    last_prices: List[float] = Field(
        ...,
        description="Lista contendo exatamente 60 preços de fechamento históricos (float). Atenção, a predição falhará se o tamanho for diferente.",
        min_length=60,
        max_length=60,
    )

    # Configuração para melhorar a usabilidade no Swagger UI
    # AJUSTE: Valores alterados para ~7.0 para evitar Drift Warning falso positivo
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "last_prices": [
                        7.0 + (i * 0.01) for i in range(60)
                    ]  # Gera 60 valores próximos da média histórica (7.0 - 7.6)
                }
            ]
        }
    }


class TrainRequest(BaseModel):
    hyperparameters: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dicionário opcional de hiperparâmetros. Se fornecido, sobrescreve os padrões para o novo treino.",
        examples=[
            {
                "learning_rate": 0.001,
                "num_epochs": 50,
                "hidden_size": 64,
                "batch_size": 32,
            }
        ],
    )

    def validate_params(self):
        """Validação manual adicional se necessário"""
        if self.hyperparameters:
            if self.hyperparameters.get("learning_rate", 1) <= 0:
                raise ValueError("learning_rate deve ser maior que 0")
            if self.hyperparameters.get("num_epochs", 1) < 1:
                raise ValueError("num_epochs deve ser pelo menos 1")


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


# --- Lógica de Negócio Auxiliar ---


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
        drift_reasons.append(
            f"Input Max ({np.max(input_arr):.2f}) > Histórico ({limit_max:.2f})"
        )

    if np.min(input_arr) < limit_min:
        drift_reasons.append(
            f"Input Min ({np.min(input_arr):.2f}) < Histórico ({limit_min:.2f})"
        )

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


@app.get("/", tags=["Monitoring"])
def root():
    """
    **Verifica o status básico da API.**

    Retorna o nome da aplicação, versão e status online.
    """
    return {"app": __app__, "version": __version__, "status": "online"}


@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    **Health Check Completo (Liveness Probe).**

    Utilizado para monitoramento de infraestrutura. Verifica:
    1. Se o modelo está carregado na memória.
    2. Se o monitoramento de Data Drift está ativo (estatísticas carregadas).
    3. Se há um treinamento em andamento.
    4. Consumo de recursos (CPU e Memória).
    """
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


@app.get("/sample-data", tags=["Inference"])
def get_sample_data():
    """
    **Obter Dados Reais de Teste.**

    Retorna os últimos 60 preços de fechamento do dataset de teste (dados reais).

    **Objetivo:** Facilitar o teste manual do endpoint `/predict`.
    O usuário pode copiar o retorno deste endpoint e colar no corpo da requisição de predição.
    """
    try:
        if not os.path.exists(config.TEST_DATA_PATH) or not ml_components["scaler"]:
            raise HTTPException(
                status_code=404, detail="Dados de teste ou Scaler não encontrados."
            )

        # Carrega dados normalizados
        test_data = np.load(config.TEST_DATA_PATH)

        # Pega os últimos 60 pontos
        seq_len = int(training_script.CURRENT_PARAMS["seq_length"])
        if len(test_data) < seq_len:
            raise HTTPException(
                status_code=400, detail="Dados insuficientes para gerar amostra."
            )

        sample_scaled = test_data[-seq_len:]

        # Desnormaliza para valores reais (R$)
        scaler = ml_components["scaler"]
        sample_real = scaler.inverse_transform(sample_scaled).flatten().tolist()

        return {
            "description": "Últimos 60 preços de fechamento do dataset de teste.",
            "last_prices": [round(x, 2) for x in sample_real],
        }
    except Exception as e:
        logger.error(f"Erro ao gerar sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict",
    tags=["Inference"],
    responses={
        200: {"description": "Predição realizada com sucesso."},
        400: {
            "description": "ERRO DE VALIDAÇÃO: Lista de entrada com tamanho incorreto (deve ser 60)."
        },
        503: {"description": "Modelo não carregado (Serviço Indisponível)."},
    },
)
def predict_next_day(request: PredictionRequest):
    """
    **Realizar Predição de Preço (D+1).**

    Recebe uma janela histórica de preços e retorna a previsão para o próximo dia.

    **⚠️ REQUISITO OBRIGATÓRIO:**
    * O corpo da requisição deve conter uma lista `last_prices` com **exatamente 60 valores** numéricos (float).
    * Valores pré-preenchidos estão disponíveis no botão "Try it out" apenas para teste de conectividade.
    * Para um teste real, utilize os dados do endpoint `/sample-data`.

    **Funcionalidades:**
    * Normaliza os dados de entrada.
    * Executa a inferência no modelo LSTM.
    * **Detecta Data Drift:** Analisa se os dados de entrada fogem estatisticamente do padrão de treino.
    """
    model = ml_components["model"]
    scaler = ml_components["scaler"]

    if not model or not scaler:
        raise HTTPException(
            status_code=503,
            detail="Serviço indisponível (Modelo ou Scaler não carregados).",
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
        # CORREÇÃO: Criar DataFrame para evitar UserWarning do sklearn
        # O scaler foi treinado com DataFrame, então espera nomes de colunas.
        input_df = pd.DataFrame(input_data, columns=[config.FEATURE_COLUMN])

        # Transforma usando o DataFrame (mantém nomes das features)
        input_scaled = scaler.transform(input_df)

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


@app.post(
    "/train",
    tags=["Training & Tuning"],
    status_code=202,
    responses={
        202: {"description": "Treinamento iniciado em background."},
        409: {"description": "Já existe um treinamento em andamento."},
    },
)
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    **Iniciar Treinamento e Tuning.**

    Dispara um processo assíncrono (Background Task) para retreinar o modelo.

    **Tuning de Hiperparâmetros:**
    * Você pode enviar novos hiperparâmetros no corpo da requisição (ex: `learning_rate`, `num_epochs`).
    * Se nenhum parâmetro for enviado, o treino usará a configuração padrão.

    **Nota:** O modelo em memória será atualizado automaticamente ao final do treino.
    """
    # Validação lógica extra (além da tipagem)
    try:
        request.validate_params()
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    if ml_components["training_active"]:
        raise HTTPException(status_code=409, detail="Treino já está em andamento.")

    params = request.hyperparameters or {}
    background_tasks.add_task(background_train_task, params)
    return {"status": "processing", "message": "Treino/Tuning iniciado em background."}


@app.get("/config", tags=["Management"], response_model=ConfigResponse)
def get_config():
    """
    **Consultar Configuração Atual.**

    Retorna os hiperparâmetros que estão sendo utilizados pelo modelo carregado atualmente.
    """
    return {"current_params": training_script.CURRENT_PARAMS}


@app.get("/model/info", tags=["Monitoring"])
def get_model_info():
    """
    **Informações Detalhadas do Modelo.**

    Retorna metadados sobre o modelo em produção, incluindo:
    * Versão da API.
    * Hiperparâmetros atuais.
    * **Métricas de Performance (MAE, RMSE):** Obtidas da última avaliação realizada no conjunto de teste.
    """
    info = {
        "version": __version__,
        "current_params": training_script.CURRENT_PARAMS,
        "metrics": None,
    }

    # Tenta carregar métricas salvas pelo script de avaliação
    if os.path.exists(config.METRICS_PATH):
        try:
            with open(config.METRICS_PATH, "r") as f:
                info["metrics"] = json.load(f)
        except Exception as e:
            logger.warning(f"Falha ao ler métricas: {e}")
            info["metrics_error"] = "Não foi possível ler metrics.json"
    else:
        info["metrics_status"] = (
            "Métricas não disponíveis (Execute o script 04_evaluate.py)"
        )

    return info


@app.post("/config", tags=["Management"])
def update_config(request: TrainRequest):
    """
    **Atualizar Configuração Global.**

    Atualiza os hiperparâmetros na memória sem disparar um treinamento imediato.
    Útil para preparar uma configuração antes de chamar o endpoint `/train`.
    """
    if request.hyperparameters:
        training_script.CURRENT_PARAMS.update(request.hyperparameters)
    return {
        "message": "Configuração atualizada.",
        "current_params": training_script.CURRENT_PARAMS,
    }


@app.post("/model/reload", tags=["Management"])
def reload_model():
    """
    **Hot Reload de Artefatos.**

    Força o recarregamento do modelo (`.pth`) e do scaler (`.joblib`) do disco para a memória.
    Útil caso você tenha substituído os arquivos manualmente e queira atualizar a API sem reiniciar o container.
    """
    load_artifacts()
    return {"message": "Artefatos recarregados."}
