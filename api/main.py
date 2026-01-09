import torch
import joblib
import numpy as np
import pandas as pd
import psutil
import time
import sys
import os
import json
import importlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel
from src import config
from api import logger, __app__, __version__

# --- Importação Dinâmica e Segura do Script de Treino ---
try:
    training_script = importlib.import_module("scripts.03_train")
except ImportError as e:
    logger.critical(
        f"ERRO CRÍTICO: Não foi possível importar o script de treino. Detalhes: {e}"
    )

    # Cria um mock para não quebrar a API inteira, mas funcionalidades de treino falharão
    class MockScript:
        CURRENT_PARAMS = config.DEFAULT_HYPERPARAMS

        def train(self, *args, **kwargs):
            raise NotImplementedError("Script de treino indisponível.")

    training_script = MockScript()

# --- Constantes Dinâmicas ---
# Garante que a validação da API esteja sempre sincronizada com o Config
SEQ_LEN = int(config.DEFAULT_HYPERPARAMS["seq_length"])

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
        "name": "1. Inference",
        "description": "Endpoints principais para consumo do modelo (Predição) e geração de dados de teste.",
    },
    {
        "name": "2. MLOps & Training",
        "description": "Funcionalidades para ciclo de vida do modelo: Retreino, Tuning e Atualização de Artefatos.",
    },
    {
        "name": "3. Observability",
        "description": "Monitoramento de saúde da aplicação, métricas de performance e status do sistema.",
    },
]


# --- Lógica de Carregamento (Lifespan) ---
def load_artifacts():
    """Carrega os artefatos de ML (Modelo, Scaler, Stats) do disco para a memória."""
    try:
        # 1. Carregar Scaler
        if os.path.exists(config.SCALER_PATH):
            ml_components["scaler"] = joblib.load(config.SCALER_PATH)
            logger.info("Scaler carregado com sucesso.")

        # 2. Carregar Estatísticas de Drift
        if os.path.exists(config.STATS_PATH):
            with open(config.STATS_PATH, "r") as f:
                ml_components["baseline_stats"] = json.load(f)
            logger.info("Baseline estatístico carregado para detecção de Drift.")
        else:
            logger.warning(
                "Baseline stats não encontrado. Monitoramento de Drift estará INATIVO."
            )

        # 3. Carregar Modelo
        if os.path.exists(config.MODEL_PATH):
            # Verifica configuração persistida para instanciar a arquitetura correta
            if os.path.exists(config.MODEL_CONFIG_PATH):
                try:
                    with open(config.MODEL_CONFIG_PATH, "r") as f:
                        saved_params = json.load(f)
                    training_script.CURRENT_PARAMS.update(saved_params)
                    logger.info(f"Configuração do modelo restaurada: {saved_params}")
                except Exception as e:
                    logger.error(f"Erro ao ler config do modelo: {e}. Usando padrões.")

            params = training_script.CURRENT_PARAMS

            model = LSTMModel(
                input_size=1,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
            )

            # Carrega pesos e move para o dispositivo configurado
            model.load_state_dict(
                torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            )
            model.to(config.DEVICE)
            model.eval()  # Modo de avaliação (desativa Dropout, etc.)

            ml_components["model"] = model
            logger.info(
                f"Modelo LSTM carregado com sucesso no dispositivo: {config.DEVICE}"
            )
        else:
            logger.warning(
                "Arquivo de modelo (.pth) não encontrado. API em modo degradado."
            )

    except Exception as e:
        logger.error(f"Erro crítico no carregamento de artefatos: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de Ciclo de Vida da Aplicação.
    Executa a carga de modelos na inicialização e limpeza no desligamento.
    """
    logger.info(f"Inicializando componentes de ML em {config.DEVICE}...")
    load_artifacts()
    yield
    logger.info("Desligando API e liberando recursos...")


# --- Descrição Dinâmica da API ---
api_description = f"""
# 📈 API de Previsão de Ações (LSTM) - Tech Challenge

Bem-vindo à documentação interativa da API de previsão financeira. Este projeto utiliza Deep Learning (LSTM) para prever o fechamento de ações da **CEMIG (CMIG4)**.

## 🌟 Visão Geral das Funcionalidades

*   **Predição Inteligente:** Estima o preço de amanhã (D+1) baseando-se nos últimos **{SEQ_LEN} dias** (Janela Deslizante).
*   **Segurança de Dados (Drift):** O sistema avisa se os dados enviados fugirem do padrão normal de mercado.
*   **MLOps Automatizado:** Permite retreinar o modelo em background sem parar a API.

## 📚 Como usar esta documentação

1.  Comece pelo endpoint **`/sample-data`** para pegar dados reais.
2.  Use esses dados no endpoint **`/predict`** para ver o modelo em ação.
3.  Explore **`/model/info`** para ver a performance técnica (Erro Médio, etc).
"""

app = FastAPI(
    title=__app__,
    description=api_description,
    version=__version__,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)


# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    last_prices: List[float] = Field(
        ...,
        description=(
            f"Lista contendo **EXATAMENTE {SEQ_LEN} preços** de fechamento históricos (float). "
            "Este tamanho é definido dinamicamente pela configuração do projeto (`src/config.py`)."
        ),
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"last_prices": [10.0 + (i * 0.1) for i in range(SEQ_LEN)]}]
        }
    }


class TrainRequest(BaseModel):
    hyperparameters: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dicionário opcional para sobrescrever os hiperparâmetros padrão. Use para Tuning.",
        examples=[
            {
                "learning_rate": 0.0005,
                "num_epochs": 100,
                "hidden_size": 128,
                "batch_size": 64,
            }
        ],
    )

    def validate_params(self):
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
    """Middleware para logar latência de cada requisição."""
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
    """
    Verifica se os dados de entrada desviam significativamente do padrão de treino (Drift).
    Utiliza estatísticas descritivas (Min, Max, Std) salvas durante o pré-processamento.
    """
    stats = ml_components["baseline_stats"]
    if not stats:
        return {"drift": False, "reason": "Baseline not loaded"}

    input_arr = np.array(input_data)
    drift_reasons = []

    # Margem de tolerância de 10% sobre os limites históricos
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

    # Verifica volatilidade excessiva
    input_std = np.std(input_arr)
    if input_std > (stats["std"] * 3):
        drift_reasons.append("Alta volatilidade detectada (3x superior ao treino).")

    is_drift = len(drift_reasons) > 0
    if is_drift:
        logger.warning(f"DATA DRIFT DETECTADO: {drift_reasons}")

    return {"drift": is_drift, "reasons": drift_reasons}


def background_train_task(params: dict):
    """Tarefa de background para executar o treinamento sem bloquear a API."""
    ml_components["training_active"] = True
    try:
        logger.info("Iniciando treino em background...")
        training_script.train(override_params=params)
        # Recarrega o modelo após o treino para atualizar a inferência
        load_artifacts()
    except Exception as e:
        logger.error(f"Erro treino background: {e}")
    finally:
        ml_components["training_active"] = False


# --- Endpoints ---


@app.get("/", tags=["3. Observability"], summary="Verificar status da API")
def root():
    """
    **Endpoint Raiz.**

    Utilizado para verificar se a API está online e acessível.

    **Retorno:**
    *   Nome da Aplicação
    *   Versão Atual
    *   Status: "online"
    """
    return {"app": __app__, "version": __version__, "status": "online"}


@app.get(
    "/health",
    tags=["3. Observability"],
    summary="Health Check Completo (Liveness Probe)",
)
def health_check():
    """
    **Monitoramento de Saúde da Infraestrutura.**

    Este endpoint é utilizado por orquestradores (como Kubernetes ou Docker Healthcheck) para saber se o container está saudável.

    **O que é verificado?**
    1.  **Modelo:** Se o arquivo `.pth` foi carregado corretamente na memória.
    2.  **Drift Monitor:** Se as estatísticas de baseline existem.
    3.  **Recursos:** Consumo atual de CPU e Memória RAM.
    4.  **Status de Treino:** Se há algum job de retreino rodando no momento.

    **Retorno:**
    *   `status`: "healthy" (operacional) ou "degraded" (com problemas).
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


# --- Descrições Dinâmicas para Endpoints ---

sample_data_description = f"""
**Gerador de Dados de Exemplo.**

Recupera os últimos **{SEQ_LEN} dias** de preços reais do dataset de teste, conforme o tamanho da janela configurada.

**Objetivo:**
Facilitar o teste manual do endpoint `/predict`. Você pode copiar o JSON retornado aqui e colar diretamente no corpo da requisição de predição.

**Retorno:**
*   `last_prices`: Lista com preços reais de fechamento.
"""


@app.get(
    "/sample-data",
    tags=["1. Inference"],
    summary="Obter dados reais para teste",
    description=sample_data_description,  # CORREÇÃO: Passando a descrição via parâmetro
)
def get_sample_data():
    try:
        if not os.path.exists(config.TEST_DATA_PATH) or not ml_components["scaler"]:
            raise HTTPException(
                status_code=404, detail="Dados de teste ou Scaler não encontrados."
            )

        test_data = np.load(config.TEST_DATA_PATH)
        # Usa a constante dinâmica SEQ_LEN
        if len(test_data) < SEQ_LEN:
            raise HTTPException(
                status_code=400, detail="Dados insuficientes para gerar amostra."
            )

        sample_scaled = test_data[-SEQ_LEN:]
        scaler = ml_components["scaler"]
        sample_real = scaler.inverse_transform(sample_scaled).flatten().tolist()

        return {
            "description": f"Últimos {SEQ_LEN} preços de fechamento do dataset de teste.",
            "last_prices": [round(x, 2) for x in sample_real],
        }
    except Exception as e:
        logger.error(f"Erro ao gerar sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


predict_description = f"""
**Realizar Inferência (Predição).**

Este é o endpoint principal da aplicação. Ele recebe uma janela histórica de preços e utiliza a rede neural LSTM para prever o fechamento do dia seguinte.

**Regras de Negócio e Fluxo:**
1.  **Validação de Entrada:** O sistema exige estritamente **{SEQ_LEN} valores** (dias). Menos que isso impede a formação da matriz de entrada da rede neural.
2.  **Detecção de Data Drift:** Antes de prever, o sistema compara estatisticamente os dados enviados com os dados usados no treinamento.
    *   Se a volatilidade for muito alta ou os valores fugirem do padrão (Min/Max), um alerta (`drift_warning: true`) é retornado.
3.  **Normalização:** Os dados são convertidos para a escala 0-1 (usando o `MinMaxScaler` salvo).
4.  **Inferência:** O modelo LSTM processa a sequência.
5.  **Desnormalização:** O resultado é convertido de volta para Reais (R$).

**Parâmetros de Entrada:**
*   `last_prices`: Lista de floats (Preços de fechamento).
"""


@app.post(
    "/predict",
    tags=["1. Inference"],
    summary="Prever preço da ação (D+1)",
    description=predict_description,  # CORREÇÃO: Passando a descrição via parâmetro
    response_description="Preço previsto e análise de anomalias.",
    responses={
        200: {
            "description": "Sucesso. Retorna o preço previsto.",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_price": 12.45,
                        "drift_warning": False,
                        "drift_details": [],
                    }
                }
            },
        },
        400: {
            "description": f"Erro de Validação. A lista não possui exatamente {SEQ_LEN} itens."
        },
        503: {"description": "Serviço Indisponível. O modelo não foi carregado."},
    },
)
def predict_next_day(request: PredictionRequest):
    model = ml_components["model"]
    scaler = ml_components["scaler"]

    if not model or not scaler:
        raise HTTPException(
            status_code=503,
            detail="Serviço indisponível (Modelo ou Scaler não carregados).",
        )

    input_data = request.last_prices

    # Validação redundante (além do Pydantic) para garantir integridade lógica
    if len(input_data) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Esperado {SEQ_LEN} preços. Recebido: {len(input_data)}.",
        )

    drift_info = detect_drift(input_data)

    try:
        input_df = pd.DataFrame(input_data, columns=[config.FEATURE_COLUMN])
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
    tags=["2. MLOps & Training"],
    summary="Disparar retreino do modelo",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Treinamento aceito e iniciado em background."},
        409: {"description": "Conflito. Já existe um treinamento em andamento."},
        400: {"description": "Hiperparâmetros inválidos (ex: learning_rate negativo)."},
    },
)
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    **Iniciar Pipeline de Treinamento (Assíncrono).**

    Permite retreinar o modelo sem parar a API. O processo roda em uma *Background Task*.

    **Funcionalidades:**
    *   **Retreino Padrão:** Se o corpo da requisição for vazio, usa os parâmetros originais.
    *   **Hyperparameter Tuning:** Você pode enviar novos valores (ex: `learning_rate`, `hidden_size`) para tentar melhorar a performance do modelo.

    **Comportamento:**
    Ao finalizar o treino, a API atualiza automaticamente o modelo em memória (Hot Reload).
    """
    try:
        request.validate_params()
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    if ml_components["training_active"]:
        raise HTTPException(status_code=409, detail="Treino já está em andamento.")

    params = request.hyperparameters or {}
    background_tasks.add_task(background_train_task, params)
    return {"status": "processing", "message": "Treino/Tuning iniciado em background."}


@app.get(
    "/config",
    tags=["2. MLOps & Training"],
    summary="Consultar hiperparâmetros atuais",
    response_model=ConfigResponse,
)
def get_config():
    """
    **Visualizar Configuração Ativa.**

    Retorna os hiperparâmetros que estão sendo utilizados pelo modelo carregado atualmente na memória.
    Útil para verificar se um Tuning recente surtiu efeito.
    """
    return {"current_params": training_script.CURRENT_PARAMS}


@app.get(
    "/model/info",
    tags=["3. Observability"],
    summary="Métricas de performance do modelo",
)
def get_model_info():
    """
    **Relatório de Performance do Modelo.**

    Exibe métricas técnicas calculadas durante a última etapa de avaliação (Test Set).

    **Métricas Retornadas:**
    *   **MAE (Mean Absolute Error):** Erro médio absoluto em Reais (R$).
    *   **RMSE (Root Mean Square Error):** Penaliza erros maiores.
    *   **MAPE:** Erro percentual (ex: 2% de erro médio).
    """
    info = {
        "version": __version__,
        "current_params": training_script.CURRENT_PARAMS,
        "metrics": None,
    }

    if os.path.exists(config.METRICS_PATH):
        try:
            with open(config.METRICS_PATH, "r") as f:
                info["metrics"] = json.load(f)
        except Exception as e:
            logger.warning(f"Falha ao ler métricas: {e}")
            info["metrics_error"] = "Não foi possível ler metrics.json"
    else:
        info["metrics_status"] = (
            "Métricas não disponíveis (Execute scripts/04_evaluate.py)"
        )

    return info


@app.post(
    "/config", tags=["2. MLOps & Training"], summary="Atualizar configuração global"
)
def update_config(request: TrainRequest):
    """
    **Atualizar Parâmetros (Sem Treino).**

    Atualiza a configuração global na memória. Útil para preparar um conjunto de parâmetros antes de chamar o endpoint `/train`.
    """
    if request.hyperparameters:
        training_script.CURRENT_PARAMS.update(request.hyperparameters)
    return {
        "message": "Configuração atualizada.",
        "current_params": training_script.CURRENT_PARAMS,
    }


@app.post(
    "/model/reload", tags=["2. MLOps & Training"], summary="Hot Reload de artefatos"
)
def reload_model():
    """
    **Forçar Recarregamento.**

    Lê novamente os arquivos `.pth` (Modelo) e `.joblib` (Scaler) do disco.
    Use este endpoint se você substituiu os arquivos de modelo manualmente no servidor e quer que a API os reconheça sem reiniciar o container.
    """
    load_artifacts()
    return {"message": "Artefatos recarregados."}
