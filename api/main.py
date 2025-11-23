import torch
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os

# Adiciona o diretório raiz ao path para conseguir importar 'src.model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LSTMModel

# --- Configurações e Carga de Artefatos ---
app = FastAPI(
    title="Tech Challenge LSTM API",
    description="API para previsão de preços de ações usando Deep Learning (PyTorch)",
    version="1.0.0",
)

# Caminhos dos arquivos (baseados na estrutura do progresso.txt)
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "models/scaler.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variáveis globais para armazenar modelo e scaler carregados
model = None
scaler = None


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
        print(f"[INFO] Scaler carregado de {SCALER_PATH}")

        # 2. Carregar a Arquitetura e os Pesos do Modelo
        # Precisamos instanciar a classe com os mesmos parâmetros do treino
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Importante: Coloca o modelo em modo de avaliação (desativa dropout, etc)
        print(f"[INFO] Modelo carregado de {MODEL_PATH}")

    except Exception as e:
        print(f"[ERRO] Falha ao carregar artefatos: {e}")
        raise e


# --- Definição dos Dados de Entrada ---
class PredictionRequest(BaseModel):
    # O usuário deve enviar uma lista de preços de fechamento recentes
    last_prices: List[float]


# --- Endpoints ---


@app.get("/")
def health_check():
    return {"status": "ok", "message": "API LSTM está online"}


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

    # Validação básica: Precisamos de pelo menos alguns dados para criar a sequência
    # O tamanho ideal depende de como o modelo "olha" para trás, mas LSTMs aceitam tamanhos variáveis.
    if len(input_data) < 10:
        raise HTTPException(
            status_code=400,
            detail="Forneça pelo menos 10 dias de preços históricos para uma previsão precisa.",
        )

    try:
        # 1. Pré-processamento: Normalizar os dados de entrada
        # O scaler espera formato 2D (n_samples, n_features)
        input_array = np.array(input_data).reshape(-1, 1)
        input_scaled = scaler.transform(input_array)

        # 2. Preparação para o PyTorch
        # Formato esperado pela LSTM: (batch_size, sequence_length, input_size)
        # Criamos um batch de tamanho 1
        sequence = (
            torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        # 3. Inferência
        with torch.no_grad():  # Desabilita cálculo de gradiente para economizar memória
            prediction_scaled = model(sequence)

        # 4. Pós-processamento: Desnormalizar o resultado
        # prediction_scaled é um tensor, precisamos converter para numpy
        prediction_scaled_np = prediction_scaled.cpu().numpy()
        prediction_value = scaler.inverse_transform(prediction_scaled_np)

        return {
            "input_days": len(input_data),
            "predicted_price": float(prediction_value[0][0]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
