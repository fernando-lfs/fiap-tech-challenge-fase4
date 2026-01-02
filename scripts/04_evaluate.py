import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts import logger

# Configuração de caminhos (para importar modules de src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TimeSeriesDataset
from src.model import LSTMModel

# ==========================================
# CONFIGURAÇÕES
# ==========================================
SEQ_LENGTH = 60
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2

DATA_DIR = os.path.join("data", "02_processed")
TEST_PATH = os.path.join(DATA_DIR, "test_scaled.npy")
MODEL_PATH = os.path.join("models", "lstm_model.pth")
SCALER_PATH = os.path.join("models", "scaler.joblib")
RESULTS_DIR = "results"  # Pasta para salvar gráficos

# Cria diretório de resultados se não existir
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_mape(y_true, y_pred):
    """Calcula o Mean Absolute Percentage Error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate():
    logger.info("=== Iniciando Avaliação do Modelo ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Carregar Dados e Artefatos
    logger.info("Carregando dados de teste e scaler...")
    test_data = np.load(TEST_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Criar Dataset e Loader
    test_dataset = TimeSeriesDataset(test_data, seq_length=SEQ_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Carregar o Modelo Treinado
    logger.info(f"Carregando modelo de {MODEL_PATH}...")
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()  # Modo de avaliação (importante!)

    # 3. Inferência (Fazer as previsões)
    predictions = []
    actuals = []

    logger.info("Realizando previsões...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Guardar resultados na CPU como lista
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())

    # 4. Transformação Inversa (Escala 0-1 -> Preço Real)
    # O scaler espera formato 2D (n_samples, n_features)
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    pred_real = scaler.inverse_transform(predictions)
    actual_real = scaler.inverse_transform(actuals)

    # 5. Cálculo das Métricas
    mae = mean_absolute_error(actual_real, pred_real)
    rmse = np.sqrt(mean_squared_error(actual_real, pred_real))
    mape = calculate_mape(actual_real, pred_real)

    logger.info("\n" + "=" * 30)
    logger.info("RESULTADOS FINAIS (DADOS DE TESTE)")
    logger.info("=" * 30)
    logger.info(f"MAE  (Erro Médio Absoluto): R$ {mae:.4f}")
    logger.info(f"RMSE (Raiz Erro Quadrático): R$ {rmse:.4f}")
    logger.info(f"MAPE (Erro Percentual): {mape:.4f}%")
    logger.info("=" * 30)

    # 6. Visualização (Gráfico)
    logger.info("Gerando gráfico comparativo...")
    plt.figure(figsize=(12, 6))
    plt.plot(actual_real, label="Preço Real", color="blue", alpha=0.7)
    plt.plot(pred_real, label="Previsão LSTM", color="red", alpha=0.7)
    plt.title("Previsão de Preços de Ações (Conjunto de Teste)")
    plt.xlabel("Dias")
    plt.ylabel("Preço de Fechamento (R$)")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(RESULTS_DIR, "prediction_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Gráfico salvo em: {plot_path}")
    logger.info("Avaliação Concluída.")


if __name__ == "__main__":
    evaluate()
