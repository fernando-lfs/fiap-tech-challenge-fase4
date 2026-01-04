import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts import logger

# Adiciona diretório raiz ao path para garantir importações do src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TimeSeriesDataset
from src.model import LSTMModel
# Importação dinâmica dos parâmetros atuais para evitar inconsistência (DRY)
from scripts.03_train import CURRENT_PARAMS 

# --- Configurações de Ambiente ---
EXPERIMENT_NAME = "Experimento_LSTM_CMIG4"
RUN_NAME = "Avaliacao_Teste"

# Caminhos
DATA_DIR = os.path.join("data", "02_processed")
TEST_PATH = os.path.join(DATA_DIR, "test_scaled.npy")
MODEL_PATH = os.path.join("models", "lstm_model.pth")
SCALER_PATH = os.path.join("models", "scaler.joblib")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_mape(y_true, y_pred):
    """Calcula o Erro Percentual Absoluto Médio."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate():
    logger.info("=== Iniciando Avaliação (Parâmetros Dinâmicos) ===")

    # Extração dos parâmetros atuais definidos no módulo de treino
    # Isso garante que a avaliação use a mesma arquitetura salva no .pth
    seq_length = int(CURRENT_PARAMS["seq_length"])
    batch_size = int(CURRENT_PARAMS["batch_size"])
    hidden_size = int(CURRENT_PARAMS["hidden_size"])
    num_layers = int(CURRENT_PARAMS["num_layers"])

    # Configura para logar no mesmo experimento do MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Carga de Dados
        test_data = np.load(TEST_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # O dataset agora recebe o seq_length dinâmico
        test_dataset = TimeSeriesDataset(test_data, seq_length=seq_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 2. Carga do Modelo (Arquitetura instanciada com parâmetros dinâmicos)
        model = LSTMModel(
            input_size=1, 
            hidden_size=hidden_size, 
            num_layers=num_layers
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        # 3. Inferência (Loop de Teste)
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.numpy())

        # 4. Desnormalização (Conversão para valores monetários reais)
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        pred_real = scaler.inverse_transform(predictions)
        actual_real = scaler.inverse_transform(actuals)

        # 5. Cálculo de Métricas
        mae = mean_absolute_error(actual_real, pred_real)
        rmse = np.sqrt(mean_squared_error(actual_real, pred_real))
        mape = calculate_mape(actual_real, pred_real)

        logger.info(
            f"Métricas Finais - MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%"
        )

        # Registro no MLflow para versionamento e comparação posterior
        mlflow.log_param("seq_length_eval", seq_length)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        # 6. Geração do Gráfico de Performance
        plt.figure(figsize=(12, 6))
        plt.plot(actual_real, label="Real", color="blue", alpha=0.7)
        plt.plot(pred_real, label="Previsto", color="red", alpha=0.7)
        plt.title(f"Resultado Final (Teste) - CMIG4 (Window: {seq_length})")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(RESULTS_DIR, "prediction_plot.png")
        plt.savefig(plot_path)

        # Log do gráfico como artefato no servidor de experimentos
        mlflow.log_artifact(plot_path)
        logger.info("Avaliação concluída e registrada no MLflow.")


if __name__ == "__main__":
    evaluate()