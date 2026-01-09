import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
import json
import mlflow
import importlib
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TimeSeriesDataset
from src.model import LSTMModel
from src import config
from scripts import logger

# Importação dinâmica para acessar parâmetros usados no treino
train_module = importlib.import_module("scripts.03_train")
CURRENT_PARAMS = train_module.CURRENT_PARAMS

EXPERIMENT_NAME = config.EXPERIMENT_NAME
RUN_NAME = "Avaliacao_Teste"


def calculate_mape(y_true, y_pred):
    """
    Calcula o Erro Percentual Absoluto Médio (MAPE).

    Adiciona um epsilon (1e-8) ao denominador para evitar erros de
    divisão por zero caso o valor real seja 0.
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def save_metrics_for_api(metrics: dict):
    """
    Persiste as métricas em JSON.

    Este arquivo é lido pelo endpoint GET /model/info da API, permitindo
    que o sistema de monitoramento saiba a performance atual do modelo em produção.
    """
    try:
        with open(config.METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Métricas exportadas para API: {config.METRICS_PATH}")
    except Exception as e:
        logger.error(f"Erro ao salvar metrics.json: {e}")


def evaluate():
    logger.info("=== Iniciando Avaliação em Dados de Teste ===")

    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Modelo não encontrado: {config.MODEL_PATH}")
        return

    # Recupera hiperparâmetros para montar a arquitetura correta
    seq_length = int(CURRENT_PARAMS["seq_length"])
    batch_size = int(CURRENT_PARAMS["batch_size"])
    hidden_size = int(CURRENT_PARAMS["hidden_size"])
    num_layers = int(CURRENT_PARAMS["num_layers"])

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        device = config.DEVICE

        # 1. Carga de Dados e Scaler
        try:
            test_data = np.load(config.TEST_DATA_PATH)
            scaler = joblib.load(config.SCALER_PATH)
        except FileNotFoundError as e:
            logger.error(f"Artefato ausente: {e}")
            return

        test_dataset = TimeSeriesDataset(test_data, seq_length=seq_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 2. Reconstrução do Modelo
        model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()  # Importante: desativa comportamentos de treino (ex: Dropout)

        # 3. Loop de Inferência
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.numpy())

        # 4. Desnormalização (Volta para escala R$)
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        pred_real = scaler.inverse_transform(predictions)
        actual_real = scaler.inverse_transform(actuals)

        # 5. Cálculo de Métricas
        mae = mean_absolute_error(actual_real, pred_real)
        rmse = np.sqrt(mean_squared_error(actual_real, pred_real))
        mape = calculate_mape(actual_real, pred_real)

        logger.info(
            f"Resultados -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%"
        )

        # Log no MLflow
        mlflow.log_param("seq_length_eval", seq_length)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        # Exportação para API
        save_metrics_for_api(
            {"mae": mae, "rmse": rmse, "mape": mape, "last_evaluation": RUN_NAME}
        )

        # 6. Plotagem
        plt.figure(figsize=(12, 6))
        plt.plot(actual_real, label="Real", color="blue", alpha=0.7)
        plt.plot(pred_real, label="Previsto", color="red", alpha=0.7)
        plt.title(f"Previsão vs Real - {config.SYMBOL}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(config.RESULTS_DIR, "prediction_plot.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        logger.info("Gráfico salvo e registrado no MLflow.")


if __name__ == "__main__":
    evaluate()
