# Arquivo: src/config.py
import os
import torch

"""
Arquivo de Configuração Centralizada (Single Source of Truth).
Define caminhos absolutos, parâmetros globais e configurações do modelo.
"""

# --- 1. Configurações de Hardware e Reprodutibilidade ---
# Centraliza a decisão de dispositivo (CPU vs GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# --- 2. Definição de Caminhos Absolutos ---
# A base é calculada a partir da localização deste arquivo (src/config.py -> raiz)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Diretórios de Dados
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "01_raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "02_processed")

# Diretórios de Artefatos
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Criação automática de diretórios essenciais
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# --- 3. Definição de Arquivos Específicos ---
# Arquivos de Dados
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_data_raw.csv")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_scaled.npy")
VALID_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_scaled.npy")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_scaled.npy")

# Arquivos de Modelo e Estatísticas
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
STATS_PATH = os.path.join(MODELS_DIR, "baseline_stats.json")

# --- 4. Configurações de Dados ---
# Configurações do Ativo
SYMBOL = "CMIG4.SA"
FEATURE_COLUMN = "Close"
DATE_START = "2018-01-01"
DATE_END = "2025-12-31"

# Divisão do Dataset
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# Test ratio é o restante (0.15)

# --- 5. Hiperparâmetros Padrão (Deep Learning) ---
DEFAULT_HYPERPARAMS = {
    "seq_length": 60,
    "batch_size": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "learning_rate": 0.001,
    "num_epochs": 50,
}

# --- 6. Configurações MLflow ---
EXPERIMENT_NAME = f"Experimento_LSTM_{SYMBOL}"
