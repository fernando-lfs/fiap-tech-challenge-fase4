# Arquivo: src/config.py
import os
import torch

"""
Módulo de Configuração Centralizada (Single Source of Truth).

Este arquivo detém a responsabilidade única de definir:
1. Caminhos de diretórios e arquivos (evitando hardcoding disperso).
2. Hiperparâmetros padrão do modelo.
3. Configurações de ambiente (CPU/GPU).

Qualquer alteração de caminho ou parâmetro estrutural deve ser feita AQUI.
"""

# --- 1. Configurações de Hardware e Reprodutibilidade ---
# Define automaticamente o dispositivo de aceleração.
# Se uma GPU NVIDIA estiver disponível, o treinamento será migrado para ela.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Semente para geradores de números aleatórios (numpy, torch).
# Essencial para garantir que dois treinos com os mesmos dados gerem o mesmo resultado.
RANDOM_SEED = 42

# --- 2. Definição de Caminhos Absolutos ---
# A base é calculada dinamicamente a partir da localização deste arquivo.
# Isso garante que o código funcione em qualquer máquina ou container Docker.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Diretórios de Dados
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "01_raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "02_processed")

# Diretórios de Artefatos (Modelos, Logs, Resultados)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Criação automática de diretórios essenciais na importação do módulo
# Evita erros de "FileNotFound" ao salvar arquivos pela primeira vez.
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# --- 3. Definição de Arquivos Específicos ---
SYMBOL = "CMIG4.SA"

# Arquivos de Dados
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_data_raw.csv")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_scaled.npy")
VALID_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_scaled.npy")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_scaled.npy")

# Arquivos de Modelo e Estatísticas
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
STATS_PATH = os.path.join(MODELS_DIR, "baseline_stats.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
MODEL_CONFIG_PATH = os.path.join(
    MODELS_DIR, "model_config.json"
)  # Persistência da arquitetura usada no treino

# --- 4. Configurações de Dados ---
FEATURE_COLUMN = "Close"
DATE_START = "2018-01-01"
DATE_END = "2025-12-31"

# Divisão do Dataset (Split)
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# O restante (0.15) será alocado automaticamente para Teste.

# --- 5. Hiperparâmetros Padrão (Deep Learning) ---
# Estes valores são usados caso a API não forneça overrides.
DEFAULT_HYPERPARAMS = {
    "seq_length": 60,  # Tamanho da janela histórica (Look-back)
    "batch_size": 32,  # Quantas amostras processar antes de atualizar pesos
    "hidden_size": 64,  # Capacidade de memória da LSTM
    "num_layers": 2,  # Profundidade da rede
    "learning_rate": 0.001,  # Passo de aprendizado do otimizador
    "num_epochs": 50,  # Quantas vezes passar pelo dataset completo
}

# --- 6. Configurações MLflow ---
EXPERIMENT_NAME = f"Experimento_LSTM_{SYMBOL}"
