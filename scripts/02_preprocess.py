import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import json
from scripts import logger

# --- Configurações ---
SYMBOL = "CMIG4.SA"
FEATURE_COLUMN = "Close"  # Coluna alvo para a previsão
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# TEST_RATIO é 0.15

# --- Caminhos (Paths) ---
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "01_raw", f"{SYMBOL}_data_raw.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "02_processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
STATS_PATH = os.path.join(
    MODELS_DIR, "baseline_stats.json"
)  # Novo arquivo de estatísticas
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_scaled.npy")
VALID_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_scaled.npy")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_scaled.npy")


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df_feature = df[[FEATURE_COLUMN]]
        logger.info(f"Dados brutos carregados: {len(df_feature)} registros.")
        return df_feature
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


def save_baseline_stats(df_train: pd.DataFrame):
    """
    Salva estatísticas descritivas dos dados de treino.
    Essencial para detectar Data Drift na API posteriormente.
    """
    stats = {
        "count": int(df_train.count().iloc[0]),
        "mean": float(df_train.mean().iloc[0]),
        "std": float(df_train.std().iloc[0]),
        "min": float(df_train.min().iloc[0]),
        "max": float(df_train.max().iloc[0]),
        "q25": float(df_train.quantile(0.25).iloc[0]),
        "q75": float(df_train.quantile(0.75).iloc[0]),
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Estatísticas de baseline salvas em: {STATS_PATH}")


def preprocess_data(df: pd.DataFrame):
    if df.empty:
        return

    # 1. Divisão Cronológica
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_data = df.iloc[:train_end]
    valid_data = df.iloc[train_end:valid_end]
    test_data = df.iloc[valid_end:]

    logger.info(f"Split realizado. Treino: {len(train_data)} registros.")

    # 2. Salvar Baseline de Estatísticas (NOVO)
    # Devemos salvar as estatísticas DOS DADOS CRUS DE TREINO, antes do scale
    save_baseline_stats(train_data)

    # 3. Normalização
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    valid_scaled = scaler.transform(valid_data)
    test_scaled = scaler.transform(test_data)

    # 4. Salvamento dos Artefatos
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    np.save(TRAIN_PATH, train_scaled)
    np.save(VALID_PATH, valid_scaled)
    np.save(TEST_PATH, test_scaled)

    logger.info("Pré-processamento concluído.")


if __name__ == "__main__":
    raw_data = load_data(RAW_DATA_PATH)
    preprocess_data(raw_data)
