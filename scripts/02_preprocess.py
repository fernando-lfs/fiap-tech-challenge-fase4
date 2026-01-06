import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import joblib
import json
from scripts import logger

# Adiciona o diretório raiz ao sys.path para permitir importação de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        # Usa a coluna definida na configuração central
        df_feature = df[[config.FEATURE_COLUMN]]
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

    # Usa caminho definido no config
    with open(config.STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Estatísticas de baseline salvas em: {config.STATS_PATH}")


def preprocess_data(df: pd.DataFrame):
    if df.empty:
        return

    # 1. Divisão Cronológica (Usa ratios do config)
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    valid_end = int(n * (config.TRAIN_RATIO + config.VALID_RATIO))

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

    # 4. Salvamento dos Artefatos (Usa caminhos do config)
    # Diretórios já são garantidos pelo config.py, mas mantemos por segurança
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    joblib.dump(scaler, config.SCALER_PATH)
    np.save(config.TRAIN_DATA_PATH, train_scaled)
    np.save(config.VALID_DATA_PATH, valid_scaled)
    np.save(config.TEST_DATA_PATH, test_scaled)

    logger.info("Pré-processamento concluído.")


if __name__ == "__main__":
    # Carrega do caminho definido no config
    raw_data = load_data(config.RAW_DATA_PATH)
    preprocess_data(raw_data)
