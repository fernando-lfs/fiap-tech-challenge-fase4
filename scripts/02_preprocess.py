import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import joblib
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from scripts import logger


def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV bruto e seleciona a feature alvo."""
    try:
        if not os.path.exists(path):
            logger.error(f"Arquivo bruto não encontrado: {path}")
            return pd.DataFrame()

        df = pd.read_csv(path, index_col="Date", parse_dates=True)

        if config.FEATURE_COLUMN not in df.columns:
            logger.error(f"Coluna alvo '{config.FEATURE_COLUMN}' inexistente.")
            return pd.DataFrame()

        df_feature = df[[config.FEATURE_COLUMN]]

        # Tratamento de nulos
        initial_len = len(df_feature)
        df_feature = df_feature.dropna()
        if len(df_feature) < initial_len:
            logger.warning(
                f"Removidos {initial_len - len(df_feature)} registros nulos/NaN."
            )

        logger.info(f"Dados carregados: {len(df_feature)} linhas.")
        return df_feature
    except Exception as e:
        logger.error(f"Erro na leitura dos dados: {e}")
        return pd.DataFrame()


def save_baseline_stats(df_train: pd.DataFrame):
    """
    Calcula e salva estatísticas descritivas dos dados de TREINO (sem escala).

    IMPORTANTE:
    Este arquivo JSON será consumido pela API para detectar 'Data Drift'.
    Se os dados de entrada na inferência fugirem muito destas estatísticas
    (ex: max, min, std), a API emitirá um alerta de degradação.
    """
    try:
        stats = {
            "count": int(df_train.count().iloc[0]),
            "mean": float(df_train.mean().iloc[0]),
            "std": float(df_train.std().iloc[0]),
            "min": float(df_train.min().iloc[0]),
            "max": float(df_train.max().iloc[0]),
            "q25": float(df_train.quantile(0.25).iloc[0]),
            "q75": float(df_train.quantile(0.75).iloc[0]),
        }

        os.makedirs(os.path.dirname(config.STATS_PATH), exist_ok=True)

        with open(config.STATS_PATH, "w") as f:
            json.dump(stats, f, indent=4)

        logger.info(
            f"Baseline estatístico (Drift Detection) salvo em: {config.STATS_PATH}"
        )
    except Exception as e:
        logger.error(f"Erro ao salvar estatísticas: {e}")


def preprocess_data(df: pd.DataFrame):
    """Executa o pipeline de transformação e salvamento de artefatos."""
    if df.empty:
        logger.error("Dataset vazio. Abortando.")
        return

    # 1. Divisão Cronológica (Time Series Split)
    # Não podemos usar random split em séries temporais para evitar vazamento de futuro.
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    valid_end = int(n * (config.TRAIN_RATIO + config.VALID_RATIO))

    train_data = df.iloc[:train_end]
    valid_data = df.iloc[train_end:valid_end]
    test_data = df.iloc[valid_end:]

    logger.info(
        f"Split -> Treino: {len(train_data)} | Validação: {len(valid_data)} | Teste: {len(test_data)}"
    )

    # 2. Salvar Baseline (usando dados originais de treino)
    save_baseline_stats(train_data)

    # 3. Normalização (Fit apenas no treino!)
    # Redes Neurais convergem mais rápido com dados entre 0 e 1.
    # O scaler é ajustado APENAS no treino para evitar Data Leakage.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    valid_scaled = scaler.transform(valid_data)
    test_scaled = scaler.transform(test_data)

    # 4. Persistência dos Artefatos (.npy e .joblib)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    joblib.dump(scaler, config.SCALER_PATH)
    np.save(config.TRAIN_DATA_PATH, train_scaled)
    np.save(config.VALID_DATA_PATH, valid_scaled)
    np.save(config.TEST_DATA_PATH, test_scaled)

    logger.info("Pré-processamento finalizado. Artefatos prontos para treino.")


if __name__ == "__main__":
    raw_data = load_data(config.RAW_DATA_PATH)
    preprocess_data(raw_data)
