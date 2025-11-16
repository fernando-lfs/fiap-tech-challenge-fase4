import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# --- Configurações ---
SYMBOL = "CMIG4.SA"
FEATURE_COLUMN = "Close"  # Coluna alvo para a previsão
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# TEST_RATIO é 0.15 (calculado automaticamente)

# --- Caminhos (Paths) ---
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "01_raw", f"{SYMBOL}_data_raw.csv")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "02_processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_scaled.npy")
VALID_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_scaled.npy")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_scaled.npy")

# --- Funções ---


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega os dados brutos, define 'Date' como índice e seleciona a coluna de interesse.
    """
    try:
        df = pd.read_csv(
            path,
            index_col="Date",  # A coluna 'Date' é o índice
            parse_dates=True,  # Converte para datetime
        )

        # Seleciona apenas a coluna de interesse
        df_feature = df[[FEATURE_COLUMN]]

        print(f"Dados brutos carregados: {len(df_feature)} registros.")
        return df_feature

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {path}")
        return pd.DataFrame()
    except KeyError as e:
        print(
            f"Erro: Coluna 'Date' ou '{FEATURE_COLUMN}' "
            f"não encontrada em {path}. {e}"
        )
        return pd.DataFrame()
    except Exception as e:
        print(f"Erro inesperado ao carregar dados: {e}")
        return pd.DataFrame()


def preprocess_data(df: pd.DataFrame):
    """
    Divide os dados cronologicamente, normaliza (escala 0-1).
    Salva os dados processados e o scaler.
    """
    if df.empty:
        print("DataFrame vazio. Abortando pré-processamento.")
        return

    # 1. Divisão Cronológica
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_data = df.iloc[:train_end]
    valid_data = df.iloc[train_end:valid_end]
    test_data = df.iloc[valid_end:]

    print(
        f"Divisão: Treino ({len(train_data)}), "
        f"Validação ({len(valid_data)}), "
        f"Teste ({len(test_data)})"
    )

    # 2. Normalização (Scaling)
    # Instancia o scaler, colocando os dados no intervalo [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Treina (fit) o scaler APENAS com os dados de TREINO.
    scaler.fit(train_data)

    # Transforma todos os conjuntos com o scaler treinado.
    train_scaled = scaler.transform(train_data)
    valid_scaled = scaler.transform(valid_data)
    test_scaled = scaler.transform(test_data)

    # 3. Salvamento dos Artefatos
    # Garante que os diretórios de saída existam
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Salvar o scaler.
    # OBS.: Ele será essencial na API para processar novas entradas e inverter a previsão
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvo em: {SCALER_PATH}")

    # Salvamos os dados como arrays numpy (.npy)
    # OBS.: Este formato é eficiente para carregar no PyTorch.
    np.save(TRAIN_PATH, train_scaled)
    np.save(VALID_PATH, valid_scaled)
    np.save(TEST_PATH, test_scaled)

    print(f"Dados processados salvos em: {PROCESSED_DATA_DIR}")


# --- Execução do Script ---

if __name__ == "__main__":
    raw_data = load_data(RAW_DATA_PATH)
    preprocess_data(raw_data)
