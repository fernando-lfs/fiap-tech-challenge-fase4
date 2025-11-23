# scripts/check_artifacts.py

import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler  # Para verificar o tipo

# --- Caminhos (Baseados no script 02_preprocess.py) ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "02_processed")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_scaled.npy")
VALID_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_scaled.npy")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_scaled.npy")

print("--- 1. Verificando o Scaler ---")
try:
    scaler = joblib.load(SCALER_PATH)

    if isinstance(scaler, MinMaxScaler):
        print(f"[SUCESSO] Scaler carregado: {type(scaler)}")
        # Estes são os valores reais que o scaler aprendeu
        print(f"    - Min (Valor Real): {scaler.data_min_}")
        print(f"    - Max (Valor Real): {scaler.data_max_}")
        print(f"    - Range da Escala: {scaler.feature_range}")
    else:
        print(f"[FALHA] Objeto carregado não é um MinMaxScaler: {type(scaler)}")

except FileNotFoundError:
    print(f"[FALHA] Arquivo do scaler não encontrado em: {SCALER_PATH}")
except Exception as e:
    print(f"[FALHA] Erro ao carregar o scaler: {e}")

print("\n--- 2. Verificando os Dados Processados (.npy) ---")

total_registros = 0
shapes = {}

# Loop para carregar e inspecionar cada arquivo .npy
for name, path in [
    ("Treino", TRAIN_PATH),
    ("Validação", VALID_PATH),
    ("Teste", TEST_PATH),
]:
    try:
        data = np.load(path)
        print(f"[SUCESSO] Dados de {name} carregados:")
        print(f"    - Shape (Formato): {data.shape}")
        print(f"    - Valor Mínimo (Escalado): {data.min():.4f}")
        print(f"    - Valor Máximo (Escalado): {data.max():.4f}")

        # Armazena o shape para o cálculo final
        shapes[name] = data.shape[0]
        total_registros += data.shape[0]

    except FileNotFoundError:
        print(f"[FALHA] Arquivo de {name} não encontrado em: {path}")
    except Exception as e:
        print(f"[FALHA] Erro ao carregar {name}: {e}")

print("\n--- 3. Verificando Proporções da Divisão ---")
if total_registros > 0:
    print(f"Total de registros processados: {total_registros}")

    # Calcula as proporções com base no que foi carregado
    p_train = (shapes.get("Treino", 0) / total_registros) * 100
    p_valid = (shapes.get("Validação", 0) / total_registros) * 100
    p_test = (shapes.get("Teste", 0) / total_registros) * 100

    print(f"    - Treino:     {p_train:.1f}% (Esperado: ~70.0%)")
    print(f"    - Validação:  {p_valid:.1f}% (Esperado: ~15.0%)")
    print(f"    - Teste:      {p_test:.1f}% (Esperado: ~15.0%)")
else:
    print("[FALHA] Nenhum dado carregado para verificar proporções.")

print("\n--- Verificação Concluída ---")
