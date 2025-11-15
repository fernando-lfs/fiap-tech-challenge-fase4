import yfinance as yf
import pandas as pd
import os

# --- Configurações Iniciais ---

# Companhia Energética de Minas Gerais - CEMIG
SYMBOL = "CMIG4.SA"

# Datas de início e fim da base (mantidas do exemplo)
START_DATE = "2018-01-01"
END_DATE = "2025-10-31"

# Caminho para salvar os dados (Data Lake - Raw)
# (Dentro de C:\fiap-tech-challenge-fase4\data\01_raw\)
RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "01_raw", f"{SYMBOL}_data_raw.csv"
)

# --- Função Principal de Coleta ---


def download_stock_data(symbol: str, start: str, end: str, output_path: str):
    """
    Baixa os dados históricos de uma ação usando o yfinance
    e salva em um arquivo CSV.

    Args:
        symbol (str): O ticker da ação (ex: 'CMIG4.SA').
        start (str): Data de início (YYYY-MM-DD).
        end (str): Data de fim (YYYY-MM-DD).
        output_path (str): Caminho completo para salvar o .csv.
    """
    print(f"Iniciando download dos dados para {symbol}...")

    try:
        # Use a função download para obter os dados
        df = yf.download(symbol, start=start, end=end)

        if df.empty:
            print(f"Nenhum dado encontrado para {symbol} no período.")
            return

        # Garantir que o diretório de saída (data/01_raw) exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salvar os dados brutos em CSV
        df.to_csv(output_path)

        print(f"Dados salvos com sucesso em: {output_path}")
        print(f"Total de {len(df)} registros baixados.")

    except Exception as e:
        print(f"Erro ao baixar os dados: {e}")


# --- Execução do Script ---

if __name__ == "__main__":
    """
    Ponto de entrada do script.
    Isso permite que o script seja executado diretamente.
    """
    download_stock_data(
        symbol=SYMBOL, start=START_DATE, end=END_DATE, output_path=RAW_DATA_PATH
    )
