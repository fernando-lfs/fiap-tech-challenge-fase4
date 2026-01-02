import yfinance as yf
import pandas as pd
import os
from scripts import logger

# --- Configurações Iniciais ---

# Companhia Energética de Minas Gerais - CEMIG
SYMBOL = "CMIG4.SA"

# Datas de início e fim da base
START_DATE = "2018-01-01"
END_DATE = "2025-10-31"

# Caminho para salvar os dados (Data Lake - Raw)
RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "01_raw", f"{SYMBOL}_data_raw.csv"
)

# --- Função Principal de Coleta ---


def download_stock_data(symbol: str, start: str, end: str, output_path: str):
    """
    Baixa os dados históricos de uma ação usando o yfinance e salva em um arquivo CSV.

    Args:
        symbol (str): O ticker da ação.
        start (str): Data de início (YYYY-MM-DD).
        end (str): Data de fim (YYYY-MM-DD).
        output_path (str): Caminho completo para salvar o .csv.
    """
    logger.info("Iniciando download dos dados para {symbol}...")
    logger.info(f"Período: {start} até {end}")

    try:
        # 1. Cria o objeto Ticker
        ticker = yf.Ticker(symbol)

        # 2. Baixa o histórico
        df = ticker.history(start=start, end=end)

        if df.empty:
            print(f"Nenhum dado encontrado para {symbol} no período.")
            return

        # 3. Transforma o índice (Date) em uma coluna regular 'Date'
        df.reset_index(inplace=True)

        # 4. Formata a data (remove fuso horário se houver)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # 5. Remove colunas que não usaremos (ex: 'Dividends')
        cols_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Filtra o DataFrame para manter apenas as colunas existentes na lista
        df_cleaned = df[[col for col in cols_to_keep if col in df.columns]]

        # Garantir que o diretório de saída exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 6. Salvar o CSV limpo (sem índice do pandas)
        df_cleaned.to_csv(output_path, index=False)

        logger.info(f"Dados salvos com sucesso em: {output_path}")
        logger.info(f"Total de {len(df_cleaned)} registros baixados.")

    except Exception as e:
        logger.error(f"Erro ao baixar os dados: {e}")


# --- Execução do Script ---

if __name__ == "__main__":
    download_stock_data(
        symbol=SYMBOL, start=START_DATE, end=END_DATE, output_path=RAW_DATA_PATH
    )
