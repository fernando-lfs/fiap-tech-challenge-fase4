import yfinance as yf
import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao sys.path para permitir importação de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from scripts import logger

# --- Função Principal de Coleta ---


def download_stock_data(symbol: str, start: str, end: str, output_path: str):
    """
    Baixa os dados históricos de uma ação e salva em CSV.

    Args:
        symbol (str): O ticker da ação (ex: 'CMIG4.SA').
        start (str): Data de início (YYYY-MM-DD).
        end (str): Data de fim (YYYY-MM-DD).
        output_path (str): Caminho completo para salvar o arquivo bruto.
    """
    logger.info(f"Iniciando download via Yahoo Finance: {symbol}")
    logger.info(f"Janela Temporal: {start} -> {end}")

    try:
        # 1. Cria o objeto Ticker
        ticker = yf.Ticker(symbol)

        # 2. Baixa o histórico
        df = ticker.history(start=start, end=end)

        if df.empty:
            logger.error(
                f"Nenhum dado retornado para {symbol}. Verifique conexão ou ticker."
            )
            return

        # 3. Reset do índice para transformar 'Date' em coluna
        df.reset_index(inplace=True)

        # 4. Padronização de datas (remove timezone awareness se existir)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # 5. Seleção de colunas relevantes
        cols_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_cleaned = df[[col for col in cols_to_keep if col in df.columns]]

        # Garantia de existência do diretório pai
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 6. Persistência
        df_cleaned.to_csv(output_path, index=False)

        logger.info(f"Download concluído. Registros: {len(df_cleaned)}")
        logger.info(f"Arquivo salvo em: {os.path.abspath(output_path)}")

    except Exception as e:
        logger.error(f"Falha crítica no download: {e}")


# --- Execução do Script ---

if __name__ == "__main__":
    # Executa utilizando as configurações centralizadas
    download_stock_data(
        symbol=config.SYMBOL,
        start=config.DATE_START,
        end=config.DATE_END,
        output_path=config.RAW_DATA_PATH,
    )
