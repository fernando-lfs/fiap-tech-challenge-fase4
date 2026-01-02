import logging
import sys
import os

# Metadados do Projeto
__app__ = "FIAP Tech Challenge - LSTM CMIG4"
__author__ = "Fernando Luiz Ferreira"
__version__ = "0.1.0"

# Configuração do Log Handler
# Formato: Data/Hora | Nome do Módulo | Nível do Log | Mensagem
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)  # Redireciona para o console/Docker logs
    ],
)

# Criamos uma instância do logger que será herdada pelos outros módulos
logger = logging.getLogger(__app__)
logger.info(f"Inicializando {__app__} v{__version__}...")
