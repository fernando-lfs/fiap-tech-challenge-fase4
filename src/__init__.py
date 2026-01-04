# Expõe as classes principais para facilitar o acesso externo
from .model import LSTMModel
from .dataset import TimeSeriesDataset

# Define o que será exportado no uso de "from src import *"
__all__ = ["LSTMModel", "TimeSeriesDataset"]
