import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Classe personalizada para carregar dados de séries temporais
    e organizá-los em janelas deslizantes (sliding windows) para LSTM.
    """

    def __init__(self, data: np.ndarray, seq_length: int = 60):
        """
        Args:
            data (np.ndarray): Array numpy com os dados normalizados (N, 1).
            seq_length (int): Tamanho da janela de sequência (ex: 60 dias).
        """
        # Convertendo para tensor float32 (padrão do PyTorch)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        """
        Retorna o número total de amostras possíveis.
        Se temos 100 dias e usamos 60 dias para prever o próximo,
        temos 100 - 60 = 40 amostras.
        """
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        """
        Retorna uma tupla (sequência, alvo) para um índice dado.
        """
        # Janela de entrada: do índice i até i + seq_length
        x = self.data[index : index + self.seq_length]

        # Alvo: o valor imediatamente após a janela
        y = self.data[index + self.seq_length]

        return x, y
