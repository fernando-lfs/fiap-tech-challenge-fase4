import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Dataset personalizado para manipulação de séries temporais em Deep Learning.

    Implementa a estratégia de 'Sliding Window' (Janela Deslizante), transformando
    uma série contínua em pares de (Sequência Histórica, Valor Futuro) para
    aprendizado supervisionado.
    """

    def __init__(self, data: np.ndarray, seq_length: int = 60):
        """
        Inicializa o dataset.

        Args:
            data (np.ndarray): Array numpy contendo a série temporal normalizada. Shape: (N, 1).
            seq_length (int): Tamanho da janela de look-back (quantos passos passados o modelo vê).
        """
        # Convertendo para tensor float32 (padrão numérico do PyTorch para redes neurais)
        # .clone().detach() é utilizado para criar uma cópia segura na memória, evitando
        # warnings sobre memória compartilhada e garantindo integridade dos dados.
        self.data = torch.tensor(data, dtype=torch.float32).clone().detach()
        self.seq_length = seq_length

    def __len__(self) -> int:
        """
        Calcula o número total de amostras válidas que podem ser geradas.

        Fórmula: Total_Pontos - Tamanho_Janela
        Exemplo: Se temos 100 dias e janela de 60, podemos gerar 40 sequências de treino.
        """
        if len(self.data) <= self.seq_length:
            return 0
        return len(self.data) - self.seq_length

    def __getitem__(self, index: int):
        """
        Gera uma amostra de treino/teste baseada no índice.

        Args:
            index (int): Ponto de partida da janela deslizante.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x (Features): Sequência de entrada de tamanho `seq_length`.
                - y (Target): O valor real imediatamente após a sequência (index + seq_length).
        """
        # Janela de entrada (Features): do índice i até i + seq_length
        x = self.data[index : index + self.seq_length]

        # Alvo (Target): o valor exato no passo seguinte ao fim da janela
        y = self.data[index + self.seq_length]

        return x, y
