import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Modelo LSTM (Long Short-Term Memory) para previsão de séries temporais univariadas.

    A arquitetura consiste em:
    1. Camada LSTM: Processa a sequência temporal e captura dependências de longo prazo.
    2. Camada Linear (Fully Connected): Mapeia o estado oculto final para um único valor contínuo (preço).

    Attributes:
        hidden_size (int): Número de features no estado oculto da LSTM.
        num_layers (int): Número de camadas LSTM empilhadas.
        lstm (nn.LSTM): Módulo LSTM do PyTorch.
        fc (nn.Linear): Camada linear de saída.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        """
        Inicializa a arquitetura da rede neural.

        Args:
            input_size (int): Número de features de entrada por passo de tempo (1 para univariado).
            hidden_size (int): Dimensão do vetor de estado oculto e célula.
            num_layers (int): Quantidade de camadas recorrentes empilhadas.
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Camada LSTM
        # batch_first=True define a entrada como (Batch, Seq_Length, Features)
        # Isso facilita a integração com DataLoaders padrão.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Camada de saída (Linear)
        # Projeta o vetor de características (hidden_size) para 1 dimensão (preço previsto)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define o fluxo de propagação (Forward Pass) dos dados pela rede.

        Args:
            x (torch.Tensor): Tensor de entrada com shape (Batch, Seq_Length, Input_Size).
                              Ex: (32, 60, 1) para batch de 32, janela de 60 dias, 1 feature.

        Returns:
            torch.Tensor: Tensor de saída com shape (Batch, 1), contendo as predições.
        """
        # Inicializando estados ocultos (h0) e estados de célula (c0) com zeros.
        # .to(x.device) garante que os estados sejam criados no mesmo dispositivo (CPU/GPU) que os dados.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Propagação pela LSTM
        # out: contém os estados ocultos de saída para cada passo de tempo. Shape: (Batch, Seq, Hidden)
        # _ : contém os estados finais (h_n, c_n), que não utilizamos diretamente aqui.
        out, _ = self.lstm(x, (h0, c0))

        # Seleção do último passo de tempo (Many-to-One architecture)
        # Queremos prever o próximo valor baseando-se em toda a sequência histórica processada.
        # Shape resultante: (Batch, Hidden)
        out = out[:, -1, :]

        # Passagem pela camada linear para obter o valor escalar final
        # Shape resultante: (Batch, 1)
        out = self.fc(out)

        return out
