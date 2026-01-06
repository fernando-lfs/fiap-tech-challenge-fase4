import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Modelo LSTM para previsão de séries temporais.
    Herda de nn.Module, a classe base para todas as redes no PyTorch.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Camada LSTM
        # batch_first=True faz com que a entrada seja esperada como (Batch, Seq, Feature)
        # Isso facilita o trabalho com DataLoaders que empilham o batch na primeira dimensão.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Camada de saída (Linear)
        # Conecta a saída da LSTM (hidden_size) a 1 única saída (preço previsto)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define o fluxo de dados pela rede (Forward Pass).
        Args:
            x (torch.Tensor): Tensor de entrada com shape (Batch, Seq, Feature)
        """
        # Inicializando estados ocultos (h0) e estados de célula (c0) com zeros
        # Usamos x.device para garantir que os estados estejam na mesma GPU/CPU que os dados
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Propagação pela LSTM
        # out possui as saídas de todos os passos de tempo
        # _ (underscore) ignora os estados ocultos finais retornados (não precisamos deles aqui)
        out, _ = self.lstm(x, (h0, c0))

        # Pegamos apenas a saída do último passo de tempo (out[:, -1, :])
        # Pois queremos prever com base em toda a sequência anterior acumulada
        out = out[:, -1, :]

        # Passamos pela camada linear para obter o valor final
        out = self.fc(out)
        return out
