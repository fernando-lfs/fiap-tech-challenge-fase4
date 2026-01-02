import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from scripts import logger

# Adicionando o diretório raiz ao path para conseguir importar os módulos de 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TimeSeriesDataset
from src.model import LSTMModel

# ==========================================
# CONFIGURAÇÕES E HIPERPARÂMETROS
# ==========================================
# Hiperparâmetros são ajustes manuais que definem como a rede aprende
SEQ_LENGTH = 60  # Tamanho da janela (dias anteriores usados para prever)
BATCH_SIZE = 32  # Quantas amostras processar antes de atualizar os pesos
HIDDEN_SIZE = 64  # Quantidade de neurônios na camada oculta da LSTM
NUM_LAYERS = 2  # Quantidade de camadas LSTM empilhadas
LEARNING_RATE = 0.001  # Taxa de aprendizado (tamanho do passo do otimizador)
NUM_EPOCHS = 50  # Quantas vezes o modelo verá o dataset completo

# Caminhos dos arquivos (baseado no progresso.txt)
DATA_DIR = os.path.join("data", "02_processed")
TRAIN_PATH = os.path.join(DATA_DIR, "train_scaled.npy")
VALID_PATH = os.path.join(DATA_DIR, "valid_scaled.npy")
MODEL_SAVE_PATH = os.path.join("models", "lstm_model.pth")


def train():
    logger.info("=== Iniciando Configuração de Treinamento ===")

    # 1. Verificação de Dispositivo (GPU vs CPU)
    # Se você tiver uma placa NVIDIA configurada com CUDA, o PyTorch usará ela.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de processamento: {device}")

    # 2. Carregamento dos Dados
    logger.info("Carregando dados...")
    try:
        train_data = np.load(TRAIN_PATH)
        valid_data = np.load(VALID_PATH)
    except FileNotFoundError:
        logger.error(
            f"Erro: Arquivos .npy não encontrados em {DATA_DIR}. Execute o preprocessamento primeiro."
        )
        return

    # 3. Preparação dos DataLoaders
    # Transformamos os arrays numpy em Datasets PyTorch prontos para LSTMs
    train_dataset = TimeSeriesDataset(train_data, seq_length=SEQ_LENGTH)
    valid_dataset = TimeSeriesDataset(valid_data, seq_length=SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Tamanho do Treino: {len(train_dataset)} amostras")
    logger.info(f"Tamanho da Validação: {len(valid_dataset)} amostras")

    # 4. Inicialização do Modelo, Loss e Otimizador
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model = model.to(device)  # Move o modelo para a GPU se disponível

    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Loop de Treinamento
    logger.info("\n=== Iniciando Loop de Treinamento ===")
    best_valid_loss = float("inf")  # Para salvar o melhor modelo

    for epoch in range(NUM_EPOCHS):
        # --- FASE DE TREINO ---
        model.train()  # Coloca o modelo em modo de treino (ativa dropout, etc.)
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Resetar gradientes (obrigatório no PyTorch antes de calcular novos)
            optimizer.zero_grad()

            # Forward Pass (Previsão)
            outputs = model(inputs)

            # Calcular Erro
            loss = criterion(outputs, targets)

            # Backward Pass (Cálculo dos Gradientes - "culpa" de cada peso no erro)
            loss.backward()

            # Atualizar Pesos (Otimização)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- FASE DE VALIDAÇÃO ---
        model.eval()  # Coloca o modelo em modo de avaliação (trava pesos)
        valid_loss = 0.0

        with torch.no_grad():  # Desliga o cálculo de gradientes (economiza memória/processamento)
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)

        # Log de progresso a cada 5 épocas
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}"
            )

        # Checkpoint: Salvar o modelo se ele for o melhor até agora
        # Isso evita que fiquemos com um modelo que "decorou" demais (overfitting) no final
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # logger.info(" -> Modelo salvo (Melhor Validação)")

    logger.info("=" * 40)
    logger.info(f"Treinamento concluído! Melhor Valid Loss: {best_valid_loss:.6f}")
    logger.info(f"Modelo salvo em: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
