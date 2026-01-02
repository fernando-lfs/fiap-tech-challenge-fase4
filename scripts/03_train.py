import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import mlflow
from scripts import logger

# Adiciona diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TimeSeriesDataset
from src.model import LSTMModel

# ==========================================
# CONFIGURAÇÕES
# ==========================================
EXPERIMENT_NAME = "Experimento_LSTM_CMIG4"
RUN_NAME = "Treino_Lightning_Padrao"

# Parâmetros Padrão (Imutáveis)
DEFAULT_PARAMS = {
    "seq_length": 60,
    "batch_size": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "learning_rate": 0.001,
    "num_epochs": 50,
}

# Parâmetros Atuais (Podem ser alterados pela API em tempo de execução)
CURRENT_PARAMS = DEFAULT_PARAMS.copy()

# Caminhos
DATA_DIR = os.path.join("data", "02_processed")
TRAIN_PATH = os.path.join(DATA_DIR, "train_scaled.npy")
VALID_PATH = os.path.join(DATA_DIR, "valid_scaled.npy")
# Caminho para salvar o modelo compatível com a API
MODEL_SAVE_PATH = os.path.join("models", "lstm_model.pth")


# ==========================================
# LIGHTNING MODULE
# ==========================================
class LSTMLightningModule(pl.LightningModule):
    """
    Wrapper do PyTorch Lightning para organizar o treino.
    """

    def __init__(self, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.save_hyperparameters()  # Salva hparams automaticamente no MLflow/Checkpoint
        self.learning_rate = learning_rate

        # Instancia a arquitetura original (definida em src/model.py)
        self.model = LSTMModel(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log automático (step-level)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log automático (epoch-level)
        self.log(
            "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# ==========================================
# PIPELINE DE EXECUÇÃO
# ==========================================
def train(override_params: dict = None):
    """
    Executa o pipeline de treinamento.
    Args:
        override_params (dict): Dicionário com novos hiperparâmetros vindos da API.
    """
    logger.info("=== Iniciando Treinamento com PyTorch Lightning + MLflow ===")

    # 1. Configuração Dinâmica de Parâmetros
    # Começa com os parâmetros atuais globais
    params = CURRENT_PARAMS.copy()

    # Se a API enviou overrides, atualiza os parâmetros para ESTA execução
    if override_params:
        params.update(override_params)
        logger.info(f"Parâmetros sobrescritos para este run: {override_params}")

    logger.info(f"Parâmetros em uso: {params}")

    # 2. Carregar Dados
    try:
        train_data = np.load(TRAIN_PATH)
        valid_data = np.load(VALID_PATH)
    except FileNotFoundError:
        logger.error("Dados .npy não encontrados.")
        return

    # Usa params['seq_length'] dinamicamente
    train_dataset = TimeSeriesDataset(train_data, seq_length=int(params["seq_length"]))
    valid_dataset = TimeSeriesDataset(valid_data, seq_length=int(params["seq_length"]))

    train_loader = DataLoader(
        train_dataset, batch_size=int(params["batch_size"]), shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    # 3. Configurar Logger do MLflow
    mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME)
    mlf_logger.log_hyperparams(params)  # Registra parâmetros efetivos

    # 4. Callbacks (Boas práticas de Engenharia)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="models/checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", patience=10, verbose=True, mode="min"
    )

    # 5. Inicializar Modelo Lightning (Usa parâmetros dinâmicos)
    model_system = LSTMLightningModule(
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        learning_rate=float(params["learning_rate"]),
    )

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=int(params["num_epochs"]),
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",  # Detecta GPU/CPU automaticamente
        devices=1,
        log_every_n_steps=5,
    )

    # 7. Executar Treino
    trainer.fit(model_system, train_loader, valid_loader)

    logger.info(f"Melhor loss de validação: {checkpoint_callback.best_model_score}")
    logger.info(f"Checkpoint salvo em: {checkpoint_callback.best_model_path}")

    # 8. EXPORTAÇÃO PARA API
    best_model = LSTMLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    torch.save(best_model.model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Modelo compatível com API salvo em: {MODEL_SAVE_PATH}")

    # Log do artefato final no MLflow
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, MODEL_SAVE_PATH, artifact_path="model_pth"
    )


if __name__ == "__main__":
    train()
