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
import json  # Necessário para salvar a config
from datetime import datetime

# Adiciona diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TimeSeriesDataset
from src.model import LSTMModel
from src import config
from scripts import logger

# ==========================================
# CONFIGURAÇÕES
# ==========================================
EXPERIMENT_NAME = config.EXPERIMENT_NAME

# Parâmetros Padrão
DEFAULT_PARAMS = config.DEFAULT_HYPERPARAMS.copy()

# Parâmetros Atuais (Estado global para API)
CURRENT_PARAMS = DEFAULT_PARAMS.copy()


# ==========================================
# LIGHTNING MODULE
# ==========================================
class LSTMLightningModule(pl.LightningModule):
    """
    Wrapper do PyTorch Lightning para organizar o treino.
    """

    def __init__(self, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

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
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
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
    # Garante reprodutibilidade
    pl.seed_everything(config.RANDOM_SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Treino_{timestamp}"

    logger.info(f"=== Iniciando Treinamento: {run_name} ===")

    # 1. Atualiza a variável GLOBAL para que a API (/config) reflita a mudança
    if override_params:
        CURRENT_PARAMS.update(override_params)
        logger.info(f"Parâmetros globais atualizados: {override_params}")

    # 2. Usa os parâmetros globais atualizados para o treino local
    params = CURRENT_PARAMS.copy()

    logger.info(f"Parâmetros em uso: {params}")

    # 3. Carregar Dados
    try:
        if not os.path.exists(config.TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"Dados não encontrados em {config.TRAIN_DATA_PATH}. Execute o preprocessamento."
            )

        train_data = np.load(config.TRAIN_DATA_PATH)
        valid_data = np.load(config.VALID_DATA_PATH)
    except Exception as e:
        logger.error(f"Erro crítico ao carregar dados: {e}")
        return

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

    # 4. Configurar Logger do MLflow
    mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=run_name)

    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=config.CHECKPOINTS_DIR,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", patience=10, verbose=True, mode="min"
    )

    # 6. Inicializar Modelo
    model_system = LSTMLightningModule(
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        learning_rate=float(params["learning_rate"]),
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=int(params["num_epochs"]),
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
    )

    # 8. Executar Treino
    trainer.fit(model_system, train_loader, valid_loader)

    logger.info(f"Melhor loss de validação: {checkpoint_callback.best_model_score}")

    # 9. EXPORTAÇÃO PARA API
    # Carrega o melhor checkpoint
    best_model = LSTMLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    # Salva apenas os pesos do modelo interno (nn.Module) para ser leve na API
    torch.save(best_model.model.state_dict(), config.MODEL_PATH)
    logger.info(f"Modelo compatível com API salvo em: {config.MODEL_PATH}")

    # --- NOVO: Salva a configuração do modelo para persistência ---
    try:
        with open(config.MODEL_CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Configuração do modelo salva em: {config.MODEL_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Erro ao salvar configuração do modelo: {e}")
    # --------------------------------------------------------------

    # Log do artefato final no MLflow
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, config.MODEL_PATH, artifact_path="model_pth"
    )


if __name__ == "__main__":
    train()
