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
import json
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

# Parâmetros Atuais (Estado global mutável para integração com API)
CURRENT_PARAMS = DEFAULT_PARAMS.copy()


# ==========================================
# LIGHTNING MODULE
# ==========================================
class LSTMLightningModule(pl.LightningModule):
    """
    Wrapper do PyTorch Lightning para orquestrar o ciclo de vida do treinamento.

    Responsabilidades:
    - Encapsular a arquitetura (LSTMModel).
    - Definir a função de perda (MSELoss).
    - Configurar otimizadores (Adam).
    - Implementar os loops de treino e validação.
    """

    def __init__(self, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.save_hyperparameters()  # Registra hiperparâmetros automaticamente no checkpoint
        self.learning_rate = learning_rate

        self.model = LSTMModel(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers
        )
        # MSELoss (Mean Squared Error) é a função de perda padrão para regressão
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """Delega a inferência para o modelo interno."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Executa um passo de otimização (Forward -> Loss)."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log da loss de treino para monitoramento em tempo real
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Executa um passo de validação (Avaliação sem gradientes)."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log da loss de validação (usada para Early Stopping)
        self.log(
            "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """Configura o otimizador Adam."""
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# ==========================================
# PIPELINE DE EXECUÇÃO
# ==========================================
def train(override_params: dict = None):
    """
    Executa o pipeline completo de treinamento (End-to-End).

    Fluxo:
    1. Configuração de sementes (Reprodutibilidade).
    2. Carregamento e preparação dos DataLoaders.
    3. Configuração do MLflow e Callbacks (Checkpoint, Early Stopping).
    4. Loop de treinamento via PyTorch Lightning.
    5. Exportação do modelo final e metadados para uso na API.

    Args:
        override_params (dict, optional): Dicionário com novos hiperparâmetros
                                          (ex: vindos de uma requisição da API).
    """
    # Garante reprodutibilidade fixando seeds do numpy e torch
    pl.seed_everything(config.RANDOM_SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Treino_{timestamp}"

    logger.info(f"=== Iniciando Pipeline de Treinamento: {run_name} ===")

    # 1. Atualiza a variável GLOBAL para que a API (/config) reflita a mudança
    if override_params:
        CURRENT_PARAMS.update(override_params)
        logger.info(f"Parâmetros globais atualizados via API: {override_params}")

    # 2. Usa os parâmetros globais atualizados para o treino local
    params = CURRENT_PARAMS.copy()
    logger.info(f"Hiperparâmetros efetivos: {params}")

    # 3. Carregar Dados
    try:
        if not os.path.exists(config.TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"Dados processados não encontrados em {config.TRAIN_DATA_PATH}. Execute 'scripts.02_preprocess' primeiro."
            )

        train_data = np.load(config.TRAIN_DATA_PATH)
        valid_data = np.load(config.VALID_DATA_PATH)
        logger.info(
            f"Dados carregados. Treino: {len(train_data)} amostras | Validação: {len(valid_data)} amostras."
        )
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
    # Salva o melhor modelo baseado na menor loss de validação
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=config.CHECKPOINTS_DIR,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Para o treino se a loss de validação não melhorar por 'patience' épocas
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
        accelerator="auto",  # Detecta GPU/CPU automaticamente
        devices=1,
        log_every_n_steps=5,
    )

    # 8. Executar Treino
    trainer.fit(model_system, train_loader, valid_loader)

    logger.info(
        f"Treino finalizado. Melhor loss de validação: {checkpoint_callback.best_model_score}"
    )

    # 9. EXPORTAÇÃO PARA API
    # Carrega o melhor checkpoint salvo pelo callback
    best_model = LSTMLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    # Salva apenas os pesos do modelo interno (nn.Module) para ser leve na API
    # Isso desacopla a inferência do PyTorch Lightning
    torch.save(best_model.model.state_dict(), config.MODEL_PATH)
    logger.info(f"Modelo otimizado para inferência salvo em: {config.MODEL_PATH}")

    # Salva a configuração do modelo para persistência (garante que a API saiba a arquitetura ao reiniciar)
    try:
        with open(config.MODEL_CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Configuração de arquitetura salva em: {config.MODEL_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Erro ao salvar configuração do modelo: {e}")

    # Log do artefato final no MLflow
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, config.MODEL_PATH, artifact_path="model_pth"
    )


if __name__ == "__main__":
    train()
