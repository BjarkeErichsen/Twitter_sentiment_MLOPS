"""Train the model"""
"""Run the code using: python twitter_sentiments_MLOPS\train_model_sweep_wandb.py in root directory"""

from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from twitter_sentiments_MLOPS.visualizations.visualize import log_confusion_matrix
import hydra.utils as hydra_utils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

#Run this at start
def sweep_config():
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "batch_size": {"values": [96, 128, 192, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120]},  # Discrete values
            "epochs": {"min": 10, "max": 100, "distribution": "int_uniform"},  # Integer range
            "lr": {"values": [0.00001, 0.00003, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
},  # Log scale for learning rate

        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweeptest", entity="twitter_sentiments_mlops")
    return sweep_id

class FCNN_model(nn.Module):
    def __init__(self):
        super(FCNN_model, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LightningModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = FCNN_model()
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        # Initialize metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=4)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=4)
    def forward(self, x):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # x.to(self.device)
        return self.model(x)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        y_idx = torch.argmax(y, dim=1)

        # Update train accuracy
        self.train_accuracy.update(preds, y_idx)


        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        y_idx = torch.argmax(y, dim=1)

        # Update validation accuracy
        self.val_accuracy.update(preds, y_idx)

        self.log("val_loss", loss)
        return loss


    def on_train_epoch_end(self):
        # Compute and log the train accuracy
        train_acc = self.train_accuracy.compute()
        self.log("train_acc", train_acc)

        print(f"Epoch {self.current_epoch}: Train Accuracy: {train_acc}")
        # Reset train accuracy
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Compute and log the validation accuracy
        val_acc = self.val_accuracy.compute()
        self.log("val_acc", val_acc)
        print(f"Epoch {self.current_epoch}: Validation Accuracy: {val_acc}")
        # Reset validation accuracy
        self.val_accuracy.reset()

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load data

        labels_tensor = torch.load("data/processed/labels.pt")#.to(self.device)
        embeddings_tensor = torch.load("data/processed/text_embeddings.pt")#.to(self.device)

        # Split dataset
        train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
            embeddings_tensor, labels_tensor, test_size=0.2, random_state=42
        )

        self.train_dataset = TensorDataset(train_embeddings, train_labels)
        self.val_dataset = TensorDataset(val_embeddings, val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6,persistent_workers=True)

def main():
    # Check if WANDB_API_KEY is set as an environment variable
    api_key = os.getenv('WANDB_API_KEY')
    # If WANDB_API_KEY is provided, use it to log in
    if api_key:
        wandb.login(key=api_key)
    else:
        # Try to use locally stored credentials (wandb will automatically look for it)
        # This will also prompt for login in the terminal if not already logged in
        wandb.login()
    
    wandb.init()
    run_name = wandb.run.name
    wandb_logger = WandbLogger(project="twitter_sentiment_MLOPS", entity="twitter_sentiments_mlops")


    gcs_checkpoint_path = 'gs://bucket_processed_data/models/FCNN'
    # Ensure the GCS filesystem is used by the ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=gcs_checkpoint_path,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    model = LightningModel(learning_rate=wandb.config.lr)
    data_module = LightningDataModule(batch_size=wandb.config.batch_size)
    accelerator ="gpu" if torch.cuda.is_available() else None

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=wandb.config.epochs,
        accelerator=accelerator,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        limit_train_batches=1.0,  # Use the entire training dataset per epoch
        limit_val_batches=1.0  # Use the entire validation dataset per epoch
    )
    trainer.fit(model, datamodule=data_module)

    

if __name__ == "__main__":
    wandb.finish()
    sweep_id = sweep_config()
    wandb.agent(sweep_id, function=main, count=30)
    wandb.finish()


#MisconfigurationException("`ModelCheckpoint(monitor='val_acc')` could not find the monitored key in the returned metrics: ['train_loss', 'val_loss', 'epoch', 'step']. HINT: Did you call `log('val_acc', value)` in the `LightningModule`?")







