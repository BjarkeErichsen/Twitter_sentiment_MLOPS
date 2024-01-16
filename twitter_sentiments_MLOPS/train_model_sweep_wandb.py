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
#wandb.login()
def sweep_config():
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "batch_size": {"values": [32, 64, 128]},  # Discrete values
            "epochs": {"min": 10, "max": 100, "distribution": "int_uniform"},  # Integer range
            "lr": {"min": 0.0001, "max": 0.01, "distribution": "uniform"},  # Log scale for learning rate
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
        # Load data
        labels_tensor = torch.load("data/processed/labels.pt")
        embeddings_tensor = torch.load("data/processed/text_embeddings.pt")

        # Split dataset
        train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
            embeddings_tensor, labels_tensor, test_size=0.2, random_state=42
        )

        self.train_dataset = TensorDataset(train_embeddings, train_labels)
        self.val_dataset = TensorDataset(val_embeddings, val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=11,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=11,persistent_workers=True)

def main():
    # Initialize wandb
    wandb.init()
    wandb_logger = WandbLogger(project="twitter_sentiment_MLOPS", entity="twitter_sentiments_mlops")
    checkpoint_callback = ModelCheckpoint(
        dirpath="twitter_sentiments_MLOPS/models/FCNN",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )
    model = LightningModel(learning_rate=wandb.config.lr)
    data_module = LightningDataModule(batch_size=wandb.config.batch_size)

    if torch.cuda.is_available():
        # If CUDA is available, use all available GPUs
        gpus = -1
        print("CUDA is available. Using GPUs.")
    else:
        # If CUDA is not available, do not use GPUs
        gpus = 0
        print("CUDA is not available. Using CPU.")

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=wandb.config.epochs,
        gpus=gpus,
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















"""Legacy code:"""

# sweep_id, sweep_configuration = sweep_config()
#
#
# # Config decorator for Hydra
#
# def main():
#     run = wandb.init()
#     # Initialize Weights & Biases
#     learning_rate = wandb.config.lr
#     batch_size = wandb.config.batch_size
#     epochs = wandb.config.epochs
#     embedding_dim = 768
#     hidden_dim = [128,64,4]
#     # model = SimpleNN(embedding_dim, hidden_dim)
#     model = FCNN_model()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     # Check if CUDA is available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("device", device)
#     # Move model to the chosen device
#     model.to(device)
#
#     # Load data
#      # Now load the data using these paths
#     labels_tensor = torch.load("../data/processed/labels.pt")
#     embeddings_tensor = torch.load("../data/processed/text_embeddings.pt")
#
#     # Split dataset
#     train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
#         embeddings_tensor, labels_tensor, test_size=0.2, random_state=42
#     )
#
#     train_dataset = TensorDataset(train_embeddings, train_labels)
#     val_dataset = TensorDataset(val_embeddings, val_labels)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#
#     # Training and Validation Loop
#     for epoch in range(epochs):
#         # Training
#         model.train()
#         train_loss = 0.0
#         correct_train = 0
#         total_train = 0
#
#         #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         #             record_shapes=True,
#         ##             profile_memory=True,
#         #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/train')) as prof:
#
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             #loss = criterion(outputs, labels.float())
#             loss = criterion(outputs, labels) # for CrossEntropyLoss as loss function
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#             predicted = torch.argmax(outputs, dim=1)
#             labels_idx = torch.argmax(labels, dim=1)
#
#             correct_train += (predicted == labels_idx).sum().item()
#             total_train += batch_size
#
#         train_accuracy = 100 * correct_train / total_train
#         wandb.log({"train_loss": train_loss / len(train_loader), "train_accuracy": train_accuracy, "epochs": epoch})
#
#         # Validation
#         all_labels = []
#         all_predictions = []
#         model.eval()
#         val_loss = 0.0
#         correct_val = 0
#         total_val = 0
#
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels.float())
#                 val_loss += loss.item()
#
#                 predicted = torch.argmax(outputs, dim=1)
#                 labels_idx = torch.argmax(labels, dim=1)
#                 correct_val += (predicted == labels_idx).sum().item()
#                 total_val += batch_size
#
#                 #for confussion matrix
#                 probabilities = torch.sigmoid(outputs)
#                 predictions = torch.argmax(probabilities, dim=1)
#                 all_labels.extend(labels.tolist())
#                 all_predictions.extend(predictions.tolist())
#
#         val_accuracy = 100 * correct_val / total_val
#         wandb.log({"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy, "epochs": epoch})
#         # Print statistics
#         print(
#             f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}%"
#         )
#
#     print("Finished Training and Validation")
#
#     #torch.save(model.state_dict(), "models/first_model_state_dict.pth")
#     torch.save(model, 'models/first_model.pth')
#
#     # wandb.save('models/first_model_state_dict.pth')
#
#     #log_confusion_matrix(all_labels, all_predictions)
#
# if __name__ == "__main__":
#     wandb.agent(sweep_id, function=main, count=10)

