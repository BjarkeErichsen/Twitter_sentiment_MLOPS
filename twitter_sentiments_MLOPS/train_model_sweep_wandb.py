import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.model import FCNN_model, CNN_model
#from twitter_sentiments_MLOPS.visualizations.visualize import log_confusion_matrix
import hydra.utils as hydra_utils

import wandb

wandb.login()
# Example sweep configuration
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [200]},
        "epochs": {"values": [100]},
        "lr": {"values":[0.01]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweeptest", entity="twitter_sentiments_mlops")
# Config decorator for Hydra
def main():

    run = wandb.init()
    # Initialize Weights & Biases
    learning_rate = wandb.config.lr
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    embedding_dim = 768 
    hidden_dim = [128,64,4]
    # model = SimpleNN(embedding_dim, hidden_dim)
    model = FCNN_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    # Move model to the chosen device
    model.to(device)

    # Load data da
     # Now load the data using these paths
    labels_tensor = torch.load("data/processed/labels.pt")
    embeddings_tensor = torch.load("data/processed/text_embeddings.pt")

    # Split dataset
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        embeddings_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Training and Validation Loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        #             record_shapes=True, 
        ##             profile_memory=True, 
        #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/train')) as prof:
            
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss = criterion(outputs, labels.float())
            loss = criterion(outputs, labels) # for CrossEntropyLoss as loss function
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1) 
            labels_idx = torch.argmax(labels, dim=1)
            
            correct_train += (predicted == labels_idx).sum().item()
            total_train += batch_size

        train_accuracy = 100 * correct_train / total_train
        wandb.log({"train_loss": train_loss / len(train_loader), "train_accuracy": train_accuracy, "epochs": epoch})

        # Validation
        all_labels = []
        all_predictions = []
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1) 
                labels_idx = torch.argmax(labels, dim=1)
                correct_val += (predicted == labels_idx).sum().item()
                total_val += batch_size

                #for confussion matrix
                probabilities = torch.sigmoid(outputs)
                predictions = torch.argmax(probabilities, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())

        val_accuracy = 100 * correct_val / total_val
        wandb.log({"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy, "epochs": epoch})
        # Print statistics
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}%"
        )

    print("Finished Training and Validation")

    #torch.save(model.state_dict(), "models/first_model_state_dict.pth")
    torch.save(model, 'models/first_model.pth') 

    # wandb.save('models/first_model_state_dict.pth')

    #log_confusion_matrix(all_labels, all_predictions)

if __name__ == "__main__":
    wandb.agent(sweep_id, function=main, count=10)

