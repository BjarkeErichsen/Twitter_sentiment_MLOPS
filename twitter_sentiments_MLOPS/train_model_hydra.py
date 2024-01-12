import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from twitter_sentiments_MLOPS.models.model import SimpleNN
#from twitter_sentiments_MLOPS.visualizations.visualize import log_confusion_matrix
import hydra.utils as hydra_utils

import wandb

# Config decorator for Hydra
@hydra.main(config_path="configurations", config_name="train_model_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    original_cwd = hydra_utils.get_original_cwd()

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize Weights & Biases
    wandb.init(project="hydra_training_train", entity="twitter_sentiments_mlops", config=config_dict)

    #wandb.init(project="twitter_sentiments_mlops", entity="twitter_sentiments_mlops", config=cfg)

    # Load data data/processed/labels.pt
    labels_path = os.path.join(original_cwd, "data/processed/labels.pt")
    embeddings_path = os.path.join(original_cwd, "data/processed/text_embeddings.pt")
     # Now load the data using these paths
    labels_tensor = torch.load(labels_path)
    embeddings_tensor = torch.load(embeddings_path)

    # Split dataset
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        embeddings_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    # Datasets and Dataloaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Model, criterion, and optimizer
    model = SimpleNN(cfg.model.embedding_dim, cfg.model.hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training and Validation Loop
    for epoch in range(cfg.training.num_epochs):
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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            #loss = criterion(outputs, labels) # for CrossEntropyLoss as loss function
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


            predicted = torch.sigmoid(outputs).data > 0.5  # Threshold at 0.5
            correct_train += (predicted == labels).sum().item()
            total_train += labels.numel()

            #for CrossEntropyLoss as loss function
            #_, predicted = torch.max(outputs.data, 1)
            #total_train += labels.size(0)
            #correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        wandb.log({"train_loss": train_loss / len(train_loader), "train_accuracy": train_accuracy}, step=epoch)

        # Validation
        all_labels = []
        all_predictions = []
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:

                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                #loss = criterion(outputs, labels) # for CrossEntropyLoss as loss function
                val_loss += loss.item()



                predicted = torch.sigmoid(outputs).data > 0.5  # Apply sigmoid and threshold
                correct_val += (predicted == labels).sum().item()
                total_val += labels.numel()

                #for confussion matrix
                probabilities = torch.sigmoid(outputs)
                predictions = torch.argmax(probabilities, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())

                #for CrossEntropyLoss as loss function
                #_, predicted = torch.max(outputs.data, 1)
                #total_val += labels.size(0)
                #correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        wandb.log({"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy}, step=epoch)
        # Print statistics
        print(
            f"Epoch {epoch+1}/{cfg.training.num_epochs}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}%"
        )

    print("Finished Training and Validation")

    #torch.save(model.state_dict(), "models/first_model_state_dict.pth")





    model_path = os.path.join(original_cwd, 'models/first_model.pth')
    torch.save(model, model_path)
    #torch.save(model, 'models/first_model.pth') # saves the full model

    # Optional: Save the model's final state to wandb
    # wandb.save('models/first_model_state_dict.pth')

    #log_confusion_matrix(all_labels, all_predictions)

# Run the main function with Hydra
if __name__ == "__main__":
    main()
    #python twitter_sentiments_MLOPS/train_model_hydra.py training.learning_rate=0.002 training.batch_size=8
