import torch
import pytorch_lightning as pl
from train_model_sweep_wandb import FCNN_model  # Import your FCNN_model
import os
from google.cloud import storage
import torch
import pytorch_lightning as pl
from train_model_sweep_wandb import FCNN_model  # Import your FCNN_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from google.cloud import storage
from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
import copy

torch.set_float32_matmul_precision('medium')
def quantize_model(model, quantize_datatype):
    # Specify the layers you want to quantize, here we're quantizing only the Linear layers
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=quantize_datatype
    )
    return quantized_model


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {source_blob_name} downloaded to {destination_file_name}."
    )

class InferenceModel(pl.LightningModule):
    def __init__(self, model_checkpoint_path):
        super().__init__()

        # Load the trained model checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")  # Load on CPU

        # Create an instance of your FCNN_model
        self.model = FCNN_model()

        # Rename the keys in the loaded state_dict to match your model's state_dict
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")  # Remove the "model." prefix
            new_state_dict[new_key] = value

        # Load the renamed state_dict
        self.model.load_state_dict(new_state_dict)

    def forward(self, x):
        # Forward pass through the model
        with torch.no_grad():
            pred = self.model(x)
        return pred

def calculate_accuracy(predictions, true_labels):
    # Assuming predictions and true_labels are both tensors
    _, predicted_indices = torch.max(predictions, 1)
    _, true_indices = torch.max(true_labels, 1)
    correct_predictions = torch.sum(predicted_indices == true_indices)
    accuracy = correct_predictions.item() / len(predictions)
    return accuracy * 100  # Convert to percentage


def main(model_checkpoint_path, tensor_path, label_path, quantize_datatype, pruning_amount):
    # If the model path is a GCS URL, download the file
    if model_checkpoint_path.startswith('gs://'):
        # Parse the GCS URL
        path_parts = model_checkpoint_path[5:].split('/')
        bucket_name = path_parts[0]
        blob_name = '/'.join(path_parts[1:])
        local_model_path = 'temp_model_checkpoint.ckpt'  # Temporary local file
        
        # Download the model checkpoint file from GCS
        download_blob(bucket_name, blob_name, local_model_path)
        
        # Use the local file path for further processing
        model_checkpoint_path = local_model_path

    #################  Regular model     ##################################
    inference_model = InferenceModel(model_checkpoint_path)

    # Load your .pt files
    input_tensor = torch.load(tensor_path)
    labels_tensor = torch.load(label_path)

    # Get predictions
    predictions = inference_model(input_tensor)

    # Calculate accuracy
    accuracy_percentage = calculate_accuracy(predictions, labels_tensor)
    _, true_indices = torch.max(labels_tensor, 1)

    
    print(f"Model Accuracy: {accuracy_percentage:.2f}%")

    _, predicted_indices = torch.max(predictions, 1)
    cm = confusion_matrix(true_indices.cpu(), predicted_indices.cpu())
    display_labels = ["irrelevant", "negative", "neutral", "positive"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    ################## Plot using the quantized model          ################################

    quantized_inference_model = quantize_model(inference_model.model, quantize_datatype)
    print("Model has been quantized.")

    # Run inference using the quantized model
    quantized_predictions = quantized_inference_model(input_tensor)

    # Calculate accuracy for the quantized model
    quantized_accuracy_percentage = calculate_accuracy(quantized_predictions, labels_tensor)
    print(f"Quantized Model Accuracy: {quantized_accuracy_percentage:.2f}%")
    
    # Calculate confusion matrix for the quantized model
    _, quantized_predicted_indices = torch.max(quantized_predictions, 1)
    cm_quantized = confusion_matrix(true_indices.cpu(), quantized_predicted_indices.cpu())
    disp_quantized = ConfusionMatrixDisplay(confusion_matrix=cm_quantized, display_labels=display_labels)

    # Plot confusion matrix for the quantized model
    disp_quantized.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Quantized Model')
    plt.show()

    ################## Plot using the pruning model    ###############################
    
    pruning_model = copy.deepcopy(inference_model)

    for name, module in pruning_model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Only prune nn.Linear layers
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)

    print("Model has been pruned.")

    # Run inference using the pruned model
    pruned_predictions = pruning_model(input_tensor)

    # Calculate accuracy for the pruned model
    pruned_accuracy_percentage = calculate_accuracy(pruned_predictions, labels_tensor)
    print(f"Pruned Model Accuracy: {pruned_accuracy_percentage:.2f}%")
    
    # Calculate confusion matrix for the pruned model
    _, pruned_predicted_indices = torch.max(pruned_predictions, 1)
    cm_pruned = confusion_matrix(true_indices.cpu(), pruned_predicted_indices.cpu())
    disp_pruned = ConfusionMatrixDisplay(confusion_matrix=cm_pruned, display_labels=display_labels)

    # Plot confusion matrix for the pruned model
    disp_pruned.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Pruned Model')
    plt.show()


    ################ Compilation            #######################




if __name__ == "__main__":

    path_model = "gs://bucket_processed_data/models/FCNN/epoch=82-val_loss=1.01.ckpt"
    path_embeds = "data/processed/text_embeddings_test.pt"
    path_labels = "data/processed/labels_test.pt"
    quantize_datatype = torch.qint8
    pruning_amount = 0.2
    """
    Labels are one hot encoded vectors. The onehot element corrosponds to the following categories {0: "irrelevant", 1: "negative", 2: "neutral", 3: "positive"}. 
    """
    main(path_model, path_embeds, path_labels, quantize_datatype, pruning_amount)
