import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import wandb

def log_confusion_matrix(all_labels, all_predictions, class_names=["positive", "negative", "neutral", "irrelevant"]):
    """
    Logs a normalized confusion matrix as an image in Weights & Biases.

    Args:
    all_labels (list): List of true labels, one-hot encoded.
    all_predictions (list): List of predicted labels.
    class_names (list): List of class names corresponding to labels.
    """
    # Convert one-hot encoded labels to class indices
    all_labels = [label.index(1) for label in all_labels]

    # Ensure all_predictions is a list of integers
    all_predictions = [int(prediction) for prediction in all_predictions]

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert to percentage
    cm_percentage = cm_normalized * 100

    # Plotting using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save the plot as an image file
    file_path = "reports/figures/confusion_matrix_normalized.png"
    plt.savefig(file_path)
    plt.close()

    # Log to wandb
    wandb.log({"conf_matrix": wandb.Image(file_path)})

# Example usage
# log_confusion_matrix(all_labels, all_predictions)
