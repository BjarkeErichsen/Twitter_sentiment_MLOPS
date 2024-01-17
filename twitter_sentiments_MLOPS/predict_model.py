import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import wandb
import torch
import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader, TensorDataset
#wandb.init(project="twitter_sentiment_MLOPS")



class InferenceModel(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()
        # Load the trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('models/first_model.pth', map_location=torch.device(device))
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.embedder = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.label_mapping = {0: "positive", 1: "negative", 2: "neutral", 3: "irrelevant"}

    def forward(self, x):
        # Forward pass through the model
        x = self.tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            x = self.embedder(**x).pooler_output
        pred = self.model(x)
        idx = torch.argmax(pred).item()
        return self.label_mapping[idx]

def tweet_parse_args():
    parser = argparse.ArgumentParser(description="Tweet sentiment inference")
    parser.add_argument("--tweet", type=str, required=True, help="Tweet text for sentiment analysis")
    return parser.parse_args()

def main():
    # Parse arguments from command line
    args = tweet_parse_args()
    tweet = args.tweet

    # Assuming you have a model path
    model_path = 'models/first_model.pth'

    # Load the model
    model = InferenceModel(model_path=model_path)
    model.eval()  # Set the model to evaluation mode

    # Make a prediction
    with torch.no_grad():
        prediction = model(tweet)
        print("Prediction:", prediction)

if __name__ == "__main__":
    main()



"""

# Evaluate the model
accuracy = evaluate_model(test_loader, tokenizer, embedding_model, model)
# print(f"Accuracy on the test set: {accuracy}%")


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    '''Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    '''
    return torch.cat([model(batch) for batch in dataloader], 0)

############ data load ###############
# Assuming test_embeddings_tensor and test_labels_tensor are your test dataset tensors
#test_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
#test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

embedding_dim = 768
hidden_dim = 128

# Load the model
#model = SimpleNN(embedding_dim, hidden_dim)
# model.load_state_dict(torch.load('path_to_save_model/first_model_state_dict.pth'))
model = torch.load('models/first_model.pth')
model.eval()  # Set the model to evaluation mode

# Load tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
embedding_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


def preprocess_and_predict(tweet, tokenizer, embedding_model, classification_model):
    # Tokenize and prepare input
    inputs = tokenizer(tweet, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = embedding_model(**inputs)

    # Extract embeddings (e.g., pooled output)
    embeddings = outputs.pooler_output

    # Prediction
    with torch.no_grad():
        logits = classification_model(embeddings)
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        print(probabilities)
        label = torch.argmax(probabilities, dim=1).item()  # Get the index of max probability
    return label

# Example tweet
tweet = "i am so angry and mad, and very unhappy"
tweet = "I am coming to the borders and I will kill you all"
label = preprocess_and_predict(tweet, tokenizer, embedding_model, model)

# Map the label to a meaningful category
label_mapping = {0: "positive", 1: "negative", 2: "neutral", 3: "irrelevant"}
predicted_category = label_mapping[label]

print(f"The predicted category for the tweet is: {predicted_category}")


##### Real evaluation #####

def evaluate_model(test_loader, tokenizer, embedding_model, classification_model):
    classification_model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Generate embeddings
            inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            embeddings = embedding_model(**inputs).pooler_output

            # Predict
            outputs = classification_model(embeddings)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # Log the accuracy to wandb
    #wandb.log({"test_accuracy": accuracy})
    return accuracy
"""