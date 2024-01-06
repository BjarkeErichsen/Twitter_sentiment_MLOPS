import torch
from transformers import AutoTokenizer, AutoModel
from twitter_sentiments_MLOPS.models.model import SimpleNN


embedding_dim = 768 
hidden_dim = 128
output_dim = 4  

# Load the model
model = SimpleNN(embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('path_to_save_model/first_model_state_dict.pth'))
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
        prediction = classification_model(embeddings)
    
    # Convert prediction to label
    label = torch.argmax(prediction, dim=1).item()
    return label

# Example tweet
tweet = "This is an example tweet to test the model"
label = preprocess_and_predict(tweet, tokenizer, embedding_model, model)

# Map the label to a meaningful category
label_mapping = {0: "positive", 1: "negative", 2: "neutral", 3: "irrelevant"}
predicted_category = label_mapping[label]

print(f"The predicted category for the tweet is: {predicted_category}")



''' 
def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)
    '''