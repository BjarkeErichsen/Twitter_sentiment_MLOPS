import torch
import numpy as np

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel
import matplotlib.pyplot as plt


############ TOKENIZER ############
#MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
#tokenizer = AutoTokenizer.from_pretrained(MODEL)
#config = AutoConfig.from_pretrained(MODEL)
#model = AutoModel.from_pretrained(MODEL)



import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  # Use softmax for multi-class classification

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.softmax(output)  # Apply softmax
        return output




############# PLOT Embeddings ###############
# Encode text
def plot_embeddings(textstring):
    text = textstring
    inputs = tokenizer(text, return_tensors="pt")
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    from sklearn.decomposition import PCA

    # Use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    embeddings_reduced = pca.fit_transform(embeddings[0].numpy())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1])

    # Optionally, annotate points
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])):
        plt.annotate(token, (embeddings_reduced[i, 0], embeddings_reduced[i, 1]))

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("2D Visualization of Embeddings")
    plt.show()