import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from twitter_sentiments_MLOPS.train_model_sweep_wandb import FCNN_model  # Import your FCNN_model
import os

torch.set_float32_matmul_precision('medium')

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

        # Load the tokenizer and embedder
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.embedder = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # Label mapping
        self.label_mapping = {0: "irrelevant", 1: "negative", 2: "neutral", 3: "positive"}

    def forward(self, x):
        # Forward pass through the model
        x = self.tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            x = self.embedder(**x).pooler_output
        pred = self.model(x)
        idx = torch.argmax(pred).item()
        return self.label_mapping[idx]
def main():
    # Replace 'path_to_your_checkpoint.ckpt' with the actual path to your saved model checkpoint file
    model_checkpoint_path = os.path.join(os.getcwd(), 'twitter_sentiments_MLOPS', 'models', 'FCNN', 'best-checkpoint.ckpt')
    inference_model = InferenceModel(model_checkpoint_path)

    # Example usage:
    text_to_classify = "This is a positive tweet."
    predicted_label = inference_model(text_to_classify)
    print(f"Predicted Sentiment: {predicted_label}")

if __name__ == "__main__":
    main()
