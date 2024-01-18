from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import wandb
import numpy as np
import pandas as pd
import hydra
from omegaconf import OmegaConf
import logging
from transformers import set_seed
from typing import List, NoReturn
from hydra.utils import to_absolute_path
import pandas as pd
import torch
from typing import NoReturn
from sklearn.preprocessing import OneHotEncoder
import os
import json


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
wandb.init(project="day4_mlops", entity="twitter_sentiments_mlops")

def save_labels(df: pd.DataFrame, path: str) -> NoReturn:
    """
    Save the labels to a one-hot encoded torch tensor and category mapping to a text file.

    :param df: DataFrame containing the labels.
    :param path: Path to save the labels and category mapping.
    """
    # Ensure the 'sentiment label' column is of type 'category'
    df["sentiment label"] = df["sentiment label"].astype("category")

    # Create a one-hot encoder
    one_hot_encoder = OneHotEncoder(sparse=False)

    # Fit and transform the labels to one-hot encoding
    one_hot_labels = one_hot_encoder.fit_transform(df[["sentiment label"]])

    # Convert to a PyTorch tensor
    tensor = torch.tensor(one_hot_labels, dtype=torch.float32)

    # Save the tensor
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if it doesn't exist
    torch.save(tensor, os.path.join(path, "labels_test.pt"))

    # Create the category mapping
    category_mapping = {category: one_hot_encoder.transform([[category]]).tolist()[0]
                        for category in df["sentiment label"].cat.categories}

    # Save the category mapping to a text file
    mapping_path = os.path.join(path, "category_mapping_test.txt")
    with open(mapping_path, 'w') as file:
        file.write(json.dumps(category_mapping, indent=4))

    print("Category mapping saved to:", mapping_path)


def save_embeddings(df: pd.DataFrame, path: str, excluded_characters: List[str], none_replacement="Nothing", batch_size=10) -> NoReturn:
    """
    Save the embedded tweets to a torch tensor using a pretrained model.

    :param df: Dataframe containing the tweets.
    :param path: Path to save the embeddings.
    :param excluded_characters: List of characters to be removed from each tweet.
    """
    # Load pre-trained model and tokenizer
    model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("device: ", device)
    texts = list(df["tweet"])
    cleaned_texts = clean_strings(texts, excluded_characters, none_replacement)

    # Assuming 'cleaned_texts' is your list of texts
    num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size  # Compute the number of batches required
    output_tensors = []
    # Process and log in batches
    for i in range(num_batches):
        # Calculate start and end indices of the current batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # Get the current batch
        batch_texts = cleaned_texts[start_idx:end_idx]

        # Tokenize and create inputs
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract embeddings (e.g., pooled output)
        embeddings = outputs.pooler_output
        #flattened_embeddings = embeddings.transpose(0, 1).flatten(start_dim=0, end_dim=1)
        output_tensors.append(embeddings.to("cpu"))
        wandb.log({"Number of tweets embedded": end_idx+1})
    
    extended_tensor = torch.tensor([])
    for tensor in output_tensors:
        extended_tensor = torch.cat((extended_tensor, tensor))
    
    torch.save(extended_tensor, path + "/text_embeddings_test.pt")

def clean_strings(string_list: List[str], remove_chars: List[str], none_replacement: str) -> List[str]:
    """
    Cleans a list of strings by removing specified characters and replacing NaN values with a specific set of characters.

    :param string_list: List of strings to be cleaned.
    :param remove_chars: List of characters to be removed from each string.
    :param none_replacement: String to replace None or NaN values.
    :return: List of cleaned strings.
    """
    cleaned_list = []
    
    for item in string_list:
        if item is None or pd.isna(item):  # Check for NaN or None
            cleaned_list.append(none_replacement)
        else:
            for char in remove_chars:
                item = item.replace(char, '')  # Remove specified characters
            cleaned_list.append(item)

    return cleaned_list


@hydra.main(config_path="configurations", config_name="make_dataset_test_config.yaml")
def main(cfg):
    hparams = cfg.experiment

    none_replacement = hparams["replace_nan_with"]
    k = hparams["n_tweets_embed"]
    seed = hparams["seed"]
    batch_size = hparams["batch_size"]
    excluded_characters =  hparams["excluded_char"]
    print("k ", k, " seed ", seed)
    processed_directory_path = to_absolute_path("data/processed")
    raw_data_path = to_absolute_path("data/raw/twitter_validation.csv")
    df = pd.read_csv(raw_data_path)
    df = df.head(k)  # k is an integer
    df.columns = ["id", "game", "sentiment label", "tweet"]

    set_seed(seed)
    save_embeddings(df, processed_directory_path, excluded_characters, none_replacement, batch_size)
    save_labels(df, processed_directory_path)


if __name__ == "__main__":
    main()
    wandb.finish()
