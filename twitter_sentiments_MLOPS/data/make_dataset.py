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

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
wandb.init(project="day1_mlops", entity="twitter_sentiments_mlops")


def save_labels(df: pd.DataFrame, path: str) -> NoReturn:
    """
    Save the labels to a one-hot encoded torch tensor.

    :param df: Dataframe containing the labels.
    :param path: Path to save the labels.
    """
    df["sentiment label"] = df["sentiment label"].astype("category")
    category_encoded = df["sentiment label"].cat.codes
    tensor = torch.tensor(category_encoded.values)
    torch.save(tensor, path + "/labels.pt")


def save_embeddings(df: pd.DataFrame, path: str, excluded_characters: List[str], none_replacement="Nothing") -> NoReturn:
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

    texts = list(df["tweet"])
    cleaned_texts = clean_strings(texts, excluded_characters, none_replacement)

    # Assuming 'cleaned_texts' is your list of texts
    batch_size = 10
    num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size  # Compute the number of batches required

    # Process and log in batches
    for i in range(num_batches):
        # Calculate start and end indices of the current batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # Get the current batch
        batch_texts = cleaned_texts[start_idx:end_idx]

        # Tokenize and create inputs
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
    
        wandb.log({"Number of tweets embedded": end_idx+1})

    # Extract embeddings (e.g., pooled output)
    embeddings = outputs.pooler_output
    torch.save(embeddings, path + "/text_embeddings.pt")

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


@hydra.main(config_path="configurations", config_name="make_dataset_config.yaml")
def main(cfg):
    hparams = cfg.experiment

    none_replacement = hparams["replace_nan_with"]
    k = hparams["n_tweets_embed"]
    seed = hparams["seed"]
    excluded_characters =  hparams["excluded_char"]
    print("k ", k, " seed ", seed)
    processed_directory_path = to_absolute_path("data/processed")
    raw_data_path = to_absolute_path("data/raw/twitter_training.csv")
    df = pd.read_csv(raw_data_path)
    df = df.head(k)  # k is an integer
    df.columns = ["id", "game", "sentiment label", "tweet"]

    set_seed(seed)
    save_labels(df, processed_directory_path)
    save_embeddings(df, processed_directory_path, excluded_characters, none_replacement)


if __name__ == "__main__":
    main()
    wandb.finish()
    
