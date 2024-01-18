import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import evidently
from twitter_sentiments_MLOPS.train_model_sweep_wandb import LightningModel, FCNN_model
from torch.utils.data import TensorDataset

# Step 2: Generate embedding for the new tweet
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_hug = AutoModel.from_pretrained(model_name)
#model = torch.load('models/first_model.pth')
#model.eval()

# Instantiate the model
model_light = LightningModel(0.001)

# Load the checkpoint
checkpoint = torch.load('twitter_sentiments_MLOPS/models/FCNN/best-checkpoint-v17.ckpt')

# Inspect the checkpoint keys
print(checkpoint.keys())

# Correctly load the state dict into the model
# This is a common key, but you should replace it with the correct one from your checkpoint
model_light.load_state_dict(checkpoint['state_dict'])



new_tweets = [
    "I’ve never actually done a psycho cosplay. Is it too late? #borderlands",
    "so bored on my flight home i drew timothy lawrence #borderlands",
    "Who is this Borderlands 3 Badass? #Borderlands #guessinggame #notapokemon",
    "If you were a villain, what kind of villain would you be? I have one answer: the savior of Pandora, kiddo.#Borderlands #Borderlands2",
    "Ready to conquer Pandora? Day 4 of Borderlands Week is about to kick off! Grab your gear and let's dive into the chaos together!#twitchaffiliate #Borderlands #Live #PukRuk"
]
"""
# Process each tweet
new_tweets_embeddings = []
for tweet in new_tweets:
    inputs = tokenizer(tweet, return_tensors="pt")
    with torch.no_grad():
        outputs = model_hug(**inputs)
        tweet_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        new_tweets_embeddings.append(tweet_embedding[0])

column_names = [f'col_{i}' for i in range(768)]

# Assign column names to dataframes
df_embeddings = pd.DataFrame(embeddings_matrix.numpy(), columns=column_names)
df_new_tweet = pd.DataFrame(new_tweets_embeddings, columns=column_names)


print(df_embeddings.shape)  

df_embeddings = df_embeddings.iloc[:5]
print(df_embeddings.head)
print(df_new_tweet.head)

# Step 4: Run Data Drift Analysis
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
print("yo")

report = Report(metrics=[DataDriftPreset()])
print("yoyo")

report.run(reference_data=df_new_tweet, current_data=df_embeddings)
print("yoyoyo")
report.save_html('report2.html')
print("done")
#"""
new_embeds = []

#############################################################



#layer_name = 'fc3'  # Replace with the actual layer name
#model_light._modules.get(layer_name).register_forward_hook(get_features(layer_name))
#model_light._modules.get('model').fc3._parameters.get('weight')

column_names = [f'col_{i}' for i in range(128)]

# Assign column names to dataframes

new_tweet_fc3_list = []
for tweet in new_tweets:

    tweet_tokenized = tokenizer(tweet, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        tweet_embedding = model_hug(**tweet_tokenized)

    # Extract embeddings (e.g., pooled output)
    embeddings = tweet_embedding.pooler_output
    new_embeds.append(embeddings[0])
    #damn, i really wish this tensor was a numpy array
    fc3_output = model_light.model.get_embed(embeddings[0]).detach().numpy()
    new_tweet_fc3_list.append(fc3_output)
new_tweet_fc3_df = pd.DataFrame(data = new_tweet_fc3_list, columns=column_names)

existing_tweet_fc3_list = []
embeddings_tensor = torch.load("data/processed/text_embeddings.pt")
for i in range(5):
    embedding = embeddings_tensor[i]
    embedding = embedding.unsqueeze(0)
    existing_tweet_fc3_list.append(model_light.model.get_embed(embedding).squeeze().detach().numpy())
existing_tweet_fc3_df = pd.DataFrame(data = existing_tweet_fc3_list, columns=column_names)

# Step 4: Run Data Drift Analysis
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("yo")

report = Report(metrics=[DataDriftPreset()])
print("yoyo")

report.run(reference_data=new_tweet_fc3_df, current_data=existing_tweet_fc3_df)
print("yoyoyo")
report.save_html('report2.html')
print("done")