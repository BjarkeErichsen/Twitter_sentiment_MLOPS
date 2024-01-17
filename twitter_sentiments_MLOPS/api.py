from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from twitter_sentiments_MLOPS.predict_model import InferenceModel
import os
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl

app = FastAPI()
#uvicorn --reload --port 8000 twitter_sentiments_MLOPS.api:app
class TweetRequest(BaseModel):
    tweet: str

# # Load your model here (update with your actual model path)
# print("Current working directory:", os.getcwd())

# full_model_path = os.path.join(os.getcwd(), model_path)
# print("Full model path:", full_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.getcwd(), 'twitter_sentiments_MLOPS', 'models', 'FCNN', 'best-checkpoint.ckpt')

model = InferenceModel(model_path)

@app.post("/predict/")
def predict_tweet(request: TweetRequest):
    tweet = request.tweet.strip()

    if not tweet:
        raise HTTPException(status_code=400, detail="The tweet text cannot be empty.")

    with torch.no_grad():
        prediction = model.forward(tweet)

    return {"tweet": tweet, "prediction": prediction}
