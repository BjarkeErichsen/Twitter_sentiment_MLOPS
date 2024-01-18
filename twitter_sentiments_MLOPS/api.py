from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from twitter_sentiments_MLOPS.predict_model import InferenceModel
import os
from google.cloud import storage
import io

app = FastAPI()
cloud_run = True

class TweetRequest(BaseModel):
    tweet: str


model_path = os.path.join(os.getcwd(), 'twitter_sentiments_MLOPS', 'models', 'FCNN', 'feasible-sweep-1' ,'epoch=82-val_loss=1.01.ckpt')
model = InferenceModel(model_path)

@app.post("/predict/")
def predict_tweet(request: TweetRequest):
    tweet = request.tweet.strip()

    if not tweet:
        raise HTTPException(status_code=400, detail="The tweet text cannot be empty.")

    with torch.no_grad():
        prediction = model.forward(tweet)

    return {"tweet": tweet, "prediction": prediction}
