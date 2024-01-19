# Import necessary libraries and modules
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from twitter_sentiments_MLOPS.predict_model import InferenceModel
import os

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for the expected request format
class TweetRequest(BaseModel):
    tweet: str

# Set up the path to the model checkpoint file
model_path = os.path.join(os.getcwd(), 'twitter_sentiments_MLOPS', 'models', 'FCNN', 'feasible-sweep-1', 'epoch=82-val_loss=1.01.ckpt')

# Load the InferenceModel with the specified checkpoint
model = InferenceModel(model_path)

# Define a POST endpoint for predicting the sentiment of a tweet
@app.post("/predict/")
def predict_tweet(request: TweetRequest):
    # Extract and strip the tweet text from the request
    tweet = request.tweet.strip()

    # Validate the input tweet text
    if not tweet:
        raise HTTPException(status_code=400, detail="The tweet text cannot be empty.")

    # Perform the prediction
    with torch.no_grad():
        prediction = model.forward(tweet)

    # Return the tweet text and its predicted sentiment
    return {"tweet": tweet, "prediction": prediction}
