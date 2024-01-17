from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from twitter_sentiments_MLOPS.predict_model import InferenceModel


app = FastAPI()


class TweetRequest(BaseModel):
    tweet: str

# Load your model here (update with your actual model path)
model_path = 'models/first_model.pth'
model = InferenceModel(model_path=model_path)
#model.eval()  # Set the model to evaluation mode

@app.post("/predict/")
def predict_tweet(request: TweetRequest):
    tweet = request.tweet.strip()

    if not tweet:
        raise HTTPException(status_code=400, detail="The tweet text cannot be empty.")

    with torch.no_grad():
        prediction = model.forward(tweet)

    return {"tweet": tweet, "prediction": prediction}
