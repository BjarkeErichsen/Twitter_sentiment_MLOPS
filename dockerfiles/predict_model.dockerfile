# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY twitter_sentiments_MLOPS twitter_sentiments_MLOPS

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

CMD ["uvicorn", "twitter_sentiments_MLOPS.api:app", "--host", "0.0.0.0", "--port", "80"]

#uvicorn --reload --port 8000 twitter_sentiments_MLOPS.api:app




#docker build -f 'dockerfiles/predict_model.dockerfile' -t twitter-sentiment-app .

#docker run -it --rm -p 80:80 twitter-sentiment-app

#http://localhost:80 or http://127.0.0.1:80 or http://127.0.0.1:80/docs