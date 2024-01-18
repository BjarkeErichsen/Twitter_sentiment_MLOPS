# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY twitter_sentiments_MLOPS/ twitter_sentiments_MLOPS/

WORKDIR /
ENTRYPOINT ["python", "-u", "twitter_sentiments_MLOPS/train_model_sweep_wandb.py"]