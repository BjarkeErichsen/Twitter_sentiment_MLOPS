# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
RUN pip install -r requirements.txt --no-cache-dir

COPY twitter_sentiments_MLOPS twitter_sentiments_MLOPS

WORKDIR /

RUN pip install . --no-deps --no-cache-dir

EXPOSE 8080

CMD exec uvicorn twitter_sentiments_MLOPS.api:app --port 8080 --host 0.0.0.0 --workers 1
#CMD ["uvicorn", "twitter_sentiments_MLOPS.api:app", "--host", "0.0.0.0", "--port", "80"]

#docker build -f 'dockerfiles/predict_model.dockerfile' -t twitter-sentiment-app .

#docker tag twitter-sentiment-app:latest gcr.io/mlops-tsa/predict-container
#docker push gcr.io/mlops-tsa/predict-container:latest

#uvicorn --reload --port 8080:8080 twitter_sentiments_MLOPS.api:app



#docker run -it --rm -p 8080:8080 twitter-sentiment-app

#http://localhost:80 or http://127.0.0.1:80 or http://127.0.0.1:8080/docs