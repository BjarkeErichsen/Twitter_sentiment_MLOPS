#going to usethe following commands to build
#docker build -f dockerfiles/KNN_train.dockerfile . -t baselinedokcer:latest


# This is IMPLICITLY executed in the base directory. 
FROM python:3.10-slim

#first line udates the package list
#second line installs essentials + gcc (library of building c/c++ programs)
#third line   DOES CLEANUP of unused files!
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 

#These lines copy files (relative to root directory) into the docker image:
#All COPY is  "COPY <FROM> <TO>" 
#<TO> is relative to the WORKDIR directory here the root directory
#<FROM> is relative to the "BUILD CONTEXT"
#Setting " . " in docker build -> BUILD CONTEXT = where i execute the CWD
#We do here: 1. requirements are copied, 2. pyproject.toml is copied  3. all files within twitter_sentiment_MLOPS  4. Alt inside data
#Docker AUTO-reproduces folders above AND below. 
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY reports/ reports/
COPY twitter_sentiments_MLOPS/baselines/ twitter_sentiments_MLOPS/baselines/
COPY data/ data/
COPY models/ models/


#Workdir /  sets the workdir to the root directory of the files system
#Therefore ALL "RUN", "COPY", "ENTRY POINT" etc commands will be executed from the root directory.
#This ONLY affects the commands that come AFTER "WORKDIR"!
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

#ENTRYPOINT specified what command will be EXECUTED WHEN RUNNING CONTAINER
#"-u" specified that python interpretor should be unbuffered.
#Any additional arguments to "DOCKER RUN" will be given as arguments to this script!
ENTRYPOINT ["python", "-u", "twitter_sentiments_MLOPS/baselines/KNN_train.py"]