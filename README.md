Project Goal: The overarching aim of this project is to develop a sentiment analysis system for classifying tweets into 'positive', 'neutral', 'negative', or 'non-applicable' categories. This tool is designed to delve into the realm of public opinion, capturing a wide spectrum of emotions and viewpoints expressed on Twitter. The importance of this project lies in its potential application across various sectors, including market research, political analysis, and social media monitoring, providing a nuanced understanding of public sentiment on diverse topics.

Frameworks and Integration: To achieve this, we will leverage the advanced capabilities of Hugging Face's transformer models, renowned for their superior performance in natural language processing tasks. These models, such as BERT and GPT, have demonstrated exceptional proficiency in understanding and interpreting the complexities of human language, which is crucial for accurate sentiment analysis.

Alongside Hugging Face, the project will utilise PyTorch and PyTorch Lightning for the neural network development. PyTorch offers an interactive and highly flexible platform for deep learning models, while PyTorch Lightning simplifies the coding process, allowing for more efficient model development and easier scalability. Here we will aim to optimise hyperparameters potentially using monitoring software such as wandb.ai.

Data Utilisation: Initially, the project will train the model using the "Twitter Entity Sentiment Analysis" dataset from Kaggle:
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis 
This dataset provides a collection of labelled tweets, offering a solid foundation for training a sentiment analysis model. To enhance the model's accuracy and adaptability, we plan to continuously incorporate new tweets/posts into the training process, ensuring the model remains up-to-date with the evolving language and expressions on Twitter/X.

Expected Models and Approach: The project will start by exploring various transformer-based models from Hugging Face, focusing on those particularly adept at contextual and emotional nuances in text. After selecting the most suitable transformer model for embedding generation, we will design a custom neural network using PyTorch. This network will specialise in detecting and interpreting sentiment-related patterns within the embeddings.

The final goal is to create a scalable, and adaptable model that can classify tweet sentiments effectively, even as it encounters new data and evolving language patterns. We aim to bridge the gap between advanced NLP technology and practical applications in sentiment analysis, creating a powerful tool for understanding the emotional undercurrents of social media discourse.

# project_name

Analysing Tweets sentiments using machine learning and with optimal MLOPS

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── project_name  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
