# Kaggle competition


This project was part of [MATH80600A's class](https://chao1224.github.io/math80600_winter2021/index.html).

The goal of the project was to classify scientific papers based on their title and abstract.

Project Organization
------


    ├── data               <- Accessible data for the competition.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    └── src                <- Source code for use in this project.
        ├── baseline.py    <- SVM classifier used as a baseline score.
        ├── embeddings.py  <- GRU with pre-trained embeddings model.
        ├── utils.py       <- Utilities used for the project.

## Installation

```bash
virtualenv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
```

## Model training
```bash
python -m src.embeddings --embedding_dim [embedding dimension] --hidden_dim [hidden dimension] --num_layers [nb of layers] --epochs [nb of epochs]
```
# Hyper-parameters used
```bash
Embedding dimension: 300
Hidden dimension: 126
Number of layer: 1
Epochs: 15
```
