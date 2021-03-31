# import gensim.downloader as api
import joblib
import logging
import nltk
import os
import pandas as pd
import random
import string
import torch

from nltk.corpus import stopwords
from pathlib import Path
from torchtext.legacy import data


def project_root() -> str:
    """Returns project root folder."""
    return str(Path(__file__).parent.parent)


def make_sets(
    train_path: str,
    text_path: str,
    nodeid2paper_path: str,
    test_path: str,
    test_size: float,
) -> pd.DataFrame:
    """Makes a training, local test set and test set"""
    assert test_size < 1, "Test size must be smaller than 1"
    # reading files
    train_df = pd.read_csv(
        project_root() + train_path, header=None, names=["label", "node_id"]
    )
    train_df = train_df[["node_id", "label"]]
    nodeid2paperid = pd.read_csv(project_root() + nodeid2paper_path)
    nodeid2paperid.rename(
        columns={"node idx": "node_id", "paper id": "paper_id"}, inplace=True
    )
    text_df = pd.read_csv(
        project_root() + text_path, header=None, names=["paper_id", "title", "abstract"]
    )
    test_df = pd.read_csv(project_root() + test_path, header=None, names=["node_id"])
    # merge paper id
    train_df = train_df.merge(nodeid2paperid, on="node_id", how="left")
    test_df = test_df.merge(nodeid2paperid, on="node_id", how="left")
    # splitting training and testing for local use
    training_df = train_df.iloc[: int(len(train_df) - (len(train_df) * test_size)), :]
    testing_df = train_df.iloc[int(len(train_df) - (len(train_df) * test_size)):, :]
    # training and test set
    training_set = text_df.merge(training_df, how="inner", on="paper_id")
    training_set = training_set[["node_id", "title", "abstract", "label"]]
    local_test_set = text_df.merge(testing_df, how="inner", on="paper_id")
    local_test_set = local_test_set[["node_id", "title", "abstract", "label"]]
    test_set = text_df.merge(test_df, how="inner", on="paper_id")
    test_set = test_set[["node_id", "title", "abstract"]]
    return training_set, local_test_set, test_set


def save_model(clf, name: str):
    output_path = project_root() + f"/models/{name}.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(clf, output_path)


def load_model(name: str):
    try:
        loaded_model = joblib.load(project_root() + f"/models/{name}.pkl")
    except FileNotFoundError:
        raise Exception("Invalid model name or non-trained model")
    return loaded_model


def make_submission_csv(predictions):
    pass


def get_logger():
    global logger
    line_format = "%(asctime)s %(levelname)s: %(filename)s:%(funcName)s():%(lineno)d - %(message)s"
    logging.basicConfig(level=logging.INFO, format=line_format)
    logger = logging.getLogger(__name__)
    logger.info("Logger initialized")
    return logger


def preprocess_text(data: pd.DataFrame, name: str, test=False):
    # concat title and abstract
    data["title_abstract"] = data["title"] + " " + data["abstract"]
    if test:
        data = data[["node_id", "title_abstract"]]
    else:
        data = data[["node_id", "title_abstract", "label"]]
    # remove punctuation
    data["title_abstract"] = data["title_abstract"].str.translate(
        str.maketrans("", "", string.punctuation)
    )
    # save to csv
    data.to_csv(project_root() + f"/data/{name}.csv", index=False)


def process_text(device):
    training_set, local_test_set, test_set = make_sets(
        "/data/train.csv",
        "/data/text.csv",
        "/data/nodeid2paperid.csv",
        "/data/test.csv",
        0.1,
    )
    training_set = pd.concat([training_set, local_test_set], ignore_index=True)
    preprocess_text(training_set, "transformed_training")
    preprocess_text(test_set, "transformed_test", test=True)
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    # tokenize, lower and remove stop words
    TEXT = data.Field(
        tokenize="spacy",
        sequential=True,
        batch_first=True,
        lower=True,
        stop_words=stop_words,
    )
    LABEL = data.LabelField(dtype=torch.float, use_vocab=False, preprocessing=float)
    train_fields = [(None, None), ("text", TEXT), ("label", LABEL)]
    test_fields = [(None, None), ("text", TEXT)]
    train_dataset = data.TabularDataset(
        path=project_root() + "/data/transformed_training.csv",
        format="csv",
        fields=train_fields,
        skip_header=True,
    )
    test_dataset = data.TabularDataset(
        path=project_root() + "/data/transformed_test.csv",
        format="csv",
        fields=test_fields,
        skip_header=True,
    )
    # split data
    train_data, valid_data = train_dataset.split(
        split_ratio=0.9, random_state=random.seed(42)
    )
    TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)
    # build iterator
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_dataset),
        batch_size=64,
        sort=False,
        shuffle=False,
        device=device,
    )
    vocab_size = len(TEXT.vocab)
    pretrained_embeddings = pretrained_embeddings = TEXT.vocab.vectors
    return (
        train_iterator,
        valid_iterator,
        test_iterator,
        vocab_size,
        pretrained_embeddings,
    )


def make_predictions_csv(test_set_path: str, predictions: torch.tensor):
    test_set = pd.read_csv(project_root() + test_set_path)
    # batch to single list of array
    predictions = [item for sublist in predictions for item in sublist.numpy()]
    test_set["label"] = predictions
    test_set = test_set[["node_id", "label"]]
    # save to csv
    test_set.to_csv(project_root() + "/data/predictions.csv", index=False)
