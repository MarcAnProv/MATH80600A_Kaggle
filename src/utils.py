import gensim.downloader as api
import joblib
import logging
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from pathlib import Path
from tqdm import tqdm


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


def download_pretrained_embeddings():
    path_word2vec = api.load("word2vec-google-news-300", return_path=True)
    model = KeyedVectors.load_word2vec_format(path_word2vec, binary=True)
    return model


def create_vocabulary(training_set: pd.DataFrame) -> list:
    """Extracts vocabulary from our corpus"""
    logger.info("Creating vocabulary")
    stop_words = set(stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    title = training_set.title.tolist()
    abstract = training_set.abstract.tolist()
    voc = []
    # concatenate both lists
    concat = [*title, *abstract]
    for sentence in tqdm(concat):
        # split sentence using tokenizer
        sentence = tokenizer.tokenize(sentence.lower())
        # removing stop words
        for word in sentence:
            if word not in stop_words:
                voc.append(word)
    # get unique words
    voc = np.unique(voc)
    return voc


def embedding_matrix(vocabulary: list) -> np.array:
    """Creates an embedding matrix for the embedding layer"""
    logger.info("Creating embedding matrix")
    embed_dim = 300
    embed = np.zeros((len(vocabulary), embed_dim))
    words_found = 0
    model = download_pretrained_embeddings()
    for index, word in enumerate(vocabulary):
        try:
            embed[index] = model[word]
            words_found += 1
        except KeyError:
            embed[index] = np.random.normal(size=(embed_dim))
    return embed


def create_embedding_layer(embedding_matrix: np.array):
    num_embeddings, embedding_dim = embedding_matrix.size()
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
    return embedding_layer, embedding_dim
