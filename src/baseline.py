import numpy as np
import pandas as pd
import time

from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from src.utils import get_logger, load_model, make_sets, save_model


logger = get_logger()


def text_preprocessing(
    training_data: pd.DataFrame, test_data: pd.DataFrame, vectorizer: str
) -> pd.DataFrame:
    # replacing digits by NUM token
    training_data.replace("\d+", "NUM", regex=True, inplace=True)
    test_data.replace("\d+", "NUM", regex=True, inplace=True)
    assert vectorizer in ["count_vectorizer", "tfidf"], "Unrecognized vectorizer"
    # vectorizing title and abstract, removing stop words, lowercasing, stripping accents, and ngrams
    if vectorizer == "count_vectorizer":
        logger.info("Vectorizing training set")
        vec_title = CountVectorizer(
            stop_words="english", strip_accents="ascii", ngram_range=(1, 2)
        )
        vec_abstract = CountVectorizer(
            stop_words="english", strip_accents="ascii", ngram_range=(1, 4), min_df=0.01
        )
        train_title = vec_title.fit_transform(training_data.title)
        train_abstract = vec_abstract.fit_transform(training_data.abstract)
    elif vectorizer == "tfidf":
        logger.info("Vectorizing training set")
        vec_title = TfidfVectorizer(
            stop_words="english",
            strip_accents="ascii",
            ngram_range=(1, 2),
            min_df=0.001,
            max_features=1000
        )
        vec_abstract = TfidfVectorizer(
            stop_words="english", strip_accents="ascii", ngram_range=(1, 4), min_df=0.01, max_features=1000
        )
        train_title = vec_title.fit_transform(training_data.title)
        train_abstract = vec_abstract.fit_transform(training_data.abstract)
    # training set
    processed_train = hstack((train_title, train_abstract))
    processed_train = pd.DataFrame.sparse.from_spmatrix(processed_train)
    processed_training_set = pd.concat(
        [training_data[["node_id"]], processed_train, training_data[["label"]]], axis=1
    )
    # test set
    logger.info("Vectorizing test set")
    test_title = vec_title.transform(test_data.title)
    test_abstract = vec_abstract.transform(test_data.abstract)
    processed_test = hstack((test_title, test_abstract))
    processed_test = pd.DataFrame.sparse.from_spmatrix(processed_test)
    processed_test_set = pd.concat(
        [test_data[["node_id"]], processed_test, test_data[["label"]]], axis=1
    )
    return processed_training_set, processed_test_set


def train(model, training_set: pd.DataFrame):
    x_train, y_train = training_set.iloc[:, :-1], training_set.iloc[:, -1]
    logger.info("Training model")
    time_start = time.time()
    clf = model.fit(x_train, y_train)
    time_elapsed = time.time() - time_start
    logger.info(
        f"Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.2f}s"
    )
    logger.info("Saving model")
    save_model(clf, model.__class__.__name__)


def predict(model_name: str, test_data: pd.DataFrame):
    logger.info("Loading model")
    clf = load_model(model_name)
    x_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    logger.info("Model inference")
    predictions = clf.predict(x_test)
    acc = np.mean(predictions == y_test)
    logger.info(f"Model accuracy : {acc}")


def main():
    training_set, local_test_set, _ = make_sets(
        "/data/train.csv",
        "/data/text.csv",
        "/data/nodeid2paperid.csv",
        "/data/test.csv",
        0.1,
    )
    processed_training_set, processed_test_set = text_preprocessing(
        training_set, local_test_set, "tfidf"
    )
    model = SVC(C=0.9, random_state=42)
    train(model, processed_training_set)
    predict("SVC", processed_test_set)


if __name__ == "__main__":
    main()
