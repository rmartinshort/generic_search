import collections
import os

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.fasttext import FastText
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from generic_search import SearchEngine
from generic_search.engine.utils import ngrams_chars

nltk.download('punkt')


def test_all_integration_fasttext():
    """

    Returns
    -------

    """

    d_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "IMDB Dataset.csv")

    dataset = pd.read_csv(d_path, nrows=10000)
    corpus = dataset["review"].tolist()
    corpus = [sent_tokenize(x)[0] for x in corpus]

    ft_model = FastText(
        sg=1,  # use skip-gram: usually gives better results
        size=50,  # embedding dimension (should be the same as the GLOVE vectors that are being used, so 50)
        window=10,  # window size: 10 tokens before and 10 tokens after to get wider context
        min_count=1,  # only consider tokens with at least n occurrences in the corpus
        negative=15,  # negative subsampling: bigger than default to sample negative examples more
        min_n=2,  # min character n-gram
        max_n=5,  # max character n-gram
    )

    save_location = os.path.join(os.path.dirname(__file__), "test_output_fasttext")

    se_test = SearchEngine.build_model(
        corpus=corpus,
        vector_model=ft_model,
        n_epochs=20,
        limit_docs=1e6,
        save_location=save_location)

    search_output = se_test.suggest("good")

    se_model_loaded = SearchEngine.load_model_from_files(
        original_matching_list=corpus,
        vector_model_file=os.path.join(save_location,"_fasttext.model"),
        vector_model_type="fasttext",
        vectorized_corpus=os.path.join(save_location,"weighted_doc_vects.p"),
        matcher=os.path.join(save_location,"saved_matcher.bin")
    )

    search_output_loaded = se_model_loaded.suggest("good")

    print("Full FASTTEXT output {}".format(search_output_loaded))

    assert len(search_output) == 10
    assert len(search_output_loaded) == 10
    assert collections.Counter(search_output.columns) == collections.Counter(["query", "score", "match"])
    assert collections.Counter(search_output_loaded.columns) == collections.Counter(["query", "score", "match"])
    assert search_output["match"].tolist()[0] == "This movie is good."
    assert search_output_loaded["match"].tolist()[0] == "This movie is good."


def test_all_integration_doc2vec():
    """

    Returns
    -------

    """

    d_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "IMDB Dataset.csv")

    dataset = pd.read_csv(d_path, nrows=10000)
    corpus = dataset["review"].tolist()
    corpus = [sent_tokenize(x)[0] for x in corpus]

    doc_model = Doc2Vec(
        vector_size=50,
        window=3,
        epochs=10,
        min_count=1,
        negative=15,
        workers=4)

    save_location = os.path.join(os.path.dirname(__file__), "test_output_doc2vec")

    se_test = SearchEngine.build_model(
        corpus=corpus,
        vector_model=doc_model,
        n_epochs=20,
        limit_docs=1e6,
        save_location=save_location)

    search_output = se_test.suggest("good")

    se_model_loaded = SearchEngine.load_model_from_files(
        original_matching_list=corpus,
        vector_model_file=os.path.join(save_location,"_doc2vec.model"),
        vector_model_type="doc2vec",
        vectorized_corpus=os.path.join(save_location,"weighted_doc_vects.p"),
        matcher=os.path.join(save_location,"saved_matcher.bin")
    )

    search_output_loaded = se_model_loaded.suggest("good")

    print("Full DOC2VEC output {}".format(search_output_loaded))

    assert len(search_output) == 10
    assert len(search_output_loaded) == 10
    assert collections.Counter(search_output.columns) == collections.Counter(["query", "score", "match"])
    assert collections.Counter(search_output_loaded.columns) == collections.Counter(["query", "score", "match"])
    assert search_output["match"].tolist()[0] == "This movie is good."
    assert search_output_loaded["match"].tolist()[0] == "This movie is good."


def test_all_integration_tfidf():
    """

    Returns
    -------

    """

    d_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "IMDB Dataset.csv")

    dataset = pd.read_csv(d_path, nrows=10000)
    corpus = dataset["review"].tolist()
    corpus = [sent_tokenize(x)[0] for x in corpus]

    tfidf_model = TfidfVectorizer(min_df=1, analyzer=ngrams_chars)

    save_location = os.path.join(os.path.dirname(__file__),"test_output_tfidf")

    se_test = SearchEngine.build_model(
        corpus=corpus,
        vector_model=tfidf_model,
        n_epochs=None,
        limit_docs=1e6,
        save_location=save_location)

    search_output = se_test.suggest("good")

    # Load the model from file
    se_model_loaded = SearchEngine.load_model_from_files(
        original_matching_list=corpus,
        vector_model_file=os.path.join(save_location,"_tfidf.model"),
        vector_model_type="tfidf",
        vectorized_corpus=os.path.join(save_location,"weighted_doc_vects.p"),
        matcher=None
    )

    search_output_loaded = se_model_loaded.suggest("good")

    print("Full TFIDF output {}".format(search_output_loaded))


    assert len(search_output) == 10
    assert len(search_output_loaded) == 10
    assert collections.Counter(search_output.columns) == collections.Counter(["query", "score", "match"])
    assert collections.Counter(search_output_loaded.columns) == collections.Counter(["query", "score", "match"])
    assert search_output["match"].tolist()[0] == "This movie is good."
    assert search_output_loaded["match"].tolist()[0] == "This movie is good."
