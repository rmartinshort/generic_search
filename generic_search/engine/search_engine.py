"""
Search engine classes
"""

import logging
import os
import pickle
import joblib

import gensim
import nmslib
import numpy as np
import pandas as pd
import spacy
from generic_search.engine.utils import EpochLogger, ngrams_chars
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sklearn
from gensim.models.fasttext import FastText
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class SearchEngine(object):

    def __init__(self):

        """

        """

        self.spacy_model = spacy.load("en_core_web_sm")
        self.epoch_logger = EpochLogger()
        self.corpus = None
        self.matching_list = None
        self.matching_list_is_corpus = False
        self.vector_model = None
        self.n_lim = None
        self.matcher = None
        self.vector_model_type = None

    @classmethod
    def build_model(cls, corpus, vector_model=None, matching_list=None, n_epochs=5, limit_docs=1e6, save_location=None):

        """

        Has support for fasttext and doc2vec models. Use as follows

        # For FastText
        ft_model = FastText(
        sg=1,  # use skip-gram: usually gives better results
        size=50,  # embedding dimension (should be the same as the GLOVE vectors that are being used, so 50)
        window=10,  # window size: 10 tokens before and 10 tokens after to get wider context
        min_count=1,  # only consider tokens with at least n occurrences in the corpus
        negative=15,  # negative subsampling: bigger than default to sample negative examples more
        min_n=1,  # min character n-gram
        max_n=5  # max character n-gram
        )

        # For Doc2Vec
        doc_model = Doc2Vec(
        vector_size=50,
        window=3,
        epochs=10,
        min_count=1,
        negative=15,
        workers=4)

        :param corpus:
        :param vector_model:
        :param n_epochs:
        :param limit_docs:
        :param save_location:
        :return:
        """

        if save_location:
            if not os.path.isdir(save_location):
                os.mkdir(save_location)

        s = SearchEngine()
        s.corpus = corpus
        s.n_lim = limit_docs

        if not matching_list:
            #If this is the case, we want to match on the same corpus as we trained on
            s.matching_list = corpus
            s.matching_list_is_corpus = True
        else:
            # If this is the case, we train on one corpus and match on another
            s.matching_list = matching_list

        s.vector_model = vector_model
        corpus_weight_vectors, model_type = s.generate_vectorized_corpus(n_epochs=n_epochs, save_location=save_location)

        s.matcher = s.generate_matcher(corpus_weight_vectors, save_location=save_location, model_type=model_type)

        return s

    @classmethod
    def load_model_from_files(cls, original_matching_list, vector_model_file, vector_model_type, vectorized_corpus,
                              matcher=None):

        """

        Parameters
        ----------
        original_corpus_file
        vector_model_file
        vectorized_corpus
        matcher

        Returns
        -------

        """

        assert vector_model_type in ["fasttext", "doc2vec", "tfidf"]
        assert isinstance(original_matching_list,list)

        s = SearchEngine()

        logging.info("Loading vector model")
        s.corpus = original_matching_list
        s.matching_list = original_matching_list
        s.matching_list_is_corpus = True

        if vector_model_type == "fasttext":
            s.vector_model = FastText.load(vector_model_file)
        elif vector_model_type == "doc2vec":
            s.vector_model = Doc2Vec.load(vector_model_file)
        elif vector_model_type == "tfidf":
            s.vector_model = joblib.load(vector_model_file)

        s.vector_model_type = vector_model_type

        logging.info("Loading vectorized corpus")
        with open(vectorized_corpus, "rb") as f:
            vectorized_corpus = pickle.load(f)

        logging.info("Generating matcher index")
        if matcher:
            if vector_model_type == "tfidf":
                # Cannot load and save matchers of this type, but they are fast to generate
                s.matcher = s.generate_matcher(vectorized_corpus, save_location=None)
            else:
                new_matcher = nmslib.init(method='hnsw', space='cosinesimil')

            new_matcher.loadIndex(matcher, load_data=False)
            s.matcher = new_matcher
        else:
            s.matcher = s.generate_matcher(vectorized_corpus, save_location=None)

        return s

    def suggest(self, input_query, n_return=10):

        """

        Parameters
        ----------
        input_query : str
            User input query
        n_return : int, optional
            The number of matches to return

        Returns
        -------
        pandas.DataFrame
            The top n matches using the search
        """

        query_parts = self._preprocess(input_query)

        if self.vector_model_type == "tfidf":
            query = self.vector_model.transform([query_parts])
            ids, distances = self.matcher.knnQueryBatch(query, k=n_return)[0]
        else:
            query = [self.vector_model[w] for w in query_parts.split()]
            query = np.mean(query, axis=0)
            ids, distances = self.matcher.knnQuery(query, k=n_return)

        # Display results of a search
        distances_list = [None] * len(distances)
        results_list = [None] * len(ids)
        original_query_list = [None] * len(ids)
        k = 0
        for i, j in zip(ids, distances):
            distances_list[k] = j
            results_list[k] = self.matching_list[i]
            original_query_list[k] = input_query
            k += 1

        return pd.DataFrame({"query": original_query_list, "score": distances_list, "match": results_list})

    def generate_vectorized_corpus(self, n_epochs=5, save_location=None):

        """

        Parameters
        ----------
        n_epochs : int, optional
            The number of training epochs for self.vector_model
        save_location : str, optional
            The path the a directory where the model files will be written. No files
            will be written if this is set to None

        Returns
        -------
        np.ndarray
            Vecotized corpus


        """

        if isinstance(self.vector_model, sklearn.feature_extraction.text.TfidfVectorizer):
            model_type = "tfidf"
        elif isinstance(self.vector_model, gensim.models.fasttext.FastText):
            model_type = "fasttext"
        elif isinstance(self.vector_model, gensim.models.doc2vec.Doc2Vec):
            model_type = "doc2vec"
        else:
            raise Warning("vector model must be of type gensim FastText, genism Doc2Vec or sklearn TfidfVectorizer")
        self.vector_model_type = model_type

        logging.info("Tokenizing text")
        tokenized_text = self._tokenize(model_type=model_type)

        if model_type == "tfidf":
            model_text_feed = tokenized_text
            self.vector_model.fit(model_text_feed)

            if self.matching_list_is_corpus:
                self.matching_list = self.corpus
            else:
                model_text_feed = self._tokenize_match_list(model_type="tfidf")
            weight_vectors = self.vector_model.transform(model_text_feed)

        else:

            if model_type == "fasttext":
                model_text_feed = tokenized_text
            elif model_type == "doc2vec":
                model_text_feed = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_text)]

            logging.info("Building vector vocab")
            self._build_vector_vocab(model_text_feed)

            logging.info("Training word vector model")
            self._train_vector_model(model_text_feed, n_epochs=n_epochs)

            logging.info("Generating BM25 weights")
            if self.matching_list_is_corpus:
                self.matching_list = self.corpus
                weight_vectors = self._assign_weights(tokenized_text)
            else:
                tokenized_text = self._tokenize_match_list(model_type=model_type)
                weight_vectors = self._assign_weights(tokenized_text)

        if save_location:
            if not os.path.isdir(save_location):
                os.mkdir(save_location)

            logging.info("Saving model")
            if model_type == "fasttext":
                self.vector_model.save(os.path.join(save_location, "_fasttext.model"))
            elif model_type == "doc2vec":
                self.vector_model.save(os.path.join(save_location, "_doc2vec.model"))
            else:
                joblib.dump(self.vector_model,os.path.join(save_location, "_tfidf.model"))

            f = open(os.path.join(save_location, "weighted_doc_vects.p"), "wb")
            pickle.dump(weight_vectors, f)
            f.close()

        return weight_vectors, model_type

    def _tokenize(self, model_type="tfidf"):

        """

        Returns
        -------

        """

        if not self.n_lim:
            self.n_lim = len(self.corpus) - 1
        else:
            self.n_lim = int(self.n_lim)

        # apply preprocessing to remove punctuation e stc if present
        corpus = [self._preprocess(x) for x in self.corpus]

        if model_type in ["fasttext","doc2vec"]:

            tokenized_text = []
            for doc in tqdm(self.spacy_model.pipe(corpus, disable=["lemmatizer", "tagger", "parser", "ner"])):
                tok = [t.text for t in doc if (t.is_ascii and not t.is_punct and not t.is_space)]
                tokenized_text.append(tok)

        else:
            tokenized_text = corpus

        return tokenized_text

    def _tokenize_match_list(self, model_type="tfidf"):

        """

        Returns
        -------

        """

        # apply preprocessing to remove punctuation e stc if present
        corpus = [self._preprocess(x) for x in self.matching_list]

        if model_type in ["fasttext","doc2vec"]:

            tokenized_text = []
            for doc in tqdm(self.spacy_model.pipe(corpus, disable=["lemmatizer", "tagger", "parser", "ner"])):
                tok = [t.text for t in doc if (t.is_ascii and not t.is_punct and not t.is_space)]
                tokenized_text.append(tok)

        else:
            tokenized_text = corpus

        return tokenized_text

    def _build_vector_vocab(self, tokenized_text):

        """

        Parameters
        ----------
        tokenized_text

        Returns
        -------

        """

        self.vector_model.build_vocab(tokenized_text[:self.n_lim])

    def _train_vector_model(self, tokenized_text, n_epochs=5):

        """

        Parameters
        ----------
        tokenized_text : list of list of str
            A corpys where each element is a doc and the docs are tokenized
        n_epochs : int, optional
            The number of training epochs for the vector model

        Returns
        -------
        Updates self.vector_model
        """

        self.vector_model.train(
            tokenized_text[:self.n_lim],
            epochs=n_epochs,
            total_examples=self.vector_model.corpus_count,
            total_words=self.vector_model.corpus_total_words,
            callbacks=[self.epoch_logger])

    def _assign_weights(self, tokenized_text):

        """

        Parameters
        ----------
        tokenized_text

        Returns
        -------

        """

        text = tokenized_text[:self.n_lim]
        bm25 = BM25Okapi(text)
        weighted_doc_vects = [None] * len(text)

        for i, doc in tqdm(enumerate(tokenized_text[:self.n_lim])):
            doc_vector = [None] * len(doc)
            for j, word in enumerate(doc):
                vector = self.vector_model[word]
                weight = (bm25.idf[word] * ((bm25.k1 + 1.0) * bm25.doc_freqs[i][word])) / (
                        bm25.k1 * (1.0 - bm25.b + bm25.b * (bm25.doc_len[i] / bm25.avgdl)) + bm25.doc_freqs[i][
                    word])
                weighted_vector = vector * weight

                doc_vector[j] = weighted_vector

            if len(doc_vector) == 0:
                weighted_doc_vects[i] = np.zeros(self.vector_model.vector_size)
            else:
                weighted_doc_vects[i] = np.mean(doc_vector, axis=0)

        return np.vstack(weighted_doc_vects)

    @staticmethod
    def generate_matcher(vectorized_corpus, save_location=None, model_type="tfidf", **kwargs):

        """
        Build a new nmslib matcher using a vectorized corpus. Note that the nmslib paramters
        are currently hardcoded if not specified

        Parameters
        ----------
        vectorized_corpus : np.ndarray
        save_location : str, optional
        **kwargs
        Keyword arguments to be fed into nmslib. The default values are often sufficient

        Returns
        -------
            nmslib index object
        """

        if model_type == "tfidf":

            # This is the case if we have a tfidf matcher, where the params have
            # already be specified
            nmslib_method = kwargs.get("method", "simple_invindx")
            nmslib_space = kwargs.get("space", "negdotprod_sparse_fast")
            nmslib_data_type = kwargs.get("data_type", nmslib.DataType.SPARSE_VECTOR)
            matcher_params = {}
            matcher = nmslib.init(method=nmslib_method, space=nmslib_space, data_type=nmslib_data_type)

        else:

            if isinstance(kwargs,dict):

                nmslib_method = kwargs.get("method", "hnsw")
                nmslib_space = kwargs.get("space", "cosinesimil")
                nmslib_M = kwargs.get("M", 30)
                nmslib_indexThreadQty = kwargs.get("indexThreadQty", 4)
                nmslib_efConstruction = kwargs.get("efConstruction", 100)
                nmslib_post = kwargs.get("post", 0)
                data_type = "auto"
                matcher_params = {'M': nmslib_M,
                              'indexThreadQty': nmslib_indexThreadQty,
                              'efConstruction': nmslib_efConstruction,
                              'post': nmslib_post}
            else:
                nmslib_method = "hnsw"
                nmslib_space = "cosinesimil"
                matcher_params = {'M': 30, 'indexThreadQty': 4, 'efConstruction': 100, 'post': 0}

            matcher = nmslib.init(method=nmslib_method, space=nmslib_space)


        matcher.addDataPointBatch(vectorized_corpus)
        matcher.createIndex(matcher_params, print_progress=True)

        if save_location and model_type != "tfidf":
            matcher.saveIndex(os.path.join(save_location, "saved_matcher.bin"), save_data=False)
            if model_type in ["fasttext","doc2vec"]:
                new_matcher = nmslib.init(method=nmslib_method, space=nmslib_space)
            elif model_type in ["tfidf"]:
                new_matcher = nmslib.init(method=nmslib_method, space=nmslib_space, data_type=nmslib_data_type)
            else:
                raise ValueError("model type unknown to matcher")
            new_matcher.loadIndex(os.path.join(save_location, "saved_matcher.bin"), load_data=False)
            return new_matcher
        else:
            return matcher

    def _preprocess(self, text):

        """

        Parameters
        ----------
        text

        Returns
        -------

        """

        # remove stopwords?
        return str(text).lower()
