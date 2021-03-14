"""
Search engine classes
"""

import os
import string
import pandas as pd
import numpy as np
import spacy
import logging
import gensim
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from generic_search.engine.utils import EpochLogger
from tqdm import tqdm
import nmslib
from rank_bm25 import BM25Okapi
import pickle


class SearchEngine(object):

    def __init__(self):

        """

        """

        self.spacy_model = spacy.load("en_core_web_sm")
        self.epoch_logger = EpochLogger()
        self.corpus = None
        self.vector_model = None
        self.n_lim = None
        self.matcher = None

    @classmethod
    def build_model(cls, corpus, vector_model, n_epochs=5, limit_docs=1e6, save_location=None):

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
        s.vector_model = vector_model
        s.n_lim = limit_docs
        corpus_weight_vectors = s.generate_vectorized_corpus(n_epochs=n_epochs, save_location=save_location)
        s.matcher = s.generate_matcher(corpus_weight_vectors, save_location=save_location)

        return s

    @classmethod
    def load_model_from_files(cls, original_corpus_file, vector_model_file, vectorized_corpus, matcher=None):

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

        logging.info("Loading original corpus")

        try:
            df = pd.read_csv(original_corpus_file, usecols=["text"])
        except Exception as e:
            raise Warning("Provided dataframe may not have a text column!")

        assert "text" in df.columns

        s = SearchEngine()

        logging.info("Loading vector model")
        s.corpus = df["text"].str.lower().values
        s.vector_model = FastText.load(vector_model_file)

        logging.info("Loading vectorized corpus")
        with open(vectorized_corpus, "rb") as f:
            vectorized_corpus = pickle.load(f)

        if matcher:
            new_matcher = nmslib.init(method='hnsw', space='cosinesimil')
            new_matcher.loadIndex(matcher, load_data=False)
            s.matcher = new_matcher
        else:
            logging.info("Generating matcher index")
            s.matcher = s.generate_matcher(vectorized_corpus, save_location=None)

        return s

    def suggest(self, input_query, n_return=10):

        """

        Parameters
        ----------
        input_query
        n_return

        Returns
        -------

        """

        query_parts = self._preprocess(input_query)

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
            results_list[k] = self.corpus[i]
            original_query_list[k] = input_query
            k += 1

        return pd.DataFrame({"query": original_query_list, "score": distances_list, "match": results_list})

    def generate_vectorized_corpus(self, n_epochs=5, save_location=None):

        """

        Parameters
        ----------
        n_epochs
        save_location

        Returns
        -------

        """

        logging.info("Tokenizing text")
        tokenized_text = self._tokenize()

        if isinstance(self.vector_model,gensim.models.fasttext.FastText):
            model_type = "fasttext"
            model_text_feed = tokenized_text
        elif isinstance(self.vector_model,gensim.models.doc2vec.Doc2Vec):
            model_type = "doc2vec"
            model_text_feed = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_text)]
        else:
            raise Warning("vector model must be of type gensim FastText or genism Doc2Vec")

        logging.info("Building vector vocab")
        self._build_vector_vocab(model_text_feed)

        logging.info("Training word vector model")
        self._train_vector_model(model_text_feed, n_epochs=n_epochs)

        logging.info("Generating BM25 weights")
        corpus_weight_vectors = self._assign_weights(tokenized_text)

        if save_location:
            logging.info("Saving model")
            if model_type == "fasttext":
                self.vector_model.save(save_location + '/_fasttext.model')
            else:
                self.vector_model.save(save_location + '/_doc2vec.model')

            f = open(save_location + "/weighted_doc_vects.p", "wb")
            pickle.dump(corpus_weight_vectors, f)
            f.close()

        return corpus_weight_vectors

    def _tokenize(self):

        """

        Returns
        -------

        """

        tokenized_text = []

        if not self.n_lim:
            self.n_lim = len(self.corpus) - 1
        else:
            self.n_lim = int(self.n_lim)

        # apply preprocessing to remove punctuation e stc if present
        corpus = [self._preprocess(x) for x in self.corpus]

        for doc in tqdm(self.spacy_model.pipe(corpus, n_threads=2, disable=["tagger", "parser", "ner"])):
            tok = [t.text for t in doc if (t.is_ascii and not t.is_punct and not t.is_space)]
            tokenized_text.append(tok)


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
        tokenized_text
        n_epochs

        Returns
        -------

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

        bm25 = BM25Okapi(tokenized_text[:self.n_lim])
        weighted_doc_vects = []

        for i, doc in tqdm(enumerate(tokenized_text[:self.n_lim])):
            doc_vector = []
            for word in doc:

                vector = self.vector_model[word]
                weight = (bm25.idf[word] * ((bm25.k1 + 1.0) * bm25.doc_freqs[i][word])) / (
                            bm25.k1 * (1.0 - bm25.b + bm25.b * (bm25.doc_len[i] / bm25.avgdl)) + bm25.doc_freqs[i][
                        word])
                weighted_vector = vector * weight

                doc_vector.append(weighted_vector)


            if len(doc_vector) == 0:
                weighted_doc_vects.append(np.zeros(self.vector_model.vector_size))
            else:
                weighted_doc_vects.append(np.mean(doc_vector, axis=0))

        return np.vstack(weighted_doc_vects)

    @staticmethod
    def generate_matcher(vectorized_corpus, save_location=None):

        """

        Parameters
        ----------
        vectorized_corpus
        save_location

        Returns
        -------

        """

        # initialize a new index, using a HNSW index on Cosine Similarity
        matcher = nmslib.init(method='hnsw', space='cosinesimil')
        matcher.addDataPointBatch(vectorized_corpus)
        matcher.createIndex({'post': 2}, print_progress=True)

        if save_location:
            matcher.saveIndex(save_location + "/saved_matcher.bin", save_data=False)
            new_matcher = nmslib.init(method='hnsw', space='cosinesimil')
            new_matcher.loadIndex(save_location + "/saved_matcher.bin", load_data=False)
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