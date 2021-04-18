[![codecov](https://codecov.io/gh/rmartinshort/generic_search/branch/main/graph/badge.svg)](https://codecov.io/gh/rmartinshort/generic_search)
[![Build Status](https://travis-ci.com/rmartinshort/generic_search.svg?branch=main)](https://travis-ci.com/rmartinshort/generic_search)

# Generic Search

---

A generic search engine tool that allows fast fuzzy matching on a corpus stored in memory

This is designed to be an easy-to-use search engine that can be trained and applied to any corpus. It can use gensim fasttext or doc2vec to
train a vector model for the corpus, bm25 to weight the vectors and nsmlib to create a mathing index. Alternatively, it can use sklearn's TfidfVectorizer to build
a sparse matrix model for the corpus.

Once this is done,
users can type a query and the n closest matching elements of the original corpus will be returned.

There is also the ability to train on one corpus and then match on another, which can be useful when building doc2vec or fasttext models that aim to learn the similarities between
words though context and so generally requre large bodies of text to work on.

## Quickstart

---


An example setup is as follows. Note that for large problems you can save the vectorizer, vectorized corpus and index to
file to prevent the need to regenerate them.

```python
from generic_search.engine.search_engine import SearchEngine 
from gensim.models.fasttext import FastText 
import pandas as pd 
import logging

logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
# Provide the path to some dataset, here the classic IMDB movie dataset is used
dataset = pd.read_csv("generic_search/fixtures/IMDB Dataset.csv")
corpus = dataset["review"].tolist()
# Some of the reviews are very long. For this example we'll make the problem easier by just 
# considering the first 10 words of each
corpus = [" ".join(x.split()[:10]) for x in corpus]
# Set up the gensim model you want to use for the vectorization. 

# Currently supports FastText, Doc2Vec and Tfidf models. See tests for examples of each
my_model = FastText(
        sg=1,  # use skip-gram method, usually desirable (other option is CBOW)
        size=50,  # embedding dimension 
        window=10,  # window size: 10 tokens before and 10 tokens after to get wider context
        min_count=5,  # only consider tokens with at least n occurrences in the corpus
        negative=15,  # negative subsampling
        min_n=2,  # min character n-gram
        max_n=10  # max character n-gram
        )

# Build vectorizer and index (can be slow to train on large corpi)
search_engine = SearchEngine.build_model(
                    corpus=corpus,
                    vector_model=my_model,
                    n_epochs=10,
                    save_location="test")
# Make a query
search_engine.suggest("my query",n_return=10)

# Load the vectoirizer and index again (should be fast)
loaded_search_engine = SearchEngine.load_model_from_files(
                    original_matching_list=corpus,
                    vector_model_file="test/_fasttext.model",
                    vector_model_type="fasttext",
                    vectorized_corpus="test/weighted_doc_vects.p",
                    matcher="/test/saved_matcher.bin"
)

```

## To do

---
- Add more tests
- Add support for non-english corpus via spacy

## References

---

Excellent article that inspired this repo
https://towardsdatascience.com/how-to-build-a-smart-search-engine-a86fca0d0795  
Gensim FastText documentation   
https://radimrehurek.com/gensim/models/fasttext.html  
NSMLIB repo  
https://github.com/nmslib/nmslib

# Questions

---

Please send questions, requests or bug reports to martinshortr@gmail.com
