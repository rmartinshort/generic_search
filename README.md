# Generic Search

---

A generic search engine tool that allows fast fuzzy matching on a corpus stored in memory

This is designed to be an easy-to-use search engine that can be trained and applied to any corpus. It uses gensim to train a vector model for the corpus, bm25 to weight the vectors and nsmlib to create a mathing index. Once this is done, users can type a query and the n closest matching elements of the original corpus will be returned.

## Quickstart 

---


An example setup is as follows. Note that for large problems you can save the vectorizer, vectorized corpus and index to file to
prevent the need to regenerate them. 

```python
from generic_search.engine.search_engine import SearchEngine 
from gensim.models.fasttext import FastText 
import pandas as pd 
import logging

logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
# Provide the path to some dataset, here the classic IMDB movie dataset is used
dataset = pd.read_csv("generic_search/test_datasets/IMDB Dataset.csv")
corpus = dataset["review"].tolist()
# Some of the reviews are very long. For this example we'll make the problem easier by just 
# considering the first 10 words of each
corpus = [" ".join(x.split()[:10]) for x in corpus]
# Set up the gensim model you want to use for the vectorization. 
# Currently supports FastText and Doc2Vec models
my_model = FastText(
        sg=1,  # use skip-gram method, usually desirable (other option is CBOW)
        size=50,  # embedding dimension 
        window=10,  # window size: 10 tokens before and 10 tokens after to get wider context
        min_count=5,  # only consider tokens with at least n occurrences in the corpus
        negative=15,  # negative subsampling
        min_n=2,  # min character n-gram
        max_n=10  # max character n-gram
        )

# Build vectorizer and index
search_engine = SearchEngine.build_model(
                    corpus=corpus,
                    vector_model=my_model,
                    n_epochs=10)
# Make a query
search_engine.suggest("my query",n_return=10)

```

## To do

---
- Improve efficiency of tokenizer
- Add test cases
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
