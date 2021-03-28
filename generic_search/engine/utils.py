"""
Utility functions to support search_engine codes
"""

from gensim.models.callbacks import CallbackAny2Vec
import re


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def ngrams_chars(string, n=3):
    """Takes an input string, cleans it and converts to ngrams"""

    string = string.lower()  # lower case
    string = string.encode("ascii", errors="ignore").decode()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'", "-"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = re.sub(' +', ' ', string).strip()
    string = ' ' + string + ' '

    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]
