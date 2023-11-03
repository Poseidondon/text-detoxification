import numpy as np
import pickle

from pathlib import Path

VOCAB_DIRNAME = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'vocab'


def load_toxicities(path=None, func=None):
    if path is None:
        path = VOCAB_DIRNAME / 'token_toxicities.txt'

    if func is None:
        # log odds ratio
        func = np.vectorize(lambda x: np.maximum(0, np.log(x / (1 - x))))

    token_toxicities = []
    with open(path, 'r') as f:
        for line in f.readlines():
            token_toxicities.append(float(line))

    token_toxicities = func(np.array(token_toxicities))

    return token_toxicities


def load_word2coef(path=None):
    if path is None:
        path = VOCAB_DIRNAME / 'word2coef.pkl'

    with open(path, 'rb') as f:
        word2coef = pickle.load(f)

    return word2coef
