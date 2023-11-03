import numpy as np


def load_words(path):
    """
    Open line break separated words as list

    :param path: file with words path
    :return: list of words
    """

    with open(path, 'r') as f:
        return f.read().split('\n')[:-1]


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
