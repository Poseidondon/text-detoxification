def load_words(path):
    """
    Open line break separated words as list

    :param path: file with words path
    :return: list of words
    """

    with open(path, 'r') as f:
        return f.read().split('\n')[:-1]
