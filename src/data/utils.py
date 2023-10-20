def load_words(path):
    with open(path, 'r') as f:
        return f.read().split('\n')[:-1]
