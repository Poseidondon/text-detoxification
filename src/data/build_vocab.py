import torch
import os
import pickle

from transformers import BertTokenizer
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_DIRNAME = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'vocab'


def build_vocab(model_name='bert-base-uncased'):
    """
    Build vocabulary, required for CondBERT.
    This vocabulary includes toxicity scores for each model token and logistic regression coefficients.

    :param model_name: model to build vocabulary for
    """

    # some paths and globals
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # datasets paths
    tox_corpus_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'train' / 'train_toxic.txt'
    norm_corpus_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'train' / 'train_normal.txt'

    # create vocab dir if not exists
    if not os.path.exists(VOCAB_DIRNAME):
        os.makedirs(VOCAB_DIRNAME)

    # count each words occurrences in the dataset
    c = Counter()
    for fn in [tox_corpus_path, norm_corpus_path]:
        with open(fn, 'r') as corpus:
            for line in corpus.readlines():
                for tok in line.strip().split():
                    c[tok] += 1
    vocab = {w for w, _ in c.most_common() if _ > 0}

    # read words from toxic and normal datasets
    with open(tox_corpus_path, 'r') as tox_corpus, open(norm_corpus_path, 'r') as norm_corpus:
        corpus_tox = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in
                      tox_corpus.readlines()]
        corpus_norm = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in
                       norm_corpus.readlines()]

    # evaluating coefficients for words in dataset
    pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
    X_train = corpus_tox + corpus_norm
    y_train = [1] * len(corpus_tox) + [0] * len(corpus_norm)
    pipe.fit(X_train, y_train)
    coefs = pipe[1].coef_[0]

    # write coefficients
    word2coef = {w: coefs[idx] for w, idx in pipe[0].vocabulary_.items()}
    with open(VOCAB_DIRNAME / 'word2coef.pkl', 'wb') as f:
        pickle.dump(word2coef, f)

    # count words occurrences in toxic and normal datasets
    toxic_counter = defaultdict(lambda: 1)
    nontoxic_counter = defaultdict(lambda: 1)
    for text in tqdm(corpus_tox):
        for token in tokenizer.encode(text):
            toxic_counter[token] += 1
    for text in tqdm(corpus_norm):
        for token in tokenizer.encode(text):
            nontoxic_counter[token] += 1

    # find ratio for each word: toxic_count / all_count
    token_toxicities = [toxic_counter[i] / (nontoxic_counter[i] + toxic_counter[i]) for i in range(len(tokenizer.vocab))]
    with open(VOCAB_DIRNAME / 'token_toxicities.txt', 'w') as f:
        for t in token_toxicities:
            f.write(str(t))
            f.write('\n')


if __name__ == '__main__':
    build_vocab()
