import torch
import os
import numpy as np

from transformers import BertTokenizer
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_DIRNAME = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'vocab'


class NgramSalienceCalculator():
    def __init__(self, tox_corpus, norm_corpus, use_ngrams=False):
        ngrams = (1, 3) if use_ngrams else (1, 1)
        self.vectorizer = CountVectorizer(ngram_range=ngrams)

        tox_count_matrix = self.vectorizer.fit_transform(tox_corpus)
        self.tox_vocab = self.vectorizer.vocabulary_
        self.tox_counts = np.sum(tox_count_matrix, axis=0)

        norm_count_matrix = self.vectorizer.fit_transform(norm_corpus)
        self.norm_vocab = self.vectorizer.vocabulary_
        self.norm_counts = np.sum(norm_count_matrix, axis=0)

    def salience(self, feature, attribute='tox', lmbda=0.5):
        assert attribute in ['tox', 'norm']
        if feature not in self.tox_vocab:
            tox_count = 0.0
        else:
            tox_count = self.tox_counts[0, self.tox_vocab[feature]]

        if feature not in self.norm_vocab:
            norm_count = 0.0
        else:
            norm_count = self.norm_counts[0, self.norm_vocab[feature]]

        if attribute == 'tox':
            return (tox_count + lmbda) / (norm_count + lmbda)
        else:
            return (norm_count + lmbda) / (tox_count + lmbda)


def build_vocab(model_name='bert-base-uncased'):
    # some paths and globals
    tokenizer = BertTokenizer.from_pretrained(model_name)

    tox_corpus_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'train' / 'train_toxic.txt'
    norm_corpus_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'train' / 'train_normal.txt'

    if not os.path.exists(VOCAB_DIRNAME):
        os.makedirs(VOCAB_DIRNAME)

    # saving positive and negative words
    c = Counter()

    for fn in [tox_corpus_path, norm_corpus_path]:
        with open(fn, 'r') as corpus:
            for line in corpus.readlines():
                for tok in line.strip().split():
                    c[tok] += 1
    vocab = {w for w, _ in c.most_common() if _ > 0}

    with open(tox_corpus_path, 'r') as tox_corpus, open(norm_corpus_path, 'r') as norm_corpus:
        corpus_tox = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in
                      tox_corpus.readlines()]
        corpus_norm = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in
                       norm_corpus.readlines()]

    neg_out_name = VOCAB_DIRNAME / 'negative-words.txt'
    pos_out_name = VOCAB_DIRNAME / 'positive-words.txt'
    threshold = 4

    sc = NgramSalienceCalculator(corpus_tox, corpus_norm, False)
    seen_grams = set()

    with open(neg_out_name, 'w') as neg_out, open(pos_out_name, 'w') as pos_out:
        for gram in set(sc.tox_vocab.keys()).union(set(sc.norm_vocab.keys())):
            if gram not in seen_grams:
                seen_grams.add(gram)
                toxic_salience = sc.salience(gram, attribute='tox')
                polite_salience = sc.salience(gram, attribute='norm')
                if toxic_salience > threshold:
                    neg_out.writelines(f'{gram}\n')
                elif polite_salience > threshold:
                    pos_out.writelines(f'{gram}\n')

    # evaluating words toxicities
    pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
    X_train = corpus_tox + corpus_norm
    y_train = [1] * len(corpus_tox) + [0] * len(corpus_norm)
    pipe.fit(X_train, y_train)
    coefs = pipe[1].coef_[0]

    word2coef = {w: coefs[idx] for w, idx in pipe[0].vocabulary_.items()}
    import pickle
    with open(VOCAB_DIRNAME / 'word2coef.pkl', 'wb') as f:
        pickle.dump(word2coef, f)

    toxic_counter = defaultdict(lambda: 1)
    nontoxic_counter = defaultdict(lambda: 1)
    for text in tqdm(corpus_tox):
        for token in tokenizer.encode(text):
            toxic_counter[token] += 1
    for text in tqdm(corpus_norm):
        for token in tokenizer.encode(text):
            nontoxic_counter[token] += 1

    token_toxicities = [toxic_counter[i] / (nontoxic_counter[i] + toxic_counter[i]) for i in range(len(tokenizer.vocab))]
    with open(VOCAB_DIRNAME / 'token_toxicities.txt', 'w') as f:
        for t in token_toxicities:
            f.write(str(t))
            f.write('\n')


if __name__ == '__main__':
    build_vocab()
