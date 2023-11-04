import numpy as np

from pathlib import Path
from cond_BERT import load_condBERT
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


if __name__ == '__main__':
    # datasets paths
    tox_corpus_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'test' / 'test_toxic.txt'
    norm_corpus_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'test' / 'test_normal.txt'

    # read words from toxic and normal datasets
    with open(tox_corpus_path, 'r') as tox_corpus, open(norm_corpus_path, 'r') as norm_corpus:
        corpus_tox = tox_corpus.read().split('\n')[:-1]
        corpus_norm = norm_corpus.read().split('\n')[:-1]

    # condBERT initialization
    condBERT = load_condBERT()

    # sentence similarity model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # evaluating on toxic dataset
    corpus_tox_scores = np.zeros(len(corpus_tox))
    corpus_tox_detox_scores = np.zeros(corpus_tox_scores.shape[0])
    tox_similarities = np.zeros(corpus_tox_scores.shape[0])
    for i, s in tqdm(enumerate(corpus_tox), total=len(corpus_tox)):
        corpus_tox_scores[i] = condBERT.sentence_toxicity_score(s)
        detox_s = condBERT(s)
        corpus_tox_detox_scores[i] = condBERT.sentence_toxicity_score(detox_s)
        tox_similarities[i] = util.pytorch_cos_sim(model.encode(s), model.encode(detox_s))[0].item()

    print('Mean toxicity score before and after detoxification test_toxic.txt: ', end='')
    print(corpus_tox_scores.mean(), '-->', corpus_tox_detox_scores.mean())
    print('Mean similarity between input and prediction:', tox_similarities.mean())

    # evaluating on normal dataset
    corpus_norm_scores = np.zeros(len(corpus_norm))
    corpus_norm_detox_scores = np.zeros(corpus_norm_scores.shape[0])
    norm_similarities = np.zeros(corpus_tox_scores.shape[0])
    for i, s in tqdm(enumerate(corpus_norm), total=len(corpus_norm)):
        corpus_norm_scores[i] = condBERT.sentence_toxicity_score(s)
        detox_s = condBERT(s)
        corpus_norm_detox_scores[i] = condBERT.sentence_toxicity_score(detox_s)
        norm_similarities[i] = util.pytorch_cos_sim(model.encode(s), model.encode(detox_s))[0].item()

    print('Mean toxicity score before and after detoxification test_normal.txt (should not change a lot): ', end='')
    print(corpus_norm_scores.mean(), '-->', corpus_norm_detox_scores.mean())
    print('Mean similarity between input and prediction:', norm_similarities.mean())
