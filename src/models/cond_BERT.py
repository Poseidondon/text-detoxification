import numpy as np
import torch
import sys
sys.path.append("..")

from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
from data.load_vocab import load_toxicities, load_word2coef

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_DIRNAME = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'vocab'


class CondBERT:
    def __init__(self, model, tokenizer, device, tok_toxicities, word2coef):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tok_toxicities = tok_toxicities
        self.word2coef = word2coef

        self.tox_stats = tok_toxicities.min(), tok_toxicities.max()
        w2c = np.array(list(self.word2coef.values()))
        self.w2c_stats = w2c.min(), w2c.max()

    def mask(self, text, threshold=0.4, min_words=1):
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        tox_scores = []
        words = []
        for i, (id, tok) in enumerate(zip(ids, tokens)):
            # skip special symbols
            if tok.startswith('['):
                tox_scores.append(0)
                continue

            tox_score = (self.tok_toxicities[id] - self.tox_stats[0]) / (self.tox_stats[1] - self.tox_stats[0])
            if not tok.startswith('##'):
                if tok in self.word2coef:
                    coef_score = (self.word2coef[tok] - self.w2c_stats[0]) / (self.w2c_stats[1] - self.w2c_stats[0])
                    tox_score = (tox_score + coef_score) / 2
                words.append([tox_score, [i]])
            else:
                if i == 0:
                    raise ValueError(f'Something went wrong with: {text}')
                else:
                    tox_score = max(tox_score, tox_scores[-1])
                    words[-1][0] = tox_score
                    words[-1][1].append(i)
            tox_scores.append(tox_score)

        # if no toxic words detected -> mask words with most tox_score
        tox_scores = torch.tensor(tox_scores)
        mask = (tox_scores > threshold).to(self.device)
        if not any(mask):
            if min_words < len(words) - 1:
                words = sorted(words, key=lambda x: x[0])[-min_words:]
                mask[[item for sublist in words for item in sublist[1]]] = True
            else:
                raise ValueError(f'min_words is too high, should be: < {len(words) - 1}')

        return torch.tensor(ids).to(self.device), mask


def load_condBERT(model_name='bert-base-uncased', vocab_dirname=None):
    if vocab_dirname is None:
        vocab_dirname = VOCAB_DIRNAME

    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tok_toxicities = load_toxicities(vocab_dirname / 'token_toxicities.txt')
    word2coef = load_word2coef(vocab_dirname / 'word2coef.pkl')

    return CondBERT(model, tokenizer, device, tok_toxicities, word2coef)

