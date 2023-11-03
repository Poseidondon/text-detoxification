import numpy as np
import torch
import fasttext.util
import sys
sys.path.append("..")

from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
from data.load_vocab import load_toxicities, load_word2coef
from data.utils import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_DIRNAME = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'vocab'


class CondBERT:
    def __init__(self, model, tokenizer, device, tok_toxicities, word2coef, ft):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tok_toxicities = tok_toxicities
        self.word2coef = word2coef
        self.ft = ft

        self.tox_stats = tok_toxicities.min(), tok_toxicities.max()
        w2c = np.array(list(self.word2coef.values()))
        self.w2c_stats = w2c.min(), w2c.max()

    def mask(self, text, threshold=0.3, min_words=1):
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

    def translate(self, ids, mask, top_n=15):
        ids = ids[None, :]
        mask_ixs = torch.arange(mask.shape[0]).to(self.device)[mask]

        logits = self.model(ids, token_type_ids=torch.ones_like(ids)).logits

        # toxic penalty
        logits[0, :] -= torch.tensor(self.tok_toxicities).to(self.device) * 1.5

        for mask_index in mask_ixs:
            # same word penalty
            logits[0, mask_index, ids[0][mask_index]] -= 10

            mask_token_logits = logits[0, mask_index, :]
            top_tokens = torch.topk(mask_token_logits, top_n, dim=0)
            words = [self.tokenizer.decode([top_tokens.indices[i]]) for i in range(top_tokens[0].shape[0])]
            orig = self.tokenizer.decode([ids[0, mask_index]])
            similarities = [cosine_similarity(self.ft[orig], self.ft[w]) for w in words]
            ids[0][mask_index] = top_tokens.indices[np.argmax(similarities)]

        return [self.tokenizer.decode(s, skip_special_tokens=True) for s in ids]


def load_condBERT(model_name='bert-base-uncased', vocab_dirname=None):
    if vocab_dirname is None:
        vocab_dirname = VOCAB_DIRNAME

    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tok_toxicities = load_toxicities(vocab_dirname / 'token_toxicities.txt')
    word2coef = load_word2coef(vocab_dirname / 'word2coef.pkl')
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

    return CondBERT(model, tokenizer, device, tok_toxicities, word2coef, ft)


condBERT = load_condBERT()
text = "There is only one word to describe this - fuck..."
ids, mask = condBERT.mask(text)
toks = condBERT.tokenizer.convert_ids_to_tokens(ids)
print(toks)
print(mask)
print(condBERT.translate(ids, mask))
