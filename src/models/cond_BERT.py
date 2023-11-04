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
    """
    Class to hold CondBERT functionality
    """

    def __init__(self, model, tokenizer, device, tok_toxicities, word2coef, ft):
        """
        :param model: Masked LM to be used
        :param tokenizer: tokenizer to be used
        :param device: cuda or cpu
        :param tok_toxicities: array, containing toxicity scores of all tokens
        :param word2coef: dict, containing logistic regression coefficients for words in a training dataset.
        :param ft: FastText object to find similarity between words
        """

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tok_toxicities = tok_toxicities
        self.word2coef = word2coef
        self.ft = ft

        # those stats are needed when evaluating word's toxicity score
        self.tox_stats = tok_toxicities.min(), tok_toxicities.max()
        w2c = np.array(list(self.word2coef.values()))
        self.w2c_stats = w2c.min(), w2c.max()

    def mask(self, sentences, threshold, min_words):
        """
        Mask (censor) each sentence

        :param sentences: list of sentences
        :param threshold: threshold in range [0, 1]. lower threshold => more words masked
        :param min_words: if no words above threshold -> mask min_words words with maximum toxicity score
        :return: list of token ids and list of masks for each sentence
        """

        all_ids = []
        masks = []
        for text in sentences:
            # encode sentence
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(ids)

            # contains toxicity score for each token in a sentence
            tox_scores = []
            # ordered dict, containing score for each word in a sentence
            words = []
            for i, (id, tok) in enumerate(zip(ids, tokens)):
                # skip special symbols
                if tok.startswith('['):
                    tox_scores.append(0)
                    continue

                # tox_score = toxicity score
                tox_score = (self.tok_toxicities[id] - self.tox_stats[0]) / (self.tox_stats[1] - self.tox_stats[0])
                if not tok.startswith('##'):
                    # if word found in dict -> tox_score = mean of word2coef and token toxicities scores
                    if tok in self.word2coef:
                        coef_score = (self.word2coef[tok] - self.w2c_stats[0]) / (self.w2c_stats[1] - self.w2c_stats[0])
                        tox_score = (tox_score + coef_score) / 2
                    words.append([tox_score, [i]])
                else:
                    if i == 0:
                        raise ValueError(f'Something went wrong with: {text}')
                    else:
                        # suffix case, they should have score more or equal to the previous word parts
                        tox_score = max(tox_score, tox_scores[-1])
                        words[-1][0] = tox_score
                        words[-1][1].append(i)
                tox_scores.append(tox_score)

            # if tox_score above threshold -> mask that token
            tox_scores = torch.tensor(tox_scores)
            mask = (tox_scores > threshold).to(self.device)

            # if no toxic words detected -> mask min_words words with the highest tox_score
            if not any(mask):
                if min_words < len(words) - 1:
                    words = sorted(words, key=lambda x: x[0])[-min_words:]
                    mask[[item for sublist in words for item in sublist[1]]] = True
                else:
                    raise ValueError(f'min_words is too high, should be: < {len(words) - 1}')

            all_ids.append(torch.tensor(ids).to(self.device))
            masks.append(mask)

        return all_ids, masks

    def translate(self, ids, mask, top_n):
        """
        Replace masks, taking similarity between prediction and input into account

        :param ids: input sentence ids
        :param mask: sentence mask
        :param top_n: n words will be predicted by MLM and the most similar will be chosen
        :return: string, translated sentence
        """

        ids = ids[None, :]
        # get indexes of masks
        mask_ixs = torch.arange(mask.shape[0]).to(self.device)[mask]

        # compute MLM logits
        logits = self.model(ids, token_type_ids=torch.ones_like(ids)).logits

        # toxic penalty
        logits[0, :] -= torch.tensor(self.tok_toxicities).to(self.device) * 1.5

        for mask_index in mask_ixs:
            # same word penalty
            logits[0, mask_index, ids[0][mask_index]] -= 10

            mask_token_logits = logits[0, mask_index, :]
            # pick top n tokens
            top_tokens = torch.topk(mask_token_logits, top_n, dim=0)
            # convert tokens to words
            words = [self.tokenizer.decode([top_tokens.indices[i]]) for i in range(top_tokens[0].shape[0])]
            orig = self.tokenizer.decode([ids[0, mask_index]])
            # find similarity between input and prediction
            similarities = [cosine_similarity(self.ft[orig], self.ft[w]) for w in words]
            # select prediction with the highest similarity
            ids[0][mask_index] = top_tokens.indices[np.argmax(similarities)]

        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    def detox(self, sentences, detect_threshold=0.2, detect_min_words=1, top_n=20):
        """
        Interface, connecting self.mask() and self.translate()

        :param sentences: sentences to be detoxed
        :param detect_threshold: threshold value of self.mask()
        :param detect_min_words: min_words value of self.mask()
        :param top_n: top_n value of self.translate()
        :return: list of detoxed words
        """

        if is_str := isinstance(sentences, str):
            sentences = [sentences]

        # mask sentences
        all_ids, masks = self.mask(sentences, threshold=detect_threshold, min_words=detect_min_words)

        # translate sentences
        detoxed = []
        for ids, mask in zip(all_ids, masks):
            detoxed.append(self.translate(ids, mask, top_n=top_n))

        if is_str:
            return detoxed[0]
        else:
            return detoxed


def load_condBERT(model_name='bert-base-uncased', vocab_dirname=None):
    """
    CondBERT class builder to create object with default parameters

    :param model_name: transformers model name
    :param vocab_dirname: location of vocabulary
    :return: CondBERT object
    """

    if vocab_dirname is None:
        vocab_dirname = VOCAB_DIRNAME

    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tok_toxicities = load_toxicities(vocab_dirname / 'token_toxicities.txt')
    word2coef = load_word2coef(vocab_dirname / 'word2coef.pkl')
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

    return CondBERT(model, tokenizer, device, tok_toxicities, word2coef, ft)
