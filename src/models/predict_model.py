import torch
import sys
sys.path.append("..")

from transformers import AutoTokenizer, BertForMaskedLM
from data.utils import load_words
from pathlib import Path
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Iterable, Union


def get_tokens(tokenizer, words: list):
    words = tokenizer(' '.join(words), return_tensors="pt").input_ids[0]
    words = words[words != 100]
    words = words[words != 101]
    words = words[words != 102]
    return words.unique()


MODEL = BertForMaskedLM.from_pretrained("bert-base-uncased")
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

BAN_WORDS_PATH = Path(__file__).parent.parent.parent / 'data' / 'external' / 'toxic_words.txt'
BAN_WORDS = load_words(BAN_WORDS_PATH)
BAN_TOKENS = get_tokens(TOKENIZER, BAN_WORDS)


def censor(phrase, ban_words):
    words = word_tokenize(phrase.replace(' - ', ' [DASH] ').replace('-', ' - '))
    for bword in ban_words:
        for i, w in enumerate(words):
            if w.lower() == bword:
                words[i] = '[MASK]'

    return TreebankWordDetokenizer().detokenize(words).replace(' - ', '-').replace(' [DASH] ', ' - ')


def replace_mask(model, tokenizer, masked_phrase, ban_tokens=None):
    inputs = tokenizer(masked_phrase, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # do not retrieve tokens that we don't want
    logits[0, :, ban_tokens] = logits.min() - 1
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

    words = tokenizer.decode(predicted_token_id).split()
    for word in words:
        masked_phrase = masked_phrase.replace('[MASK]', word, 1)
    masked_phrase = masked_phrase[0].upper() + masked_phrase[1:]
    return masked_phrase


def detox(phrases: Union[str, Iterable[str]], return_mask: bool=False):
    if is_str := isinstance(phrases, str):
        phrases = [phrases]

    detoxed = []
    for phrase in phrases:
        detox_phrase = censor(phrase, BAN_WORDS)
        if not return_mask:
            detox_phrase = replace_mask(MODEL, TOKENIZER, detox_phrase, ban_tokens=BAN_TOKENS)
        detoxed.append(detox_phrase)

    if is_str:
        return detoxed[0]
    else:
        return detoxed


if __name__ == '__main__':
    sentences = ["What the hell is going on? I am very confused and pissed off!",
                 "I don't give a fuck.",
                 "I told you she is a bitch.",
                 "This guy is a dick!",
                 "This situation is literally fucked.",
                 "Stop shit-talking, you stupid motherfucker!",
                 "There is only one word to describe this - fuck...",
                 "Damn! It's fucking great!"]
    print('Censored phrases:')
    print('\n'.join(detox(sentences, True)))
    print('Detoxified phrases:')
    print('\n'.join(detox(sentences)))
