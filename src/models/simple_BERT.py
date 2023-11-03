import torch
import sys
sys.path.append("..")

from transformers import BertTokenizer, BertForMaskedLM
from data.utils import load_words
from pathlib import Path
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Iterable, Union


def get_tokens(tokenizer, words: list):
    """
    Get unique tokens

    :param tokenizer: tokenizer
    :param words: list of words
    :return: tensor of unique tokens
    """

    words = tokenizer(' '.join(words), return_tensors="pt").input_ids[0]
    words = words[words != 100]
    words = words[words != 101]
    words = words[words != 102]
    return words.unique()


MODEL = BertForMaskedLM.from_pretrained("bert-base-uncased")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

BAN_WORDS_PATH = Path(__file__).parent.parent.parent / 'data' / 'external' / 'toxic_words.txt'
BAN_WORDS = load_words(BAN_WORDS_PATH)
BAN_TOKENS = get_tokens(TOKENIZER, BAN_WORDS)


def censor(phrase, ban_words):
    """
    Find toxic words and replace them with '[MASK]'

    :param phrase: phrase to censor
    :param ban_words: list of words that should be censored
    :return: censored phrase
    """

    words = word_tokenize(phrase.replace(' - ', ' [DASH] ').replace('-', ' - '))
    for bword in ban_words:
        for i, w in enumerate(words):
            if w.lower() == bword:
                words[i] = '[MASK]'

    return TreebankWordDetokenizer().detokenize(words).replace(' - ', '-').replace(' [DASH] ', ' - ')


def replace_mask(model, tokenizer, masked_phrase, ban_tokens=None):
    """
    Interface for Masked Language Models from transformers.

    :param model: Masked Language Model
    :param tokenizer: tokenizer
    :param masked_phrase: phrase to fill masks
    :param ban_tokens: prohibited tokens, model can't fill masks with those tokens
    :return: phrase with filled masks
    """

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


def detox(phrases: Union[str, Iterable[str]], return_mask: bool=False) -> Union[str, Iterable[str]]:
    """
    Detoxifies phrase or batch of phrases.
    Masks toxic words and then uses BERT Masked Language Model

    :param phrases: phrases to detoxify
    :param return_mask: if True, return censored phrase (with '[MASK]' instead of toxic words)
    :return: detoxified or censored phrases
    """

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
