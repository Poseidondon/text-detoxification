# Solution Building Report
## Baseline: Masked Language Modeling
### Idea
Replace toxic words, using *Masked Language Model*.
### Algorithm
1. Iterate through words and mask them if they are part of a set of toxic words
2. Use modified [transformers.BertForMaskedLM](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM)
to predict missing words, such that it does not predict toxic words.
### Code
Code for this algorithm is contained at [simple_BERT.py](../src/models/simple_BERT.py).
Run it using:
```python
from simple_BERT import detox
detox("Toxic text here", return_mask=False)
```
### Examples
Here we can see, how phrase is censored and then missing words are predicted.
```python
>>> detox("Damn! It's fucking great!", return_mask=True)
'[MASK]! It's [MASK] great!'
>>> detox("Damn! It's fucking great!")
'Oh! It's so great!'

>>> detox("Stop shit-talking, you stupid motherfucker!")
'Stop trash-talking, you stupid fool!'
>>> detox("What the hell is going on? I am very confused and pissed off!")
'What the heck is going on? I am very confused and run off!'
>>> detox("There is only one word to describe this - fuck...")
'There is only one word to describe this - "...'
```
### Problems
- It's difficult to keep a set of all toxic words, since some of them can be used in a positive way.
Also, it's hard to store all the toxic words in practice.
- Cases when replacing toxic word would inevitably lead to sense loss.
Such cases require to reconstruct the full sentence.
- Toxic words meaning does not affect prediction. Thus, sentence content may be twisted.

To overcome these problems, it was decided to develop an improved model.
## Results
...
