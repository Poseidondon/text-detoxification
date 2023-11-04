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
1. It's difficult to keep a set of all toxic words, since some of them can be used in a positive way.
Also, it's hard to store all the toxic words in practice.
2. Cases when replacing toxic word would inevitably lead to sense loss.
Such cases require to reconstruct the full sentence.
3. Toxic words meaning is not preserved during prediction. Thus, sentence content may be twisted.
4. Sometimes complex words with suffixes are not masked correctly.

To overcome these problems, it was decided to apply following hypotheses.

## Hypothesis 1: Create a metric to determine word toxicity
### Idea
Toxicity metric used in project is an aggregated score consisting of 2 sub-scores:
1. **Token toxicity.** Each MLM token ia assigned a toxicity.
2. **Word coefficients.** Toxicity score will be mapped to each word.

To get final toxicity, those scores would be combined (for example mean).

Such an approach would efficiently solve **problem (1)**.

## Hypothesis 2: Preserve input and prediction similarity
### Idea
When choosing a new word, using MLM similarity with original should be preserved.
To achieve that goal, it is proposed to select *top n* predictions from MLM and
then select most similar with input.

Such an approach would efficiently solve **problem (3)**.

## Hypothesis 3: Mask suffixes individually
### Idea
In a previously described naive approach it is only possible to mask the whole word.
To achieve proper masking of a complex words it is proposed to mask each suffix individually,
and if some part of the word occurs to be toxic, then the entire word would be marked as such.

Such an approach would efficiently solve **problem (4)**.

## Results
Naive approach described in this report has its problems,
however the majority of them appear to be solvable, except for **problem (2)**.
Thus, it was decided to develop a solution on that base that would overcome those issues.

A new solution is implemented in [cond_BERT.py](../src/models/cond_BERT.py)
and thoroughly described in [Final Solution Report](report_2.md).
