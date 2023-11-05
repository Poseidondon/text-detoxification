# Final Solution Report: CondBert
## Introduction
This solution is heavily based on the *naive approach* that is thoroughly described
in the [Solution Building Report](report_1.md).
Please, make sure you are familiar with it before proceeding.

This project ideas are inspired by [s-nlp condBERT](https://github.com/s-nlp/detox/tree/0ebaeab817957bb5463819bec7fa4ed3de9a26ee/emnlp2021/style_transfer/condBERT).
Some functions could be partially copied from there.

## Dataset
Instead of [dataset provided](../data/raw/filtered.tsv) by assignment,
[another dataset](https://github.com/s-nlp/detox/tree/0ebaeab817957bb5463819bec7fa4ed3de9a26ee/emnlp2021/data/train) was used,
because it's not parallel which is better for proposed model.

## Data preprocessing
*Code implementation is located in [build_vocab.py](../src/data/build_vocab.py)*

First task was to create a metric to determine word toxicity.
To accomplish that problem, the same methods as [here](https://github.com/s-nlp/detox/blob/0ebaeab817957bb5463819bec7fa4ed3de9a26ee/emnlp2021/style_transfer/condBERT/condbert_compile_vocab.ipynb) were used.
### Word coefficients
First method involves a **simple logistic regression** that was trained to detect toxic words as shown:
```python
pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
```
That makes the coefficients for each word represent its toxicity,
where higher coefficient means more toxic word.
### Token toxicity
In second method, for each token **number of occurrences** in a toxic and normal datasets is counted.
Then, token toxicity is got by: **token_toxicity = toxic_count / total_count**.
Later in the program *log odds ratio* is applied to those values.
### Code
When `build_vocab()` is called it generates 2 files: [token_toxicities.txt](../data/interim/vocab/token_toxicities.txt)
and [word2coef.pkl](../data/interim/vocab/word2coef.pkl).
Those files contain **token toxicities** and **word-to-coefficient dictionary** correspondingly.

## Model Specification
*Code implementation is located in [cond_BERT.py](../src/models/cond_BERT.py)*

In this section, model architecture is described. Text detoxification consists of 3 main steps:
1. **Mask toxic words**
2. **Apply Masked LM to get predictions**
3. **Select prediction according to similarity with input**

Let's cover those steps in more details.
### Masking
Masking is implemented by `CondBERT.mask(sentences, threshold, min_words)`.
After encoding using selected tokenizer, it consists of 3 steps:
1. **Assign toxicity to each token id.** This procedure is completed as follows:
    1. Initialize each id with **toxicity token** score.
    2. If token is present in **word2coef dictionary**,
    then calculate mean of step (1) and coefficient values.
    3. If token is a **suffix**, then its **toxicity = max(cur_toxicity, prev_toxicity)**,
    where **prev_toxicity** is a toxicity of the previous token.
2. **Calculate mask**, where each **element = toxicity > threshold**.
3. **If all elements of mask are False**, top **min_words** words with maximum toxicity are masked.

This function returns list of original ids and list of masks
### Translating
Translating is implemented by `CondBERT.translate(ids, mask, top_n)`.
It consists of 2 steps:
1. **Get Masked LM predictions**. Apply required changes on the logits:
    1. Penalize toxic words.
    2. Prohibit the same word as in **input**.
2. Select **top_n** best predictions and chose most similar with original token,
where similarity is defined as **cosine similarity** between words embeddings,
computed by **FastText** object.

### Detoxing
Detoxing using **CondBERT** is as simple as follows:
```python
>>> condBERT = load_condBERT()
>>> condBERT.detox(sentences)
```
Where `load_condBERT()` is a class builder to create object with default parameters and
`CondBERT.detox(sentences)` is a simple interface,
connecting `CondBERT.mask()` and `CondBERT.translate()`.

## Inference examples
As compared to [naive approach](report_1.md), **CondBERT** gives more sensible and accurate results:
```python
>>> condBERT.detox("Damn! It's fucking great!")
'oh! it's christ great!'
>>> condBERT.detox("Stop shit-talking, you stupid motherfucker!")
'stop christ - talking, you fool fatherick!'
>>> condBERT.detox("What the hell is going on? I am very confused and pissed off!")
'what the heck is going on? i am very confused and annoyed off!'
>>> condBERT.detox("There is only one word to describe this - fuck...")
'there is only one word to describe this - christ...'
```

## Evaluation
*Code implementation is located in [evaluation.py](../src/models/evaluation.py)*

A total of 2 evaluation metrics were used:
1. **Mean toxicity score**. Calculate mean toxicity score all tokens in text as in `CondBERT.mask()`.
2. **Sentence similarity**. To determine sentence similarity, [sentence transformers](https://pypi.org/project/sentence-transformers/)
module was used: this method uses pre-trained models to generate sentence embeddings.
Then *cosine similarity* of those embeddings is calculated.

Running [evaluation](../src/models/evaluation.py) on the [test dataset](../data/external/test) gave the following results:

|                     Dataset                     | Input toxicity | Output toxicity | Similarity |
|:-----------------------------------------------:|:--------------:|:---------------:|:----------:|
| [Normal](../data/external/test/test_normal.txt) |     0.111      |      0.106      |   0.981    |
|  [Toxic](../data/external/test/test_toxic.txt)  |     0.187      |      0.135      |   0.901    |

As demonstrated, **toxicity** is successfully reduced and semantic **similarity** is preserved.

## Results
Majority of the problems that [naive approach](report_1.md) had, were successfully solved and
generally speaking model tends to give adequate predictions.
However, model has its **disadvantages**, such as:
- Slow model initialization
- Difficulties with sentences that implies to be toxic not by a single word sense,
but by a semantic construction of a whole sentence.
