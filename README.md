### Boris Zarubin
### b.zarubin@innopolis.university
### B21-DS-01

# Text Detoxification
Text Detoxification Task is a process of transforming the text with toxic style into
the text with the same meaning but with neutral style.

## Installation
```shell
git clone https://github.com/Poseidondon/text-detoxification
pip install -r requirements.txt
```
## Data preprocessing
You can skip that part if you want to use [pre-built vocab](data/interim/vocab).

Python
```python
from src.data.build_vocab import build_vocab
build_vocab()
```

CLI
```shell
cd src/data
python make_dataset.py
```

## Inference
Python
```python
from src.models.cond_BERT import load_condBERT
condBERT = load_condBERT()
toxic_example = "I like that f***ing show!"
print(condBERT(toxic_example))
```

CLI
```shell
cd src/models
python predict_model.py -f input.txt -o out.txt
# or
python predict_model.py -s "I like that f***ing show!"
```

## Evaluation
CLI
```shell
cd src/models
python evaluation.py
```
