import pandas as pd

from pathlib import Path


def load_references(path=None):
    if path is None:
        path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'filtered.tsv'

    df = pd.read_csv(path, sep='\t')
    return df['reference'].to_list()


path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'filtered.tsv'
df = pd.read_csv(path, sep='\t')
a = 500
print('Reference:', df['reference'][a])
print('Translation:', df['translation'][a])
