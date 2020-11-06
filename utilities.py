import numpy as np
import pandas as pd
import collections
import string


def load_data(n, SOS: str = '[', EOS: str = ']', PAD: str = '$'):
    df = pd.read_csv(n)
    names = df['name'].tolist()
    seq_length = df['name'].str.len().max() + 1
    chars = set(df['name'].sum())
    vocab = dict(zip(chars, range(len(chars))))

    len_chars = len(chars)

    pad_idx = len_chars + 2
    vocab[SOS] = len_chars + 1
    vocab[EOS] = len_chars + 2
    vocab[PAD] = len_chars + 3

    names_length = np.array([len(n)+1 for n in names])
    names_input = [(SOS+s).ljust(seq_length, PAD) for s in names]
    names_output = [(s+EOS).ljust(seq_length, PAD) for s in names]
    names_input = np.array([np.array(list(map(vocab.get, s)))
                            for s in names_input])
    names_output = np.array([np.array(list(map(vocab.get, s)))
                             for s in names_output])

    return names_input, names_output, vocab, names_length, pad_idx
