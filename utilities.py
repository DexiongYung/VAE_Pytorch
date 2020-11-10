import numpy as np
import pandas as pd
import collections
import string
import matplotlib.pyplot as plt
from random import randrange


def plot_losses(losses, folder: str = "plot", filename: str = "checkpoint.png"):
    x = list(range(len(losses)))
    plt.plot(x, losses, 'b--', label="Unsupervised Loss")
    plt.title("Loss Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()


def load_data(n: str, SOS: str = '[', EOS: str = ']', PAD: str = '$'):
    df = pd.read_csv(n).iloc[:10000]
    names = df['name'].tolist()
    chars = string.ascii_letters + SOS + EOS + PAD
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))

    pad_idx = c_to_n_vocab[PAD]
    sos_idx = c_to_n_vocab[SOS]
    eos_idx = c_to_n_vocab[EOS]

    return names, c_to_n_vocab, n_to_c_vocab, sos_idx, pad_idx, eos_idx


def create_batch(all_names: list, batch_size: int, vocab: dict, SOS: str, PAD: str, EOS: str):
    names = np.random.choice(all_names, batch_size, replace=False)
    seq_length = len(max(names, key=len)) + 1

    names_length = np.array([len(n)+1 for n in names])
    names_input = [(SOS+s).ljust(seq_length, PAD) for s in names]
    names_output = [(s+EOS).ljust(seq_length, PAD) for s in names]
    names_input = np.array([np.array(list(map(vocab.get, s)))
                            for s in names_input])
    names_output = np.array([np.array(list(map(vocab.get, s)))
                             for s in names_output])

    return names_input, names_output, names_length


def noise_name(name: str, allowed_chars: str):
    len_allowed = len(allowed_chars)
    ret = ''
    for c in name:
        action = np.random.choice(4)

        if action == 0:
            ret += c
        elif action == 1:
            ret += c
            add_idx = np.random.choice(len_allowed)
            ret += allowed_chars[add_idx]
        elif action == 2:
            sub_idx = np.random.choice(len_allowed)
            ret += allowed_chars[sub_idx]

    return ret
