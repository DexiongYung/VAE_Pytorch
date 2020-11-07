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


def load_data(n, SOS: str = '[', EOS: str = ']', PAD: str = '$'):
    df = pd.read_csv(n).iloc[:10000]
    names = df['name'].tolist()
    seq_length = len(max(names, key=len)) + 1
    chars = string.ascii_letters + SOS + EOS + PAD
    vocab = dict(zip(chars, range(len(chars))))

    pad_idx = vocab[PAD]

    names_length = np.array([len(n)+1 for n in names])
    names_input = [(SOS+s).ljust(seq_length, PAD) for s in names]
    names_output = [(s+EOS).ljust(seq_length, PAD) for s in names]
    names_input = np.array([np.array(list(map(vocab.get, s)))
                            for s in names_input])
    names_output = np.array([np.array(list(map(vocab.get, s)))
                             for s in names_output])

    return names_input, names_output, vocab, names_length, pad_idx


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
