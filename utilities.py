import numpy as np
import pandas as pd
import collections
import string
import matplotlib.pyplot as plt

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
    df = pd.read_csv(n)
    names = df['name'].tolist()
    seq_length = df['name'].str.len().max() + 1
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
