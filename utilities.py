import numpy as np
import pandas as pd
import collections
import string
import torch
import matplotlib.pyplot as plt
import os
from random import randrange
from os import path


def plot_losses(losses, folder: str = "plot", filename: str = "checkpoint.png"):
    if not path.exists(folder):
        os.mkdir(folder)

    x = list(range(len(losses)))
    plt.plot(x, losses, 'b--', label="Unsupervised Loss")
    plt.title("Loss Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()


def load_data(n: str, SOS: str = '[', EOS: str = ']', PAD: str = '$'):
    df = pd.read_csv(n)
    names = df['name'].tolist()
    name_probs = df['probs'].tolist()
    chars = string.ascii_letters + SOS + EOS + PAD
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))

    pad_idx = c_to_n_vocab[PAD]
    sos_idx = c_to_n_vocab[SOS]
    eos_idx = c_to_n_vocab[EOS]

    return names, name_probs, c_to_n_vocab, n_to_c_vocab, sos_idx, pad_idx, eos_idx


def create_batch(all_names: list, probs_list: list, batch_size: int, vocab: dict, SOS: str, PAD: str, EOS: str):
    distribution = torch.distributions.Categorical(
        torch.FloatTensor(probs_list))
    names = [all_names[distribution.sample().item()]
             for i in range(batch_size)]

    seq_length = len(max(names, key=len)) + 1

    # Names length should be length of the name + SOS xor EOS
    names_length = [len(n)+1 for n in names]
    names_input = [(SOS+s).ljust(seq_length, PAD) for s in names]
    names_input = [list(map(vocab.get, s)) for s in names_input]
    names_output = [(s+EOS).ljust(seq_length, PAD) for s in names]
    names_output = [list(map(vocab.get, s))for s in names_output]

    return names_input, names_output, names_length
