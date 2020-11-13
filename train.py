from model.AutoEncoder import AutoEncoder
from utilities import *
from os import path
import torch.optim as optim
import numpy as np
import argparse
import torch
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='new_eps')
parser.add_argument('--max_name_length',
                    help='Max name generation length', type=int, default=40)
parser.add_argument('--batch_size', help='batch_size', type=int, default=100)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--RNN_hidden_size',
                    help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--word_embed_dim',
                    help='Word embedding size', type=int, default=200)
parser.add_argument(
    '--num_layers', help='number of rnn layer', type=int, default=3)
parser.add_argument('--num_epochs', help='epochs', type=int, default=1000)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument(
    '--percent_train', help='Percent of the data used for training', type=float, default=0.75)
parser.add_argument('--name_file', help='CSVs of names for training and testing',
                    type=str, default='data/first.csv')
parser.add_argument('--weight_dir', help='save dir',
                    type=str, default='weight/')
parser.add_argument('--save_every',
                    help='Number of iterations before saving', type=int, default=200)
parser.add_argument('--continue_train',
                    help='Continue training', type=bool, default=False)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def fit(model: AutoEncoder, optimizer, X: torch.Tensor, X_lengths: torch.Tensor, Y: torch.Tensor):
    model.train()
    optimizer.zero_grad()
    logits, probs, mu, sigmas = model.forward(
        X, X_lengths, is_teacher_force=True)
    loss = ELBO_loss(logits, Y, mu, sigmas)
    loss.backward()
    optimizer.step

    return loss


def test(model: AutoEncoder, X: torch.Tensor, X_lengths: torch.Tensor, Y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits, probs, mu, sigmas = model.forward(X, X_lengths)
        loss = ELBO_loss(logits, Y, mu, sigmas)

    return loss


def ELBO_loss(Y_hat: torch.Tensor, Y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    batch_size = Y.shape[0]
    length = Y.shape[1]
    CE_loss_sum = 0

    for i in range(length):
        CE_loss_sum += criterion(Y_hat[:, i, :], Y[:, i])

    latent_entropy = torch.sum(0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))

    return CE_loss_sum - latent_entropy


# Generate number to char dict, char to number dict, sos, pad and eos idx, put all names into a list
names, name_probs, c_to_n_vocab, n_to_c_vocab, sos_idx, pad_idx, eos_idx = load_data(
    args.name_file)

if args.continue_train:
    json_file = json.load(open(f'json/{args.name}.json', 'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(json_file)
    args = parser.parse_args(namespace=t_args)
    model.load(f'{args.weight_dir}/{args.name}')
else:
    args.vocab = c_to_n_vocab
    args.sos_idx = sos_idx
    args.eos_idx = eos_idx
    args.pad_idx = pad_idx

    if not path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)

    if not path.exists('json'):
        os.mkdir('json')

    with open(f'json/{args.name}.json', 'w') as f:
        json.dump(vars(args), f)

model = AutoEncoder(DEVICE, args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

total_train_loss = []
total_test_loss = []

# Run through data for num_epochs
for epoch in range(args.num_epochs):
    train_loss = []
    test_loss = []

    num_train_data = int(len(names)*args.percent_train)
    num_test_data = len(names) - num_train_data

    SOS = n_to_c_vocab[sos_idx]
    EOS = n_to_c_vocab[eos_idx]
    PAD = n_to_c_vocab[pad_idx]

    # Train on num_train_data with batch size
    for iteration in range(num_train_data//args.batch_size):
        train_names_input, train_names_output, train_lengths = create_batch(
            names, name_probs, args.batch_size, c_to_n_vocab, SOS, PAD, EOS)
        # x = name in index form inputs, y = name in index form labels, l = name lengths
        x = torch.LongTensor(train_names_input).to(DEVICE)
        y = torch.LongTensor(train_names_output).to(DEVICE)
        l = torch.LongTensor(train_lengths).to(DEVICE)

        cost = fit(model, optimizer, x, l, y)

        train_loss.append(cost.item())

        if iteration % args.save_every == 0:
            model.checkpoint(f'{args.weight_dir}/{args.name}.path.tar')
            total_train_loss.append(np.mean(train_loss))
            plot_losses(total_train_loss, filename=f'{args.name}_train.png')
            train_loss = []

    # Run test loss for eval
    for iteration in range(num_test_data//args.batch_size):
        test_names_input, test_names_output, test_lengths = create_batch(
            names, name_probs, args.batch_size, c_to_n_vocab, SOS, PAD, EOS)
        x = torch.LongTensor(test_names_input).to(DEVICE)
        y = torch.LongTensor(test_names_output).to(DEVICE)
        l = torch.LongTensor(test_lengths).to(DEVICE)

        cost = test(model, x, l, y)

        test_loss.append(cost.item())

        if iteration % args.save_every == 0:
            total_test_loss.append(np.mean(test_loss))
            plot_losses(total_test_loss, filename=f'{args.name}_test.png')
            test_loss = []
