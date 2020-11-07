from model.AutoEncoder import AutoEncoder
from utilities import load_data, plot_losses
import torch.optim as optim
import numpy as np
import argparse
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='new_setup')
parser.add_argument('--max_name_length',
                    help='Max name generation length', type=int, default=20)
parser.add_argument('--batch_size', help='batch_size', type=int, default=200)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--RNN_hidden_size',
                    help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--word_embed_dim',
                    help='Word embedding size', type=int, default=16)
parser.add_argument(
    '--num_layers', help='number of rnn layer', type=int, default=3)
parser.add_argument('--num_epochs', help='epochs', type=int, default=10000)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--name_file', help='CSVs of names for training and testing',
                    type=str, default='data/first.csv')
parser.add_argument('--plot_dir', help='save dir', type=str, default='plot/')
parser.add_argument('--weight_dir', help='save dir',
                    type=str, default='weight/')
parser.add_argument('--save_every',
                    help='Number of iterations before saving', type=int, default=200)
parser.add_argument('--continue_train',
                    help='Continue training', type=bool, default=False)
args = parser.parse_args()

DEVICE = "cpu"


def fit(model, optimizer, X: torch.Tensor, X_lengths: torch.Tensor, Y: torch.Tensor):
    model.train()
    optimizer.zero_grad()
    probs, mu, sigmas = model.forward(X, X_lengths)
    loss = ELBO_loss(probs, Y, mu, sigmas)
    loss.backward()
    optimizer.step

    return loss


def test(model, X: torch.Tensor, X_lengths: torch.Tensor, Y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        probs, mu, sigmas = model.forward(X, X_lengths)
        loss = ELBO_loss(probs, Y, mu, sigmas)

    return loss


def ELBO_loss(Y_hat, Y, mu, logvar):
    length = Y.shape[1]
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0

    for i in range(length):
        loss += criterion(Y_hat[:, i, :], Y[:, i])

    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + KL_divergence


if args.continue_train:
    json_file = json.load(open(f'json/{args.name}.json', 'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(json_file)
    args = parser.parse_args(namespace=t_args)
    model.load(f'weight/{args.name}')
else:
    with open(f'json/{args.name}.json', 'wt') as f:
        json.dump(vars(args), f)

names_input, names_output, vocab, names_length, pad_idx = load_data(
    args.name_file)
model = AutoEncoder(vocab, pad_idx, args.max_name_length, DEVICE, args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

total_train_loss = []
total_test_loss = []

for epoch in range(args.num_epochs):
    train_loss = []
    test_loss = []

    num_train_data = int(len(names_input)*0.75)
    train_names_input = names_input[0:num_train_data]
    test_names_input = names_input[num_train_data:-1]

    train_names_output = names_output[0:num_train_data]
    test_names_output = names_output[num_train_data:-1]

    train_lengths = names_length[0:num_train_data]
    test_lengths = names_length[num_train_data:-1]

    for iteration in range(len(train_names_input)//args.batch_size):
        n = np.random.randint(len(train_names_input), size=args.batch_size)
        x = torch.LongTensor([train_names_input[i] for i in n]).to(DEVICE)
        y = torch.LongTensor([train_names_output[i] for i in n]).to(DEVICE)
        l = torch.LongTensor([train_lengths[i] for i in n]).to(DEVICE)

        cost = fit(model, optimizer, x, l, y)

        train_loss.append(cost.item())

        if iteration % args.save_every == 0:
            model.checkpoint(f'weight/{args.name}.path.tar')
            total_train_loss.append(np.mean(train_loss))
            plot_losses(total_train_loss, filename=f'{args.name}_train.png')
            train_loss = []

    for iteration in range(len(test_names_input)//args.batch_size):
        n = np.random.randint(len(test_names_input), size=args.batch_size)
        x = torch.LongTensor([test_names_input[i] for i in n]).to(DEVICE)
        y = torch.LongTensor([test_names_output[i] for i in n]).to(DEVICE)
        l = torch.LongTensor([test_lengths[i] for i in n]).to(DEVICE)

        cost = test(model, x, l, y)

        test_loss.append(cost.item())

        if iteration % args.save_every == 0:
            total_test_loss.append(np.mean(test_loss))
            plot_losses(total_test_loss, filename='test.png')
            test_loss = []
