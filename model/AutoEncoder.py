import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, vocab: dict, pad_idx: int, max_length: int, device: str, args):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(vocab, pad_idx, device, args)
        self.decoder = Decoder(vocab, pad_idx, device, args)
        self.to(device)

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor):
        Z, mu, sigmas = self.encoder.forward(X, X_lengths)
        probs = self.decoder.forward(Z, X, X_lengths)
        return probs, mu, sigmas

    def checkpoint(self, path: str):
        torch.save(self.state_dict(), path)


class Encoder(nn.Module):
    def __init__(self, vocab: dict, pad_idx: int, device: str, args):
        super(Encoder, self).__init__()
        self.input_size = len(vocab)
        self.hidden_size = args.RNN_hidden_size
        self.num_layers = args.num_layers
        self.word_embed_dim = args.word_embed_dim
        self.latent_size = args.latent_size
        self.mlp_input_size = self.num_layers * self.hidden_size
        self.device = device

        self.word_embedding = nn.Embedding(
            num_embeddings=self.input_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=pad_idx
        )
        torch.nn.init.uniform_(self.word_embedding.weight)
        self.lstm = nn.LSTM(self.word_embed_dim,
                            self.hidden_size, self.num_layers, batch_first=True)
        self.mu_mlp = NeuralNet(self.mlp_input_size, self.latent_size)
        self.sigma_mlp = NeuralNet(self.mlp_input_size, self.latent_size)

    def reparameterize_trick(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)

        return sample

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor, H=None):
        batch_size = X.shape[0]

        if H is None:
            H = self.__init_hidden(batch_size)

        X_embed = self.word_embedding(X)
        X_pps = torch.nn.utils.rnn.pack_padded_sequence(
            X_embed, X_lengths, enforce_sorted=False, batch_first=True)

        _, H = self.lstm(X_pps, H)

        H0 = torch.flatten(H[0].transpose(0, 1), 1, 2)
        mu = self.mu_mlp(H0)
        sigmas = self.sigma_mlp(H0)

        z = self.reparameterize_trick(mu, sigmas)

        return z, mu, sigmas

    def __init_hidden(self, batch_size: int, dtype=torch.float32):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=dtype).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=dtype).to(self.device))


class Decoder(nn.Module):
    def __init__(self, vocab: dict, pad_idx: int, device: str, args):
        super(Decoder, self).__init__()
        self.vocab_size = len(vocab)
        self.hidden_size = args.RNN_hidden_size
        self.num_layers = args.num_layers
        self.word_embed_dim = args.word_embed_dim
        self.latent_size = args.latent_size
        self.device = device

        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=args.word_embed_dim,
            padding_idx=pad_idx
        )
        torch.nn.init.uniform_(self.word_embedding.weight)
        self.lstm_input = self.word_embed_dim + self.latent_size
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size,
                            self.num_layers, batch_first=True)
        self.fc1 = NeuralNet(self.hidden_size, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, Z: torch.Tensor, X: torch.Tensor, X_lengths: torch.Tensor, H=None):
        batch_size = X.shape[0]
        sequence_length = X.shape[1]

        if H is None:
            H = self.__init_hidden(batch_size)

        X_embed = self.word_embedding(X)

        probs = None

        for i in range(sequence_length):
            input = torch.cat((X_embed[:, i, :], Z), dim=1).unsqueeze(1)
            lstm_out, H = self.lstm(input, H)
            fc1_out = self.fc1(lstm_out)

            if i == 0:
                probs = self.softmax(fc1_out)
            else:
                probs = torch.cat((probs, self.softmax(fc1_out)), dim=1)

        return probs

    def __init_hidden(self, batch_size: int):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(NeuralNet, self).__init__()
        self.ll = nn.Linear(input_size, output_size)
        torch.nn.init.xavier_uniform_(self.ll.weight)
        self.selu = nn.SELU()

    def forward(self, X: torch.Tensor):
        X = self.ll(X)
        return self.selu(X)
