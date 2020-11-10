import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, vocab: dict, sos_idx, pad_idx: int, device: str, args):
        super(AutoEncoder, self).__init__()
        self.sos_idx = sos_idx
        self.device = device
        self.encoder = Encoder(vocab, pad_idx, device, args)
        self.decoder = Decoder(vocab, pad_idx, device, args)
        self.to(device)

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor):
        max_len = X.shape[1]
        Z, mu, sigmas = self.encoder.forward(X, X_lengths)
        decoder_input = torch.LongTensor([self.sos_idx]).to(self.device)
        logits, probs = self.decoder.forward(decoder_input, Z, max_len)
        return logits, probs, mu, sigmas

    def test(self, X: torch.Tensor, X_lengths: torch.Tensor, max_output_len: int):
        Z, mu, sigmas = self.encoder.forward(X, X_lengths)
        decoder_input = torch.LongTensor([self.sos_idx]).to(self.device)
        logits, probs = self.decoder.forward(decoder_input, Z, max_output_len)
        return probs

    def checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


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

    def reparam_trick(self, mu: torch.Tensor, log_sigma: torch.Tensor):
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

        _, HC = self.lstm(X_pps, H)

        H = torch.flatten(HC[0].transpose(0, 1), 1, 2)
        mu = self.mu_mlp(H)
        sigmas = self.sigma_mlp(H)

        z = self.reparam_trick(mu, sigmas)

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
            embedding_dim=self.word_embed_dim,
            padding_idx=pad_idx
        )
        torch.nn.init.uniform_(self.word_embedding.weight)
        self.lstm = nn.LSTM(self.latent_size + self.word_embed_dim, self.hidden_size,
                            self.num_layers, batch_first=True)
        self.fc1 = NeuralNet(self.hidden_size, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input: torch.Tensor, Z: torch.Tensor, max_len: int, H=None):
        batch_size = Z.shape[0]

        if H is None:
            H = self.__init_hidden(batch_size)

        embeded_input = self.word_embedding(input)

        all_logits = None
        input = torch.cat((Z, embeded_input.repeat(
            (batch_size, 1))), dim=1).unsqueeze(1)

        for i in range(max_len):
            lstm_out, H = self.lstm(input, H)
            logits = self.fc1(lstm_out)
            probs = self.softmax(logits)
            sampled_chars = torch.argmax(probs, dim=2).squeeze(1)
            embeded_input = self.word_embedding(sampled_chars)
            input = torch.cat((Z, embeded_input), dim=1).unsqueeze(1)

            if i == 0:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), dim=1)

        return all_logits, self.softmax(all_logits)

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
