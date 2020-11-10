import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, vocab: dict, sos_idx, pad_idx: int, device: str, args):
        super(AutoEncoder, self).__init__()
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
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
        self.vocab_size = len(vocab)
        self.hidden_size = args.RNN_hidden_size
        self.num_layers = args.num_layers
        self.word_embed_dim = args.word_embed_dim
        self.latent_size = args.latent_size
        # MLPs should take the hidden state size, which is num_layers * hidden_size
        self.mlp_input_size = self.num_layers * self.hidden_size
        self.device = device
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=pad_idx
        )
        # Molecular SMILES VAE initialized embedding to uniform [-0.1, 0.1]
        torch.nn.init.uniform_(self.char_embedder.weight,  -0.1, 0.1)
        self.lstm = nn.LSTM(self.word_embed_dim,
                            self.hidden_size, self.num_layers, batch_first=True)
        self.mu_mlp = NeuralNet(self.mlp_input_size, self.latent_size)
        self.sigma_mlp = NeuralNet(self.mlp_input_size, self.latent_size)

    def reparam_trick(self, mu: torch.Tensor, log_sigma: torch.Tensor, m: int = 0, d: int = 1):
        batch_size = mu.shape[0]
        latent_size = mu.shape[1]
        sd = torch.exp(0.5 * log_sigma)
        # Molecular VAE multiplied std by sample from normal with SD 1 and mu 0 and samples unique value for each latent index
        mu_tensor = torch.zeros((batch_size, latent_size)) + m
        sd_tensor = torch.zeros((batch_size, latent_size)) + d
        eps = torch.distributions.Normal(mu_tensor, sd_tensor).sample()
        sample = mu + (eps * sd)

        return sample

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor, H=None):
        batch_size = X.shape[0]

        if H is None:
            H = self.__init_hidden(batch_size)

        X_embed = self.char_embedder(X)
        # Pack padded sequence
        X_pps = torch.nn.utils.rnn.pack_padded_sequence(
            X_embed, X_lengths, enforce_sorted=False, batch_first=True)

        # Forward through X_pps to get hidden and cell states
        _, HC = self.lstm(X_pps, H)

        # Linear layers require batch first, batch in dim=1 in hidden states, then flatten num layers and hidden state dims together
        H = torch.flatten(HC[0].transpose(0, 1), 1, 2)

        # Get mu and sigma
        mu = self.mu_mlp(H)
        sigmas = self.sigma_mlp(H)

        # Use reparam trick to sample latents
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
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=pad_idx
        )
        # Molecular SMILES VAE initialized embedding to uniform [-0.1, 0.1]
        torch.nn.init.uniform_(self.char_embedder.weight, -0.1, 0.1)
        self.lstm = nn.LSTM(self.latent_size + self.word_embed_dim, self.hidden_size,
                            self.num_layers, batch_first=True)
        self.fc1 = NeuralNet(self.hidden_size, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input: torch.Tensor, Z: torch.Tensor, max_len: int, H=None):
        batch_size = Z.shape[0]

        if H is None:
            H = self.__init_hidden(batch_size)

        # Embed input
        embeded_input = self.char_embedder(input)

        all_logits = None
        # All inputs should have Z appended to input
        input = torch.cat((Z, embeded_input.repeat(
            (batch_size, 1))), dim=1).unsqueeze(1)

        for i in range(max_len):
            lstm_out, H = self.lstm(input, H)
            logits = self.fc1(lstm_out)
            # Softmax the vocab dimension
            probs = self.softmax(logits)

            # Sample argmax char and generate next input
            sampled_chars = torch.argmax(probs, dim=2).squeeze(1)
            embeded_input = self.char_embedder(sampled_chars)
            input = torch.cat((Z, embeded_input), dim=1).unsqueeze(1)

            # Save logits for back prop
            if i == 0:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), dim=1)

        # Return logits and probs
        return all_logits, self.softmax(all_logits)

    def __init_hidden(self, batch_size: int):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(NeuralNet, self).__init__()
        self.ll = nn.Linear(input_size, output_size)
        # Molecular VAE initializes linear layer using Xavier
        torch.nn.init.xavier_uniform_(self.ll.weight)
        # Trying out SELU, Molecular VAE doesn't use any activations
        # self.selu = nn.SELU()

    def forward(self, X: torch.Tensor):
        X = self.ll(X)
        return X
        # return self.selu(X)
