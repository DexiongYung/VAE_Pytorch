import torch
import torch.nn as nn


def init_cell_and_hidd(num_layers: int, batch_size: int, hidden_size: int, device: str):
    return (torch.zeros(num_layers, batch_size, hidden_size).to(device),
            torch.zeros(num_layers, batch_size, hidden_size).to(device))


class AutoEncoder(nn.Module):
    def __init__(self, device: str, args):
        super(AutoEncoder, self).__init__()
        self.sos_idx = args.sos_idx
        self.device = device
        self.encoder = Encoder(device, args)
        self.decoder = Decoder(device, args)
        self.to(device)

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor, max_len: int = None, is_teacher_force: bool = False):
        if max_len is None:
            max_len = X.shape[1]

        Z, mu, logit_sigma = self.encoder.forward(X, X_lengths)
        if is_teacher_force:
            logits, probs = self.decoder.forward(X, Z, X_lengths=X_lengths)
        else:
            decoder_input = torch.LongTensor([self.sos_idx]).to(self.device)
            logits, probs = self.decoder.forward(decoder_input, Z, max_len)
        return logits, probs, mu, logit_sigma

    def checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Encoder(nn.Module):
    def __init__(self, device: str, args):
        super(Encoder, self).__init__()
        self.vocab_size = len(args.vocab)
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
            padding_idx=args.pad_idx
        )
        self.lstm = nn.LSTM(self.word_embed_dim,
                            self.hidden_size, self.num_layers, batch_first=True)
        self.mu_mlp = NeuralNet(self.mlp_input_size, self.latent_size)
        self.sigma_mlp = NeuralNet(self.mlp_input_size, self.latent_size)

        # Sample added noise from normal distribution
        mu_tensor = torch.zeros((args.batch_size, args.latent_size)) + 0
        sd_tensor = torch.zeros((args.batch_size, args.latent_size)) + 1

        # Molecular SMILES VAE initialized embedding to uniform [-0.1, 0.1]
        torch.nn.init.uniform_(self.char_embedder.weight,  -0.1, 0.1)

    def reparam_trick(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        sd = torch.exp(0.5 * log_sigma)
        eps = 1e-2 * torch.randn_like(sd)
        sample = mu + (eps * sd)

        return sample

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor):
        batch_size = X.shape[0]
        HC = init_cell_and_hidd(self.num_layers, batch_size,
                                self.hidden_size, self.device)
        X_embed = self.char_embedder(X)
        # Pack padded sequence runs through LSTM
        X_pps = torch.nn.utils.rnn.pack_padded_sequence(
            X_embed, X_lengths, enforce_sorted=False, batch_first=True)

        # Forward through X_pps to get hidden and cell states
        _, HC = self.lstm(X_pps, HC)

        # Linear layers require batch first, batch in dim=1 in hidden states transpose required
        # flatten num layers and hidden state dims together
        H = torch.flatten(HC[0].transpose(0, 1), 1, 2)

        # Get mu and sigma
        mu = self.mu_mlp(H)
        logit_sigma = self.sigma_mlp(H)

        # Use reparam trick to sample latents
        z = self.reparam_trick(mu, logit_sigma)

        return z, mu, logit_sigma


class Decoder(nn.Module):
    def __init__(self, device: str, args):
        super(Decoder, self).__init__()
        self.vocab_size = len(args.vocab)
        self.hidden_size = args.RNN_hidden_size
        self.num_layers = args.num_layers
        self.word_embed_dim = args.word_embed_dim
        self.latent_size = args.latent_size
        self.device = device
        self.pad_idx = args.pad_idx
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=args.pad_idx
        )
        self.lstm = nn.LSTM(self.latent_size + self.word_embed_dim, self.hidden_size,
                            self.num_layers, batch_first=True)
        self.fc1 = NeuralNet(self.hidden_size, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=2)

        # Molecular SMILES VAE initialized embedding to uniform [-0.1, 0.1]
        torch.nn.init.uniform_(self.char_embedder.weight, -0.1, 0.1)

    def forward(self, X: torch.Tensor, Z: torch.Tensor, max_len: int = None, X_lengths: torch.Tensor = None):
        is_teacher_force = X_lengths is not None
        batch_size = Z.shape[0]
        HC = init_cell_and_hidd(self.num_layers, batch_size,
                                self.hidden_size, self.device)

        # Embed input
        embeded_input = self.char_embedder(X)

        if is_teacher_force:
            max_len = X.shape[1]
            input = torch.cat((Z.unsqueeze(1).repeat(
                (1, max_len, 1)), embeded_input), dim=2)
            # Pack sequence runs entire sequence through LSTM
            X_ps = torch.nn.utils.rnn.pack_padded_sequence(
                input, X_lengths, enforce_sorted=False, batch_first=True)
            out_ps, H = self.lstm(X_ps, HC)
            # Convert from pack output to tensor form
            lstm_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out_ps, batch_first=True, total_length=max_len, padding_value=self.pad_idx)
            # Reshape because LL is (batch, features)
            lstm_outs = lstm_outs.reshape((batch_size * max_len, -1))
            fc1_outs = self.fc1(lstm_outs)
            all_logits = fc1_outs.reshape((batch_size, max_len, -1))
        else:
            all_logits = None

            # All inputs should have Z appended to input
            input = torch.cat((Z, embeded_input.repeat(
                (batch_size, 1))), dim=1).unsqueeze(1)

            for i in range(max_len):
                lstm_out, HC = self.lstm(input, HC)
                logits = self.fc1(lstm_out)
                max_chars = torch.argmax(logits, dim=2).squeeze(1)
                embeded_input = self.char_embedder(max_chars)
                input = torch.cat((Z, embeded_input), dim=1).unsqueeze(1)

                # Save logits for back prop
                if i == 0:
                    all_logits = logits
                else:
                    all_logits = torch.cat((all_logits, logits), dim=1)

        # Return logits and probs
        return all_logits, self.softmax(all_logits)


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(NeuralNet, self).__init__()
        self.ll = nn.Linear(input_size, output_size)
        self.selu = nn.SELU()

        # Molecular VAE initializes linear layer using Xavier
        torch.nn.init.xavier_uniform_(self.ll.weight)

    def forward(self, X: torch.Tensor):
        X = self.ll(X)
        return self.selu(X)
